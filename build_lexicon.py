# -*- coding: utf-8 -*-
"""
build_lexicon.py — ساختِ واژه‌نامهٔ عروضی از پیکرهٔ *برچسب‌خوردهٔ* P3R

ایده
────
موتور نمی‌داند واژه در شعرِ کلاسیک چه وزنی می‌گیرد، پس برای هر مصراع هزاران
خوانش می‌سازد و وزنِ غلط هم تصادفاً «تطبیقِ کامل» پیدا می‌کند.
ولی وقتی وزنِ بیت را *از پیش* بدانیم، هم‌ترازیِ وزن‌مقید می‌گوید هر واژه
**در عمل** چه رشتهٔ وزنی گرفته است. روی ۱.۳ میلیون بیت این می‌شود یک
واژه‌نامهٔ آماریِ تلفظِ عروضی.

★ چرا این با bootstrap_lexicon.py فرق دارد
   آن یکی وزن را با حدسِ خودِ موتور برچسب می‌زند و بعد روی همان حدس‌ها
   آموزش می‌بیند — دوری است و خطاهای موتور را تقویت می‌کند.
   این یکی از برچسبِ *انسانیِ* P3R استفاده می‌کند: غیرِدوری.

★ تفکیکِ آموزش/آزمون
   فقط روی بخشِ 'train' آموزش می‌بیند (p3r.split_of)، تا eval_p3r.py بتواند
   دقتِ صادقانه بسنجد.

اجرا
────
    # دانلودِ پیکره (۲۰۸ مگابایت، یک‌بار)
    curl -L -o P3R.csv https://media.githubusercontent.com/media/m-shahrestani/Prosody-Recognition-in-Persian-Poetry/master/dataset/P3R.csv

    python build_lexicon.py P3R.csv                 # کاملِ پیکره
    python build_lexicon.py P3R.csv --limit 200000  # تندتر، کم‌دقت‌تر
    python build_lexicon.py P3R.csv --workers 8

هر ۲۰۰۰۰ بیت یک checkpoint روی lexicon.json نوشته می‌شود؛ می‌شود وسطِ کار
متوقفش کرد و همان را برداشت.
"""
import sys, os, json, time, argparse
import multiprocessing as mp
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r

MIN_COUNT = 2          # واژه‌ای که کمتر از این دیده شده نویز است
BATCH = 500


def _init():
    """در هر پردازه: موتور را بدونِ واژه‌نامه بار کن (اینجا فقط align لازم است،
    و واژه‌نامهٔ قدیمی نباید در ساختِ واژه‌نامهٔ تازه دخالت کند)."""
    import arooz
    arooz.LEXICON = {}
    global _PAT_OF
    _PAT_OF = {ark: pat for _, ark, pat, _ in arooz.METERS}


def _batch(items):
    """یک دسته بیت → (شمارشِ واژه‌ها، تعدادِ مصراعِ موفق، ناموفق)"""
    import arooz
    lex = defaultdict(Counter)
    ok = bad = 0
    for v1, v2, ark in items:
        pat = _PAT_OF.get(ark)
        if not pat:
            continue
        for mesra in (v1, v2):
            if not mesra:
                continue
            try:
                a = arooz.align(mesra, pat)
            except Exception:
                a = None
            if not a:
                bad += 1
                continue
            ok += 1
            # align → [(واژه، رشتهٔ وزن), ...]   (واژه‌های وصل‌دار خودش انداخته)
            for w, wt in a:
                lex[w][wt] += 1
        # کشِ موتور بی‌کران است و رم را می‌خورد
        if len(arooz._EXP_CACHE) > 4000:
            arooz._EXP_CACHE.clear()
    return {w: dict(c) for w, c in lex.items()}, ok, bad


def _chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def save(lex, out):
    final = {w: dict(c) for w, c in lex.items() if sum(c.values()) >= MIN_COUNT}
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)
    os.replace(tmp, out)          # اتمیک: checkpoint نصفه‌کاره فایل را خراب نکند
    return len(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lexicon.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    print(f">> پیکره: {args.csv}")
    print(f">> خروجی: {args.out}")
    print(f">> {args.workers} پردازه  |  فقط بخشِ train (۹۵٪)", flush=True)

    lex = defaultdict(Counter)
    n_ok = n_bad = n_bayt = 0
    t0 = time.time()
    last_save = 0

    src = p3r.rows(args.csv, split="train", limit=args.limit)
    with mp.Pool(args.workers, initializer=_init) as pool:
        for part, ok, bad in pool.imap_unordered(_batch, _chunks(src, BATCH)):
            for w, c in part.items():
                lex[w].update(c)
            n_ok += ok
            n_bad += bad
            n_bayt += BATCH
            if n_bayt - last_save >= 20000:
                last_save = n_bayt
                nw = save(lex, args.out)
                el = time.time() - t0
                rate = n_bayt / el
                print(f"  بیت={n_bayt:>8}  مصراعِ هم‌ترازشده={n_ok:>8}  "
                      f"ناموفق={n_bad:>7}  واژه={nw:>7}   "
                      f"({el/60:.1f} دقیقه، {rate:.0f} بیت/ثانیه)", flush=True)

    nw = save(lex, args.out)
    el = time.time() - t0
    print()
    print(f"✅ {args.out}: {nw} واژه")
    print(f"   {n_ok} مصراعِ هم‌ترازشده، {n_bad} ناموفق  |  {el/60:.1f} دقیقه")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
