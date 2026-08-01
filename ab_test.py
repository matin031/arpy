# -*- coding: utf-8 -*-
"""
ab_test.py — مقایسهٔ *جفتی* دو پیکربندی روی همان بیت‌ها، با بازهٔ اطمینان.

چرا این فایل لازم شد
────────────────────
یک‌بار بر پایهٔ «۳۳۵ در برابر ۳۳۶ از ۴۰۰» نتیجه گرفتم واژه‌نامه بی‌اثر است و
حذفش کردم. آن استنتاج غلط بود: خطای نمونه‌گیریِ ۴۰۰ بیت حدودِ ±۱.۸ واحد است،
پس آن اندازه‌گیری اصلاً نمی‌توانست «صفر اثر» را از «۲ واحد اثر» تشخیص بدهد.
روی ۲۵۰۰ بیت جهتِ اختلاف برعکس شد.

سه چیز این‌جا فرق می‌کند:
  ۱) مقایسهٔ جفتی: هر دو پیکربندی *همان* بیت‌ها را می‌بینند، پس سختیِ نمونه
     حذف می‌شود. آزمونِ McNemar روی بیت‌هایی که نتیجه‌شان فرق کرده.
  ۲) بازهٔ اطمینان، و اگر صفر داخلش بود «نمی‌دانیم» گفته می‌شود نه «بی‌اثر است».
  ۳) ★ نقطهٔ‌ذخیره: این محیط کانتینر را وسطِ کار راه‌اندازیِ دوباره می‌کند و
     اجرای بلند بی‌نتیجه از بین می‌رود (پنج بار). هر بلوک روی دیسک ذخیره
     می‌شود و اجرای دوباره از همان‌جا ادامه می‌دهد.

هر دو پیکربندی برای هر بیت در *یک* پردازه اجرا می‌شوند: تقطیع میانشان مشترک
است و کشِ گسترش، فراخوانیِ دوم را ارزان می‌کند.

    python ab_test.py /tmp/p3r/P3R.csv --n 6000
"""
import sys, os, json, argparse, math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r


def _init():
    import arooz
    global _FULL_LEX
    _FULL_LEX = arooz.LEXICON


def _one(item):
    """→ (درست‌بودن با واژه‌نامه، درست‌بودن بدونِ آن)"""
    v1, v2, gold = item
    import arooz
    try:
        arooz.LEXICON = _FULL_LEX
        a = arooz.detect(v1, v2)[0][0]["ark"] == gold
        arooz.LEXICON = {}
        b = arooz.detect(v1, v2)[0][0]["ark"] == gold
    except Exception:
        return None
    finally:
        arooz.LEXICON = _FULL_LEX
    return (a, b)


def stats(pairs, labels):
    n = len(pairs)
    a_ok = sum(1 for x, _ in pairs if x)
    b_ok = sum(1 for _, y in pairs if y)
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    print(f"\n{labels[0]:<16}: {a_ok}/{n} = {100*a_ok/n:.1f}%")
    print(f"{labels[1]:<16}: {b_ok}/{n} = {100*b_ok/n:.1f}%")
    print(f"{'اختلاف':<16}: {100*(a_ok-b_ok)/n:+.2f} واحد")
    print(f"\nفقط {labels[0]} درست : {a_only} بیت")
    print(f"فقط {labels[1]} درست : {b_only} بیت")

    d = a_only + b_only
    if d == 0:
        print("\nهیچ بیتی فرق نکرد — دو پیکربندی هم‌ارزند.")
        return
    se = math.sqrt(d) / n
    diff = (a_only - b_only) / n
    lo, hi = 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)
    z = (abs(a_only - b_only) - 1) / math.sqrt(d)
    print(f"\nاختلاف، بازهٔ اطمینانِ ۹۵٪: {100*diff:+.2f} واحد  [{lo:+.2f} , {hi:+.2f}]")
    print(f"McNemar z = {z:.2f}")
    if lo <= 0 <= hi:
        need = math.ceil((1.96 / max(abs(diff), 1e-9)) ** 2 * (d / n))
        print("→ صفر داخلِ بازه است: این داده نمی‌تواند بگوید کدام بهتر است.")
        print(f"   برای قطعیت حدودِ {need} بیت لازم است"
              + ("  (فراتر از پیکره — یعنی اثر اگر هست بسیار کوچک است)"
                 if need > 60000 else ""))
    else:
        print(f"→ معنادار؛ «{labels[0] if diff > 0 else labels[1]}» بهتر است.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--ckpt", default="/tmp/ab_ckpt.json")
    ap.add_argument("--fresh", action="store_true", help="نقطهٔ‌ذخیره را دور بریز")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    if a.fresh and os.path.exists(a.ckpt):
        os.remove(a.ckpt)
    done = []
    if os.path.exists(a.ckpt):
        with open(a.ckpt, encoding="utf-8") as f:
            done = json.load(f)
        print(f">> ادامه از نقطهٔ‌ذخیره: {len(done)} بیت از قبل انجام شده")

    sample = list(p3r.rows(a.csv, split="test", limit=a.n))
    todo = sample[len(done):]
    print(f">> {len(sample)} بیتِ کنارگذاشته، {len(todo)} مانده، "
          f"{a.workers} پردازه", flush=True)

    if todo:
        with mp.Pool(a.workers, initializer=_init) as pool:
            for i in range(0, len(todo), a.block):
                blk = todo[i:i + a.block]
                res = pool.map(_one, blk, chunksize=4)
                done.extend([r for r in res if r is not None])
                tmp = a.ckpt + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(done, f)
                os.replace(tmp, a.ckpt)     # اتمیک: نقطهٔ‌ذخیره نیم‌نوشته نماند
                p = [(x, y) for x, y in done]
                ao = sum(1 for x, _ in p if x); bo = sum(1 for _, y in p if y)
                print(f"   {len(done)}/{len(sample)}  "
                      f"با={100*ao/len(p):.1f}%  بدون={100*bo/len(p):.1f}%",
                      flush=True)

    stats([(x, y) for x, y in done], ("با واژه‌نامه", "بدونِ آن"))


if __name__ == "__main__":
    main()
