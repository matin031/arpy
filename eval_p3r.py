# -*- coding: utf-8 -*-
"""
eval_p3r.py — سنجشِ دقتِ واقعیِ موتور روی مجموعهٔ آزمونِ P3R

    python eval_p3r.py /tmp/p3r/P3R.csv --n 500
    python eval_p3r.py /tmp/p3r/P3R.csv --n 500 --no-lex     # بدونِ واژه‌نامه
    python eval_p3r.py /tmp/p3r/P3R.csv --n 500 --workers 4

برخلافِ bench.py (۳۹ بیتِ دستچین‌شده و سوگیرانه)، این روی نمونهٔ تصادفیِ
بیت‌هایی اجرا می‌شود که واژه‌نامه هرگز آن‌ها را ندیده است.
"""
import sys, os, time, random, argparse, collections
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r

_NO_LEX = False


def _init(no_lex, lex_path=None, beam=None):
    import arooz
    if beam:
        arooz.BEAM = beam
    if no_lex:
        arooz.LEXICON = {}
    elif lex_path:
        arooz.load_lexicon(lex_path)


def _one(item):
    v1, v2, gold = item
    import arooz
    try:
        rows, conf, *_ = arooz.detect(v1, v2)
    except Exception as e:
        return (gold, None, "خطا", None)
    got = rows[0]["ark"]
    rank = next((k + 1 for k, x in enumerate(rows) if x["ark"] == gold), None)
    return (gold, got, conf, rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-lex", action="store_true")
    ap.add_argument("--lex", default=None, help="مسیرِ واژه‌نامهٔ جایگزین")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beam", type=int, default=None)
    args = ap.parse_args()

    print(f">> خواندنِ مجموعهٔ آزمون…", flush=True)
    pool_rows = []
    for i, r in enumerate(p3r.rows(args.csv, split="test")):
        pool_rows.append(r)
        if i >= 60000:
            break
    random.Random(args.seed).shuffle(pool_rows)
    sample = pool_rows[: args.n]
    _tag = 'خاموش' if args.no_lex else (args.lex or 'پیش‌فرض')
    print(f">> {len(sample)} بیت از مجموعهٔ آزمون  |  واژه‌نامه: {_tag}"
          f"  |  {args.workers} پردازه", flush=True)

    t0 = time.time()
    ok = 0
    by_conf = collections.Counter()
    ok_by_conf = collections.Counter()
    ranks = []
    done = 0

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.no_lex, args.lex, args.beam)) as pool:
        for gold, got, conf, rank in pool.imap_unordered(_one, sample, chunksize=1):
            done += 1
            hit = (got == gold)
            ok += hit
            by_conf[conf] += 1
            ok_by_conf[conf] += hit
            if rank:
                ranks.append(rank)
            if done % 25 == 0:
                el = time.time() - t0
                print(f"   {done}/{len(sample)}  دقت={100*ok/done:.1f}%  "
                      f"({el/done:.1f} ثانیه/بیت)", flush=True)

    el = time.time() - t0
    n = len(sample)
    print("=" * 62)
    print(f"دقتِ کل: {ok}/{n} = {100*ok/n:.1f}%     [{el/60:.1f} دقیقه]")
    print()
    print("به تفکیکِ اطمینانِ اعلامیِ موتور:")
    for c in ("بسیار بالا", "بالا", "متوسط", "پایین", "نامطمئن", "خطا"):
        if by_conf[c]:
            print(f"   {c:<12} {ok_by_conf[c]:>4}/{by_conf[c]:<4} = "
                  f"{100*ok_by_conf[c]/by_conf[c]:5.1f}%")
    if ranks:
        ranks.sort()
        top3 = sum(1 for r in ranks if r <= 3)
        top10 = sum(1 for r in ranks if r <= 10)
        print()
        print(f"پاسخِ درست در ۳ نامزدِ اول: {100*top3/n:.1f}%    "
              f"در ۱۰ نامزدِ اول: {100*top10/n:.1f}%")
        print(f"(یعنی سقفِ ممکن با بهبودِ رتبه‌بندی، بدونِ تغییرِ تولیدِ نامزدها)")


if __name__ == "__main__":
    main()
