# -*- coding: utf-8 -*-
"""
ab_test.py — مقایسهٔ *جفتی* دو پیکربندی روی همان بیت‌ها، با بازهٔ اطمینان.

چرا این فایل لازم شد
────────────────────
یک‌بار بر پایهٔ «۳۳۵ در برابر ۳۳۶ از ۴۰۰» نتیجه گرفتم واژه‌نامه بی‌اثر است و
حذفش کردم. آن استنتاج غلط بود: خطای نمونه‌گیریِ ۴۰۰ بیت حدودِ ±۱.۸ واحد است،
پس آن اندازه‌گیری اصلاً نمی‌توانست «صفر اثر» را از «۲ واحد اثر» تشخیص بدهد.

دو چیز این‌جا فرق می‌کند:
  ۱) مقایسهٔ جفتی: هر دو پیکربندی *همان* بیت‌ها را می‌بینند، پس سختیِ نمونه
     حذف می‌شود و فقط اختلافِ واقعی می‌ماند. آزمونِ McNemar روی بیت‌هایی که
     نتیجه‌شان فرق کرده، نه روی دو درصدِ مستقل.
  ۲) نمونهٔ بزرگ‌تر، و گزارشِ بازهٔ اطمینان — تا اگر اختلاف در حدِ نویز بود،
     «نمی‌دانیم» گفته شود نه «بی‌اثر است».

    python ab_test.py /tmp/p3r/P3R.csv --n 2500
"""
import sys, os, json, argparse, math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r

_CFG = None


def _init(cfg):
    global _CFG
    import arooz
    _CFG = cfg
    if cfg == "no-lex":
        arooz.LEXICON = {}


def _one(item):
    v1, v2, gold = item
    import arooz
    try:
        rows, *_ = arooz.detect(v1, v2)
    except Exception:
        return None
    return rows[0]["ark"] == gold


def run(csv, cfg, sample, workers):
    with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        return pool.map(_one, sample, chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    sample = list(p3r.rows(a.csv, split="test", limit=a.n))
    print(f">> {len(sample)} بیتِ کنارگذاشته، مقایسهٔ جفتی\n", flush=True)

    A = run(a.csv, "lex", sample, a.workers)      # با واژه‌نامه
    print("   با واژه‌نامه: انجام شد", flush=True)
    B = run(a.csv, "no-lex", sample, a.workers)   # بدونِ واژه‌نامه
    print("   بدونِ واژه‌نامه: انجام شد\n", flush=True)

    pairs = [(x, y) for x, y in zip(A, B) if x is not None and y is not None]
    n = len(pairs)
    a_ok = sum(1 for x, _ in pairs if x)
    b_ok = sum(1 for _, y in pairs if y)
    # فقط بیت‌هایی که نتیجه‌شان فرق کرده اطلاعات دارند (McNemar)
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)

    print(f"با واژه‌نامه   : {a_ok}/{n} = {100*a_ok/n:.1f}%")
    print(f"بدونِ واژه‌نامه : {b_ok}/{n} = {100*b_ok/n:.1f}%")
    print(f"اختلاف        : {100*(a_ok-b_ok)/n:+.2f} واحد\n")
    print(f"فقط با واژه‌نامه درست : {a_only} بیت")
    print(f"فقط بدونِ آن درست     : {b_only} بیت")

    d = a_only + b_only
    if d == 0:
        print("\nهیچ بیتی نتیجه‌اش فرق نکرد — واژه‌نامه عملاً بی‌اثر است.")
        return
    # بازهٔ اطمینانِ ۹۵٪ برای اختلاف (خطای معیارِ جفتی)
    se = math.sqrt(d) / n
    diff = (a_only - b_only) / n
    lo, hi = 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)
    z = (abs(a_only - b_only) - 1) / math.sqrt(d) if d else 0.0
    print(f"\nاختلاف با بازهٔ اطمینانِ ۹۵٪: {100*diff:+.2f} واحد  [{lo:+.2f} , {hi:+.2f}]")
    print(f"McNemar z = {z:.2f}")
    if lo <= 0 <= hi:
        print("→ صفر داخلِ بازه است: این داده نمی‌تواند اثبات کند واژه‌نامه اثر دارد یا ندارد.")
        need = math.ceil((1.96 / max(abs(diff), 1e-9)) ** 2 * (d / n))
        print(f"   برای قطعیت به حدودِ {need} بیت نیاز است.")
    else:
        better = "با واژه‌نامه" if diff > 0 else "بدونِ واژه‌نامه"
        print(f"→ اختلاف معنادار است؛ {better} بهتر است.")


if __name__ == "__main__":
    main()
