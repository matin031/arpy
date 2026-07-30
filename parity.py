# -*- coding: utf-8 -*-
"""
parity.py — «آیا موتورِ TypeScript همان چیزی را می‌گوید که arooz.py؟»

وزن‌های رتبه‌بند این‌جا (پایتون) آموخته می‌شوند و آن‌جا (مرورگر) اجرا. اگر
ویژگی‌ها در دو پیاده‌سازی یکی نباشند، وزن‌های آموخته بی‌معنا می‌شوند و — بدترین
بخش — هیچ خطایی رخ نمی‌دهد: صفحه جواب می‌دهد، فقط جوابِ بدتری. پس این بررسی
باید بخشی از هر تغییرِ موتور باشد، نه کارِ یک‌بار.

یک‌بار همین‌جا واگرایی پیدا شد: BEAM در TypeScript ۴۰۰۰۰ مانده بود و در پایتون
۲۰۰۰ شده بود، پس ویژگیِ spen فرق می‌کرد.

    python parity.py                    # ۲۵ بیتِ نمونه از پیکره
    python parity.py --n 60
"""
import sys, os, json, subprocess, argparse, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SITE = os.environ.get("AROOZ_SITE", "/workspace/arooz")
P3R = os.environ.get("P3R_CSV", "/tmp/p3r/P3R.csv")


def sample_bayts(n):
    """از مجموعهٔ *آزمونِ* پیکره — تا مقایسه روی بیت‌هایی باشد که هیچ‌کدام ندیده‌اند."""
    import p3r
    return [[v1, v2] for v1, v2, _ in p3r.rows(P3R, split="test", limit=n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()

    import arooz
    pairs = sample_bayts(a.n)
    if not pairs:
        sys.exit(f"پیکره پیدا نشد: {P3R}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
        tmp = f.name
    try:
        raw = subprocess.run(
            ["node", "scripts/aruz-parity.mjs", tmp],
            cwd=SITE, capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(tmp)
    if raw.returncode != 0:
        sys.exit("اجرای موتورِ TypeScript شکست خورد:\n" + raw.stderr[-3000:])
    ts = json.loads(raw.stdout)

    same_ark = same_score = same_conf = same_summ = 0
    for (m1, m2), t in zip(pairs, ts):
        rows, conf, _, _ = arooz.detect(m1, m2)
        b = rows[0]
        ok_a = b["ark"] == t["ark"]
        ok_s = abs(b["score"] - t["score"]) < 1e-6
        ok_m = abs(b["summ"] - t["summ"]) < 1e-6
        same_ark += ok_a; same_score += ok_s
        same_conf += (conf == t["conf"]); same_summ += ok_m
        if not (ok_a and ok_s):
            pl = b.get("lex"); tl = t.get("lex")
            print(f"{'✗ وزن' if not ok_a else '≈ امتیاز'} — {m1[:38]}")
            print(f"    پایتون: {b['ark']:<30} امتیاز={b['score']:+.6f} "
                  f"هزینه={b['summ']:.6f} واژه‌نامه={pl}")
            print(f"    TS    : {t['ark']:<30} امتیاز={t['score']:+.6f} "
                  f"هزینه={t['summ']:.6f} واژه‌نامه={tl}")

    n = len(pairs)
    print(f"\nوزنِ یکسان          : {same_ark}/{n}")
    print(f"هزینهٔ خامِ یکسان    : {same_summ}/{n}   (تقطیع + جدولِ اوزان)")
    print(f"امتیازِ یکسان (۱e-۶): {same_score}/{n}   (+ رتبه‌بند و واژه‌نامه)")
    print(f"اطمینانِ یکسان      : {same_conf}/{n}")
    sys.exit(0 if same_ark == n and same_score == n else 1)


if __name__ == "__main__":
    main()
