# -*- coding: utf-8 -*-
"""
ablate.py — «آیا این ویژگی‌ها کمک می‌کنند؟» روی *یک* دادهٔ ثابت

چرا لازم شد
───────────
دو رتبه‌بند را با هم مقایسه کردم که هر کدام روی یک فایلِ ویژگیِ جدا برازش
شده بودند، و نتیجه گرفتم ویژگی‌های تازه بد هستند. ولی *مبنای دستی* هم بین
دو اجرا تکان خورده بود (۶۹.۴٪ → ۶۷.۲٪) — و مبنا که ثابت است، پس آن‌که عوض
شده بود نمونهٔ اعتبارسنجی بود، نه ویژگی‌ها. مقایسه از پایه بی‌معنا بود.

این اسکریپت آن اشتباه را ممکن نمی‌کند: هر دو مدل روی *همین یک* فایل و
*همین یک* تفکیک برازش می‌شوند، و تنها چیزی که فرق می‌کند ستون‌های ویژگی است.

    python ablate.py /tmp/feats_v3.npz --drop hpos1 hpos2 early1 early2
    python ablate.py /tmp/feats_v3.npz --groups
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EPOCHS, LR, LAM = 400, 0.5, 1e-4


def _fit(Z, y, tr, va):
    """softmax روی نامزدها → (بهترین دقتِ اعتبارسنجی، وزن‌ها)"""
    k = Z.shape[2]
    w = np.zeros(k)
    best = (-1.0, w)
    for ep in range(EPOCHS):
        S = Z[tr] @ w
        S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(len(tr)), y[tr]] -= 1.0
        w -= LR * (np.einsum('bn,bnk->k', G, Z[tr]) / len(tr) + LAM * w)
        if (ep + 1) % 20 == 0:
            acc = float(((Z[va] @ w).argmax(1) == y[va]).mean())
            if acc > best[0]:
                best = (acc, w.copy())
    return best


def run(X, y, feats, keep, tr, va):
    ix = [feats.index(f) for f in keep]
    Xs = X[:, :, ix]
    k = len(ix)
    mu = Xs[tr].reshape(-1, k).mean(0)
    sd = Xs[tr].reshape(-1, k).std(0) + 1e-6
    return _fit((Xs - mu) / sd, y, tr, va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--drop", nargs="*", default=None,
                    help="ویژگی‌هایی که حذف شوند (یک اجرای دو‌مدلی)")
    ap.add_argument("--groups", action="store_true",
                    help="هر دستهٔ ویژگیِ تازه را جدا بسنج")
    a = ap.parse_args()

    d = np.load(a.file, allow_pickle=True)
    X, y, feats = d["X"], d["y"], [str(s) for s in d["feats"]]
    n = X.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, va = idx[:cut], idx[cut:]
    print(f">> {n} بیت × {X.shape[1]} نامزد × {len(feats)} ویژگی  "
          f"({len(tr)} آموزش / {len(va)} اعتبارسنجی، تفکیکِ یکسان برای همه)\n")

    full, _ = run(X, y, feats, feats, tr, va)
    print(f"همهٔ {len(feats)} ویژگی            : {100*full:.1f}%")

    if a.groups:
        V2 = ['c1', 'c2', 'cmin', 'cmax', 'cdiff', 'lfreq', 'rare', 'generic',
              'nfeet', 'plen', 'fam', 'hard1', 'hard2', 'soft1', 'soft2',
              'spen1', 'spen2', 'vpen', 'samevar', 'nofit', 'vlen']
        base = [f for f in V2 if f in feats]
        b, _ = run(X, y, feats, base, tr, va)
        print(f"فقط {len(base)} ویژگیِ نسخهٔ ۲ (مبنا) : {100*b:.1f}%   "
              f"({100*(full-b):+.1f})")
        GROUPS = {
            "موقعیتِ خطا (hpos, early)": ['hpos1', 'hpos2', 'early1', 'early2'],
            "نسبی‌سازی (dcost, dspen, dhard)": ['dcost', 'dspen', 'dhard'],
            "رتبه (isbest, nties)": ['isbest', 'nties'],
        }
        print()
        for label, g in GROUPS.items():
            g = [f for f in g if f in feats]
            if not g:
                continue
            keep = base + g
            acc, _ = run(X, y, feats, keep, tr, va)
            print(f"  مبنا + {label:<34} {100*acc:.1f}%   ({100*(acc-b):+.1f})")

    if a.drop:
        keep = [f for f in feats if f not in a.drop]
        acc, _ = run(X, y, feats, keep, tr, va)
        print(f"\nبدونِ {a.drop}: {100*acc:.1f}%   ({100*(acc-full):+.1f})")


if __name__ == "__main__":
    main()
