# -*- coding: utf-8 -*-
"""
train_ranker.py — آموختنِ وزنِ رتبه‌بندی از پیکرهٔ برچسب‌خورده

مسئله
─────
موتور نامزدِ درست را تقریباً همیشه تولید می‌کند (۸۷٪ در ۳ نامزدِ اول، ۹۵٪ در ۱۰
نامزدِ اول) ولی رتبهٔ ۱ فقط ۶۸٪ است. یعنی گلوگاه، *امتیازدهی* است نه تولید.
امتیازِ فعلی جمعِ چند ضریبِ دستی است (MU، RARE_TAIL_PENALTY، …) که هر کدام روی
چند ده نمونه کوک شده‌اند — هر بار یکی درست می‌شد و یکی خراب.

راه‌حل
──────
همان ویژگی‌ها را نگه می‌داریم ولی وزن‌هایشان را از ۱.۲۷ میلیون بیتِ برچسب‌خورده
*یاد می‌گیریم*: رگرسیونِ softmax روی نامزدها (یادگیریِ رتبه‌بندی). امتیاز خطی
می‌ماند، پس هم قابلِ‌فهم است هم پورتش به TypeScript یک ضربِ داخلی است.

    # ۱) استخراجِ ویژگی (کند — موازی و قابلِ ادامه)
    python train_ranker.py extract /tmp/p3r/P3R.csv --n 4000 --out /tmp/feats.npz

    # ۲) برازشِ وزن‌ها (ثانیه‌ای)
    python train_ranker.py fit /tmp/feats.npz --out ranker.json
"""
import sys, os, json, time, argparse
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r


# ─────────────────────────── استخراج ───────────────────────────
def _init():
    import arooz
    arooz.RANKER = None          # ویژگی‌ها نباید به رتبه‌بندِ قبلی وابسته باشند


def _one(item):
    """یک بیت → (ماتریسِ ویژگی، نمایهٔ پاسخِ درست) یا None"""
    v1, v2, gold = item
    import arooz
    try:
        rows, _, _ = arooz.candidate_rows(v1, v2)
    except Exception:
        return None
    gi = next((k for k, r in enumerate(rows) if r["ark"] == gold), None)
    if gi is None:
        return None              # وزنِ برچسب در جدولِ ما نیست
    X = np.array([[r["feat"][f] for f in arooz.FEATURES] for r in rows],
                 dtype=np.float32)
    return X, gi


def extract(args):
    import arooz
    print(f">> ویژگی‌ها: {list(arooz.FEATURES)}")
    print(f">> {args.n} بیتِ آموزشی، {args.workers} پردازه", flush=True)
    sample = list(p3r.rows(args.csv, split="train", limit=args.n))
    Xs, gis = [], []
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_one, sample, chunksize=4), 1):
            if res is not None:
                Xs.append(res[0]); gis.append(res[1])
            if i % 200 == 0:
                el = time.time() - t0
                print(f"   {i}/{len(sample)}  ({el/60:.1f} دقیقه، "
                      f"{i/el:.1f} بیت/ثانیه)", flush=True)
    X = np.stack(Xs)                       # (بیت، نامزد، ویژگی)
    y = np.array(gis, dtype=np.int32)
    np.savez_compressed(args.out, X=X, y=y, feats=np.array(arooz.FEATURES))
    print(f"\n✅ {args.out}:  {X.shape[0]} بیت × {X.shape[1]} نامزد × "
          f"{X.shape[2]} ویژگی   [{(time.time()-t0)/60:.1f} دقیقه]")


# ─────────────────────────── برازش ───────────────────────────
def fit(args):
    d = np.load(args.file, allow_pickle=True)
    X, y, feats = d["X"], d["y"], [str(s) for s in d["feats"]]
    n, m, k = X.shape
    print(f">> {n} بیت × {m} نامزد × {k} ویژگی")

    # ٪۲۰ برای اعتبارسنجی، تا بیش‌برازش را ببینیم
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, va = idx[:cut], idx[cut:]

    # استانداردسازی (شرطِ لازمِ گرادیانِ سالم)
    mu = X[tr].reshape(-1, k).mean(0)
    sd = X[tr].reshape(-1, k).std(0) + 1e-6
    Z = (X - mu) / sd

    # امتیاز = -(w·z)  ⇒  کمترین امتیاز بهترین. برای softmax از +w·z استفاده
    # می‌کنیم و در پایان علامت را برمی‌گردانیم.
    w = np.zeros(k, dtype=np.float64)
    lr, lam = 0.5, 1e-4
    best = (-1, None)
    for ep in range(400):
        S = Z[tr] @ w                              # (بیت، نامزد)
        S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        G = P.copy()
        G[np.arange(len(tr)), y[tr]] -= 1.0
        grad = np.einsum('bn,bnk->k', G, Z[tr]) / len(tr) + lam * w
        w -= lr * grad
        if (ep + 1) % 20 == 0:
            acc_tr = (( Z[tr] @ w).argmax(1) == y[tr]).mean()
            acc_va = (( Z[va] @ w).argmax(1) == y[va]).mean()
            if acc_va > best[0]:
                best = (acc_va, w.copy())
            print(f"   دورهٔ {ep+1:>3}  آموزش={100*acc_tr:.1f}%  "
                  f"اعتبارسنجی={100*acc_va:.1f}%", flush=True)
    acc_va, w = best

    # مبنا: امتیازِ دستیِ فعلی روی همان مجموعهٔ اعتبارسنجی
    fi = {f: i for i, f in enumerate(feats)}
    import arooz
    base = -(X[va][:, :, fi['c1']] + X[va][:, :, fi['c2']]
             + (-arooz.MU) * X[va][:, :, fi['lfreq']]
             + arooz.RARE_TAIL_PENALTY * X[va][:, :, fi['rare']])
    acc_base = (base.argmax(1) == y[va]).mean()

    print()
    print(f"دقتِ رتبهٔ ۱ روی اعتبارسنجی:")
    print(f"   امتیازِ دستیِ فعلی : {100*acc_base:.1f}%")
    print(f"   رتبه‌بندِ آموخته   : {100*acc_va:.1f}%")

    # w روی z آموخته شد؛ به ویژگیِ خام برگردان و علامت را وارونه کن
    # (موتور کمینه‌ی امتیاز را برنده می‌داند)
    w_raw = {f: float(-w[i] / sd[i]) for i, f in enumerate(feats)}
    bias = float(sum(w[i] * mu[i] / sd[i] for i in range(k)))
    out = {"w": w_raw, "bias": bias,
           "val_acc": float(acc_va), "baseline_acc": float(acc_base),
           "n_train": int(len(tr)), "feats": feats}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n✅ {args.out}")
    for f_, v in sorted(w_raw.items(), key=lambda z: -abs(z[1])):
        print(f"   {f_:<9} {v:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("csv")
    e.add_argument("--n", type=int, default=3000)
    e.add_argument("--out", default="/tmp/feats.npz")
    e.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    f_ = sub.add_parser("fit"); f_.add_argument("file")
    f_.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ranker.json"))
    a = ap.parse_args()
    (extract if a.cmd == "extract" else fit)(a)
