# -*- coding: utf-8 -*-
"""
train_mlp.py — رتبه‌بندِ غیرخطی با هدفِ *درستِ* رتبه‌بندی

چرا این فایل هست
────────────────
شکافِ اصلیِ موتور در رتبه‌بندی است، نه در تولیدِ نامزد: پاسخِ درست در ۸۹.۵٪
مواقع میانِ ۳ نامزدِ اول و در ۹۷.۲٪ میانِ ۱۰ نامزدِ اول هست، ولی رتبهٔ ۱ فقط
۷۵.۵٪ است. رتبه‌بندِ خطی آن ۱۴ واحد را نمی‌بندد چون تعاملِ ویژگی‌ها را نمی‌بیند
(مثلاً «هزینهٔ کم» تنها وقتی معنا دارد که وزن عمومی نباشد).

یک‌بار این را با درخت‌های تقویت‌شده امتحان کردم و نتیجه بی‌اعتبار بود، چون
مسئله را «آیا این نامزد درست است؟» (یک مثبت در برابرِ ۱۷۵ منفی) مدل کردم نه
«کدامِ این ۱۷۶ نامزد درست است؟». این‌جا هدف از پایه رتبه‌بندی است: softmax روی
نامزدهای *همان بیت*، دقیقاً همان چیزی که موتور در زمانِ اجرا انجام می‌دهد.

شبکه کوچک است (۲۱→۳۲→۱، حدودِ ۷۰۰ پارامتر) به دو دلیل: با ۴۸۰۰ بیت بیش‌برازش
نکند، و پورتش به TypeScript یک ضربِ ماتریسیِ ساده بماند.

    python train_mlp.py /tmp/feats_v3.npz --hidden 32 --out ranker_mlp.json
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ویژگی‌های نسخهٔ ۲ — همان‌هایی که رتبه‌بندِ خطیِ فعلی (۷۵.۵٪) استفاده می‌کند.
V2_FEATS = ['c1', 'c2', 'cmin', 'cmax', 'cdiff', 'lfreq', 'rare', 'generic',
            'nfeet', 'plen', 'fam', 'hard1', 'hard2', 'soft1', 'soft2',
            'spen1', 'spen2', 'vpen', 'samevar', 'nofit', 'vlen']


def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def fit_linear(Z, y, tr, va, epochs=400, lr=0.5, lam=1e-4):
    """مبنا: همان رتبه‌بندِ خطیِ فعلی، روی همین تفکیک — تا مقایسه معنا داشته باشد."""
    w = np.zeros(Z.shape[2])
    best = (-1.0, w)
    for ep in range(epochs):
        P = softmax_rows(Z[tr] @ w)
        G = P.copy(); G[np.arange(len(tr)), y[tr]] -= 1.0
        w -= lr * (np.einsum('bn,bnk->k', G, Z[tr]) / len(tr) + lam * w)
        if (ep + 1) % 20 == 0:
            acc = float(((Z[va] @ w).argmax(1) == y[va]).mean())
            if acc > best[0]:
                best = (acc, w.copy())
    return best


def fit_mlp(Z, y, tr, va, hidden=32, epochs=300, lr=3e-3, lam=1e-4,
            batch=256, seed=0, patience=40, quiet=False):
    """
    ۲۱→hidden(tanh)→۱، آموزش با Adam و آنتروپیِ متقابلِ softmax روی نامزدها.
    tanh (نه ReLU): امتیاز باید هموار باشد تا رتبه‌بندی روی نمونه‌های نزدیک
    ناپایدار نشود، و tanh با ویژگی‌های استانداردشده مرکزِ درستی دارد.
    """
    rng = np.random.default_rng(seed)
    k = Z.shape[2]
    # مقیاس‌دهیِ Xavier — با tanh واریانسِ سیگنال را در عبور حفظ می‌کند
    W1 = rng.normal(0, np.sqrt(1.0 / k), (k, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(1.0 / hidden), hidden)
    b2 = 0.0
    params = [W1, b1, w2, b2]
    ms = [np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0 for p in params]
    vs = [np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0 for p in params]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    step = 0

    def forward(Zb, W1, b1, w2, b2):
        H = np.tanh(Zb @ W1 + b1)            # (بیت، نامزد، نهفته)
        return H, H @ w2 + b2                # (بیت، نامزد)

    def acc_on(sub, W1, b1, w2, b2):
        _, S = forward(Z[sub], W1, b1, w2, b2)
        return float((S.argmax(1) == y[sub]).mean())

    best = (-1.0, None)
    since = 0
    for ep in range(epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(order), batch):
            bi = tr[order[s:s + batch]]
            Zb, yb = Z[bi], y[bi]
            H, S = forward(Zb, W1, b1, w2, b2)
            P = softmax_rows(S)
            G = P.copy(); G[np.arange(len(bi)), yb] -= 1.0
            G /= len(bi)                      # (بیت، نامزد)
            gw2 = np.einsum('bn,bnh->h', G, H) + lam * w2
            gb2 = G.sum()
            dH = np.einsum('bn,h->bnh', G, w2) * (1.0 - H * H)
            gW1 = np.einsum('bnk,bnh->kh', Zb, dH) + lam * W1
            gb1 = dH.sum(axis=(0, 1))
            step += 1
            for i, g in enumerate((gW1, gb1, gw2, gb2)):
                ms[i] = b1_ * ms[i] + (1 - b1_) * g
                vs[i] = b2_ * vs[i] + (1 - b2_) * (g * g)
                mh = ms[i] / (1 - b1_ ** step)
                vh = vs[i] / (1 - b2_ ** step)
                upd = lr * mh / (np.sqrt(vh) + eps)
                if i == 0:   W1 = W1 - upd
                elif i == 1: b1 = b1 - upd
                elif i == 2: w2 = w2 - upd
                else:        b2 = b2 - upd
        if (ep + 1) % 5 == 0:
            av = acc_on(va, W1, b1, w2, b2)
            if av > best[0]:
                best = (av, (W1.copy(), b1.copy(), w2.copy(), float(b2)))
                since = 0
            else:
                since += 5
            if not quiet:
                at = acc_on(tr[:1000], W1, b1, w2, b2)
                print(f"   دورهٔ {ep+1:>3}  آموزش={100*at:.1f}%  "
                      f"اعتبارسنجی={100*av:.1f}%", flush=True)
            if since >= patience:
                if not quiet:
                    print(f"   توقفِ زودهنگام (بی‌بهبود در {patience} دوره)")
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-feats", action="store_true",
                    help="همهٔ ویژگی‌های فایل (پیش‌فرض: فقط ۲۱ ویژگیِ نسخهٔ ۲)")
    ap.add_argument("--out", default="/tmp/ranker_mlp.json")
    a = ap.parse_args()

    d = np.load(a.file, allow_pickle=True)
    X, y, allf = d["X"], d["y"], [str(s) for s in d["feats"]]
    feats = allf if a.all_feats else [f for f in V2_FEATS if f in allf]
    X = X[:, :, [allf.index(f) for f in feats]].astype(np.float64)
    n, m, k = X.shape

    rng = np.random.default_rng(0)          # همان تفکیکِ train_ranker/ablate
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, va = idx[:cut], idx[cut:]
    mu = X[tr].reshape(-1, k).mean(0)
    sd = X[tr].reshape(-1, k).std(0) + 1e-6
    Z = (X - mu) / sd
    print(f">> {n} بیت × {m} نامزد × {k} ویژگی  "
          f"({len(tr)} آموزش / {len(va)} اعتبارسنجی)\n")

    acc_lin, _ = fit_linear(Z, y, tr, va)
    print(f"مبنا — رتبه‌بندِ خطی روی همین تفکیک: {100*acc_lin:.1f}%\n")

    print(f">> شبکه {k}→{a.hidden}→1")
    acc, P = fit_mlp(Z, y, tr, va, hidden=a.hidden, epochs=a.epochs,
                     lr=a.lr, lam=a.lam, seed=a.seed)
    W1, b1, w2, b2 = P
    print(f"\nدقتِ رتبهٔ ۱ روی اعتبارسنجی:")
    print(f"   خطی      : {100*acc_lin:.1f}%")
    print(f"   غیرخطی   : {100*acc:.1f}%   ({100*(acc-acc_lin):+.1f})")

    # موتور کمینهٔ امتیاز را برنده می‌داند، پس علامت را وارونه ذخیره می‌کنیم.
    out = {"kind": "mlp", "feats": feats,
           "mu": mu.tolist(), "sd": sd.tolist(),
           "W1": W1.tolist(), "b1": b1.tolist(),
           "w2": (-w2).tolist(), "b2": float(-b2),
           "hidden": a.hidden, "val_acc": float(acc),
           "linear_acc": float(acc_lin), "n_train": int(len(tr))}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n✅ {a.out}  ({os.path.getsize(a.out)/1024:.0f} کیلوبایت)")


if __name__ == "__main__":
    main()
