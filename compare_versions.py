# -*- coding: utf-8 -*-
"""
compare_versions.py — «موتور نسبت به اول چقدر بهتر شده؟» با مقایسهٔ جفتی.

عددهای این جلسه در مقاطعِ مختلف و روی نمونه‌های مختلف گرفته شده‌اند (۴۰۰ بیت،
بعد ۶۰۰۰ بیت)، پس کنارِ هم گذاشتنشان دقیق نیست. این اسکریپت نسخهٔ اولِ موتور
(از کامیتِ نخست) و نسخهٔ فعلی را روی *همان* بیت‌ها اجرا می‌کند و اختلاف را با
بازهٔ اطمینان می‌دهد.

نسخهٔ اول در /tmp/orig گذاشته می‌شود:
    git show <کامیتِ‌اول>:arooz.py    > /tmp/orig/arooz.py
    git show <کامیتِ‌اول>:lexicon.json > /tmp/orig/lexicon.json

    python compare_versions.py /tmp/p3r/P3R.csv --n 3000
"""
import sys, os, json, argparse, math
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3r

ORIG = "/tmp/orig"


def _one(item):
    """یک بیت زیرِ هر دو نسخه. هر پردازه هر دو ماژول را جدا وارد می‌کند."""
    v1, v2, gold = item
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    g = globals()
    if "_OLD" not in g:
        cwd = os.getcwd()
        os.chdir(ORIG)                      # تا واژه‌نامهٔ کنارِ خودش را بردارد
        g["_OLD"] = load(os.path.join(ORIG, "arooz.py"), "arooz_old")
        os.chdir(cwd)
        g["_NEW"] = load(os.path.join(HERE, "arooz.py"), "arooz_new")

    def ark(mod):
        try:
            r = mod.detect(v1, v2)
            return (r[0] if isinstance(r, tuple) else r)[0]["ark"]
        except Exception:
            return None

    return (ark(g["_OLD"]) == gold, ark(g["_NEW"]) == gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--block", type=int, default=250)
    ap.add_argument("--ckpt", default="/tmp/cmp_ckpt.json")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    if a.fresh and os.path.exists(a.ckpt):
        os.remove(a.ckpt)
    done = []
    if os.path.exists(a.ckpt):
        done = json.load(open(a.ckpt, encoding="utf-8"))
        print(f">> ادامه از نقطهٔ‌ذخیره: {len(done)} بیت")

    sample = list(p3r.rows(a.csv, split="test", limit=a.n))
    todo = sample[len(done):]
    print(f">> {len(sample)} بیتِ کنارگذاشته، {len(todo)} مانده", flush=True)

    if todo:
        with mp.Pool(a.workers) as pool:
            for i in range(0, len(todo), a.block):
                res = pool.map(_one, todo[i:i + a.block], chunksize=2)
                done.extend([r for r in res if r is not None])
                tmp = a.ckpt + ".tmp"
                json.dump(done, open(tmp, "w", encoding="utf-8"))
                os.replace(tmp, a.ckpt)
                o = sum(1 for x, _ in done if x); n_ = sum(1 for _, y in done if y)
                print(f"   {len(done)}/{len(sample)}  "
                      f"اولیه={100*o/len(done):.1f}%  فعلی={100*n_/len(done):.1f}%",
                      flush=True)

    n = len(done)
    o_ok = sum(1 for x, _ in done if x)
    n_ok = sum(1 for _, y in done if y)
    o_only = sum(1 for x, y in done if x and not y)
    n_only = sum(1 for x, y in done if y and not x)

    print(f"\n{'موتورِ اولیه':<14}: {o_ok}/{n} = {100*o_ok/n:.1f}%")
    print(f"{'موتورِ فعلی':<14}: {n_ok}/{n} = {100*n_ok/n:.1f}%")
    print(f"{'بهبود':<14}: {100*(n_ok-o_ok)/n:+.1f} واحد")
    print(f"\nبیت‌هایی که فعلی درست کرد و اولیه غلط : {n_only}")
    print(f"بیت‌هایی که اولیه درست کرد و فعلی غلط : {o_only}")

    d = o_only + n_only
    if d:
        se = math.sqrt(d) / n
        diff = (n_only - o_only) / n
        print(f"\nبازهٔ اطمینانِ ۹۵٪: [{100*(diff-1.96*se):+.1f} , {100*(diff+1.96*se):+.1f}] واحد")
        z = (abs(n_only - o_only) - 1) / math.sqrt(d)
        print(f"McNemar z = {z:.1f}  → " +
              ("معنادار" if abs(z) > 1.96 else "داخلِ نویز"))


if __name__ == "__main__":
    main()
