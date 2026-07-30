# -*- coding: utf-8 -*-
"""
eval_class.py — دقت روی بیت‌هایی که برچسبشان یک وزنِ مشخص است.

چرا: وقتی یک تغییر دقتِ کل را بالا می‌برد ولی یک بیتِ آشنا را خراب می‌کند،
وسوسه‌ای هست که بر پایهٔ همان یک بیت قضاوت شود — همان کاری که ضرایبِ دستیِ
اولیه را ساخت. این اسکریپت جای آن یک بیت، *دستهٔ* او را می‌سنجد.

    python eval_class.py /tmp/p3r/P3R.csv "مفعول مفاعیل مفاعیل فعل" \
           --rankers ranker.json /tmp/ranker_mlp_old.json
"""
import sys, os, json, argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3r

_RANKER = None


def _init(path):
    global _RANKER
    import arooz
    with open(path, encoding="utf-8") as f:
        arooz.RANKER = json.load(f)
    arooz._MLP = None
    _RANKER = path


def _one(item):
    v1, v2, gold = item
    import arooz
    try:
        rows, conf, _, _ = arooz.detect(v1, v2)
    except Exception:
        return None
    return rows[0]["ark"] == gold


def run(csv, gold, path, limit, workers):
    sample = [r for r in p3r.rows(csv, split="test") if r[2] == gold][:limit]
    if not sample:
        return None, 0
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool:
        res = [x for x in pool.map(_one, sample, chunksize=4) if x is not None]
    return (sum(res) / len(res) if res else 0.0), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("gold")
    ap.add_argument("--rankers", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    print(f">> برچسب: {a.gold}")
    for p in a.rankers:
        acc, n = run(a.csv, a.gold, p, a.limit, a.workers)
        if acc is None:
            print(f"   {os.path.basename(p):<26} — بیتی با این برچسب نبود")
        else:
            print(f"   {os.path.basename(p):<26} {100*acc:5.1f}%   ({n} بیت)")


if __name__ == "__main__":
    main()
