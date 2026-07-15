# -*- coding: utf-8 -*-
"""
bootstrap_lexicon.py — ساختِ واژه‌نامهٔ عروضی از پیکرهٔ بی‌برچسبِ گنجور (نسخهٔ ۳)

اجرا:
    python -u bootstrap_lexicon.py ChronologicalPersianPoetryDataset/poems_with_more_info.tsv --poems 3000

• هر ۱۰ شعرِ پذیرفته‌شده lexicon.json ذخیره می‌شود (checkpoint).
• ادامه از جایی که بودی:  --skip <شمارهٔ شعر>

اصلاحِ کلیدیِ نسخهٔ ۳:
  کشِ موتور (_EXP_CACHE) بی‌کران است و بعد از چند صد شعر تمامِ رم را می‌خورد.
  علتِ اصلیِ MemoryError همین بود — حالا بعد از هر شعر خالی می‌شود.
"""
import sys, json, time, re, io, gc
from collections import defaultdict, Counter
import arooz

SAMPLE    = 5
MIN_LINES = 6
MAX_ROW   = 200000
TOP_METERS = [(n, a, p, f) for n, a, p, f in arooz.METERS if f >= 0.03]


def clear_caches():
    """★ کشِ بی‌کرانِ موتور رم را می‌خورد — بعد از هر شعر خالی می‌شود."""
    try:
        arooz._EXP_CACHE.clear()
    except Exception:
        pass
    try:
        arooz._VAR_CACHE.clear()
    except Exception:
        pass


def safe_lines(path, max_bytes=MAX_ROW):
    """خواندنِ ایمن: سطرِ غول‌پیکرِ خراب را قبل از خواندنِ کامل دور می‌ریزد."""
    with open(path, 'rb') as f:
        buf = b''
        skipping = False
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b'\n')
                if i < 0:
                    break
                line = buf[:i]
                buf = buf[i + 1:]
                if skipping:
                    skipping = False
                    continue
                if len(line) <= max_bytes:
                    yield line.decode('utf-8', 'replace')
            if len(buf) > max_bytes:
                buf = b''
                skipping = True
        if buf and not skipping and len(buf) <= max_bytes:
            yield buf.decode('utf-8', 'replace')


def fast_scan(m):
    line = arooz.normalize(m)
    if not (3 <= len(line.split()) <= 14):
        return None
    out = {}
    for seq, pen in arooz.expand_ambiguous(arooz.tokenize_line(line), full=False):
        for sc, sp in arooz.scan_seq(seq).items():
            t = pen + sp
            if sc not in out or t < out[sc]:
                out[sc] = t
    return out or None


def poem_meter(mesras):
    idxs = []
    for m in mesras[:SAMPLE]:
        s = fast_scan(m)
        if s:
            idxs.append(arooz._by_len(s))
    if len(idxs) < 3:
        return None
    best = None
    second = 9e9
    for n, a, p, f in TOP_METERS:
        cs = [min(arooz.meter_cost(ix, p, n), 5.0) for ix in idxs]
        avg = sum(cs) / len(cs)
        worst = max(cs)
        if best is None or avg < best[0]:
            if best:
                second = best[0]
            best = (avg, worst, n, a, p)
        elif avg < second:
            second = avg
    if best is None:
        return None
    avg, worst, n, a, p = best
    if avg <= 0.45 and worst <= 1.3 and (second - avg) >= 0.35:
        return n, a, p
    return None


def save(lex, out):
    final = {w: dict(c) for w, c in lex.items() if sum(c.values()) >= 2}
    with io.open(out, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False)
    return len(final)


def build(path, max_poems=3000, skip=0, out='lexicon.json'):
    lex = defaultdict(Counter)
    n_poem = 0
    n_used = 0
    n_line = 0
    t0 = time.time()

    it = safe_lines(path)
    header = next(it).rstrip('\r').split('\t')
    ci = header.index('poem') if 'poem' in header else len(header) - 1

    for raw in it:
        cols = raw.rstrip('\r').split('\t')
        if len(cols) <= ci:
            continue
        text = cols[ci].strip()
        if not text:
            continue

        mesras = [m.strip() for m in re.split(r'\s{3,}', text) if m.strip()]
        mesras = [m for m in mesras if 3 <= len(m.split()) <= 14]
        if len(mesras) < MIN_LINES:
            continue

        n_poem += 1
        if n_poem <= skip:
            continue
        if n_poem - skip > max_poems:
            break

        clear_caches()
        try:
            r = poem_meter(mesras)
        except (MemoryError, RecursionError):
            clear_caches()
            gc.collect()
            r = None

        if not r:
            clear_caches()
            continue

        name, ark, pat = r
        n_used += 1

        for m in mesras[:40]:
            try:
                a = arooz.align(m, pat)
            except (MemoryError, RecursionError):
                clear_caches()
                a = None
            if not a:
                continue
            n_line += 1
            pairs = a.items() if isinstance(a, dict) else a
            ws = arooz.normalize(m).split()
            for k, wt in pairs:
                if isinstance(k, int):
                    w = ws[k] if k < len(ws) else None
                elif isinstance(k, str):
                    w = k
                else:
                    w = None
                if w:
                    lex[w][wt] += 1

        clear_caches()

        if n_used % 10 == 0:
            gc.collect()
            nw = save(lex, out)
            el = time.time() - t0
            rate = (n_poem - skip) / el if el else 0
            print("  شعر=%6d  پذیرفته=%5d  مصراع=%7d  واژه=%6d   (%.1f دقیقه، %.2f شعر/ثانیه)"
                  % (n_poem, n_used, n_line, nw, el / 60, rate))

    nw = save(lex, out)
    print("\nOK  %s: %d واژه  |  %d شعر پذیرفته از %d، %d مصراع  |  %.1f دقیقه"
          % (out, nw, n_used, n_poem - skip, n_line, (time.time() - t0) / 60))
    return nw


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    lim = int(sys.argv[sys.argv.index('--poems') + 1]) if '--poems' in sys.argv else 3000
    skip = int(sys.argv[sys.argv.index('--skip') + 1]) if '--skip' in sys.argv else 0
    try:
        build(sys.argv[1], lim, skip)
    except KeyboardInterrupt:
        print("\nمتوقف شد — lexicon.json تا آخرین checkpoint ذخیره شده است.")
