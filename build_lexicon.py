# -*- coding: utf-8 -*-
"""
build_lexicon.py — ساختِ «واژه‌نامهٔ عروضیِ کلاسیک» از پیکرهٔ برچسب‌خوردهٔ گنجور

ایده
────
موتورِ قاعده‌محور نمی‌داند واژه در شعر چه وزنی می‌گیرد، پس برای هر مصراع
هزاران خوانش می‌سازد و وزنِ غلط هم تصادفاً جور درمی‌آید.
ولی اگر بیتی را داشته باشیم که وزنش *معلوم* است، می‌توان با هم‌ترازیِ
وزن‌مقید فهمید هر واژه **در عمل** چه رشتهٔ وزنی گرفته است.
روی ۱.۳ میلیون بیت، این می‌شود یک واژه‌نامهٔ آماریِ تلفظِ عروضی.

خروجی:  lexicon.json  =  { "واژه": {"-U": 1523, "U-": 44}, ... }
arooz.py خودش آن را می‌خوانَد (اگر کنارش باشد).

اجرا
────
    python build_lexicon.py  P3R.csv  [--limit N]

دانلودِ دیتاست (۲۰۸ مگابایت، یک‌بار):
    https://media.githubusercontent.com/media/m-shahrestani/Prosody-Recognition-in-Persian-Poetry/master/dataset/P3R.csv
"""
import sys, json, csv, re
from collections import defaultdict, Counter
import arooz


# ═══════════════════ هم‌ترازیِ وزن‌مقید (DP) ═══════════════════
# ═══════════════════ ساختِ واژه‌نامه ═══════════════════
align = arooz.align


def build(csv_path, limit=None, out='lexicon.json'):
    pat_of = {}
    for name, ark, pat, fr in arooz.METERS:
        pat_of[ark] = pat

    lex = defaultdict(Counter)
    n_ok = n_bad = n_nometer = 0

    with open(csv_path, encoding='utf-8') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        cv = next((c for c in cols if 'verse' in c.lower() or 'text' in c.lower()
                   or 'mesra' in c.lower() or 'poem' in c.lower()), cols[0])
        cm = next((c for c in cols if 'prosody' in c.lower() and 'id' not in c.lower()), None)
        if cm is None:
            cm = next(c for c in cols if 'prosody' in c.lower())
        print(f"ستون‌ها: متن={cv}  وزن={cm}")

        for i, row in enumerate(rd):
            if limit and i >= limit:
                break
            if i and i % 20000 == 0:
                print(f"  {i:>8} بیت | موفق={n_ok} ناموفق={n_bad} بی‌وزن={n_nometer} | واژه={len(lex)}")
            ark = (row.get(cm) or '').strip()
            pat = pat_of.get(ark)
            if not pat:
                n_nometer += 1
                continue
            text = (row.get(cv) or '').strip()
            for mesra in re.split(r'[|\t]|\s{3,}', text):
                mesra = mesra.strip()
                if not mesra:
                    continue
                try:
                    a = align(mesra, pat)
                except Exception:
                    a = None
                if not a:
                    n_bad += 1
                    continue
                n_ok += 1
                ws = arooz.normalize(mesra).split()
                for wi, wt in a.items():
                    if wi < len(ws):
                        lex[ws[wi]][wt] += 1

    # فقط واژه‌هایی که دستِ‌کم ۲ بار دیده شده‌اند (حذفِ نویز)
    final = {w: dict(c) for w, c in lex.items() if sum(c.values()) >= 2}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False)
    print(f"\n✅ {out}: {len(final)} واژه  |  مصراعِ هم‌ترازشده: {n_ok}  ناموفق: {n_bad}")
    return final


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    lim = None
    if '--limit' in sys.argv:
        lim = int(sys.argv[sys.argv.index('--limit') + 1])
    build(sys.argv[1], lim)
