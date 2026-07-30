#!/bin/bash
# زنجیرهٔ پایانی: تکهٔ دوم → ادغام → برازش → سنجشِ کنارگذاشته
# (تکهٔ اول جداگانه اجرا شده؛ این ادامهٔ همان است)
set -u
cd "$(dirname "$0")"
CSV=/tmp/p3r/P3R.csv
L=/tmp/spread.log
: > "$L"

echo "── انتظار برای تکهٔ اول" | tee -a "$L"
until grep -q "✅" /tmp/exg_a.log 2>/dev/null; do sleep 20; done
tail -1 /tmp/exg_a.log | tee -a "$L"

echo "── تکهٔ دوم (۶۰۰۰ بیت، همان گام)" | tee -a "$L"
python3 train_ranker.py extract "$CSV" --n 6000 --skip 6000 --spread 12000 \
        --workers 3 --out /tmp/g_b.npz 2>&1 | tail -2 | tee -a "$L"

echo "── ادغام" | tee -a "$L"
python3 train_ranker.py merge /tmp/g_a.npz /tmp/g_b.npz \
        --out /tmp/g_12k.npz 2>&1 | tail -2 | tee -a "$L"

echo "── برازش" | tee -a "$L"
python3 train_mlp.py /tmp/g_12k.npz --hidden 32 \
        --out /tmp/ranker_spread.json 2>&1 | tail -6 | tee -a "$L"

echo "── سنجشِ کنارگذاشته (رتبه‌بندِ تازه)" | tee -a "$L"
cp ranker.json /tmp/ranker_keep.json
cp /tmp/ranker_spread.json ranker.json
python3 eval_p3r.py "$CSV" --n 400 --workers 4 2>&1 | tail -13 | tee -a "$L"
# رتبه‌بندِ سنجیده‌شده را برمی‌گردانیم؛ جایگزینی فقط اگر برده باشد، دستی.
cp /tmp/ranker_keep.json ranker.json
echo "── پایان (ranker.json به نسخهٔ سنجیدهٔ ۸۰.۰٪ برگشت)" | tee -a "$L"
