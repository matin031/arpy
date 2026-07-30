#!/bin/bash
# استخراجِ ترتیبی — هر تکه با همهٔ هسته‌ها، پس زود تمام و *نوشته* می‌شود.
#
# چرا ترتیبی و نه موازی: کلِ کار ثابت است (~۴.۵ بیت بر ثانیه، مستقل از تقسیم)،
# ولی extract فقط در پایانِ تکه می‌نویسد. این محیط کانتینر را گاهی راه‌اندازیِ
# دوباره می‌کند و هر تکهٔ نیمه‌کاره صفر خروجی می‌دهد. چهار تکهٔ موازیِ ۳۶ دقیقه‌ای
# یعنی ۳۶ دقیقه ریسک برای هر چهار؛ چهار تکهٔ ترتیبیِ ۱۱ دقیقه‌ای یعنی حداکثر
# ۱۱ دقیقه کارِ ازدست‌رفته و بقیه روی دیسک.
set -u
cd "$(dirname "$0")"
CSV=/tmp/p3r/P3R.csv
for i in 0 1 2 3; do
  out=/tmp/h_$i.npz
  [ -s "$out" ] && { echo "تکه $i از قبل هست"; continue; }
  echo "── تکه $i (پرش=$((i*3000)))"
  python3 train_ranker.py extract "$CSV" --n 3000 --skip $((i*3000)) \
          --stride 106 --offset 53 --workers 3 --out "$out" 2>&1 | tail -2
done
echo "── ادغامِ ۲۴۰۰۰ بیت"
python3 train_ranker.py merge /tmp/g_a.npz /tmp/g_b.npz \
        /tmp/h_0.npz /tmp/h_1.npz /tmp/h_2.npz /tmp/h_3.npz \
        --out /tmp/g_24k.npz 2>&1 | tail -2
