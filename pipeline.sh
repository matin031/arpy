#!/bin/bash
# pipeline.sh — زنجیرهٔ کاملِ «اندازه‌گیری → استخراج → برازش → اندازه‌گیری»
#
# هر تغییری در معناشناسیِ موتور (هرس، تقطیع، جدولِ اوزان) توزیعِ ویژگی‌ها را
# جابه‌جا می‌کند، پس رتبه‌بندی که روی معناشناسیِ قبلی آموخته شده دیگر دقیقاً
# جور نیست. این اسکریپت همان چهار قدم را به ترتیبِ درست انجام می‌دهد تا این
# مرحله فراموش نشود.
#
#   bash pipeline.sh <تعدادِ‌بیتِ‌آموزش>
set -u
cd "$(dirname "$0")"
N=${1:-12000}
CSV=/tmp/p3r/P3R.csv
LOG=/tmp/pipeline.log
: > "$LOG"

say() { echo -e "\n═══ $* ═══" | tee -a "$LOG"; }

say "۱/۴ دقتِ کنارگذاشته با رتبه‌بندِ فعلی، روی معناشناسیِ تازه"
python3 eval_p3r.py "$CSV" --n 400 --workers 4 2>&1 | tee -a "$LOG" | tail -14

say "۲/۴ استخراجِ ویژگی از $N بیت (دو تکه، نمونه از سرتاسرِ پیکره)"
# ★ دو نکته که هر کدام یک‌بار گران تمام شد:
#   --spread: بدونِ آن، «n ردیفِ نخست» برداشته می‌شود و چون پیکره بر پایهٔ
#     منبع مرتب است نمونه به‌شدت کج می‌شود (فاصلهٔ توزیع تا آزمون ۳۶٪ در برابر
#     ۲.۴٪). آن‌وقت اعتبارسنجی بالا می‌رود و دقتِ واقعی تکان نمی‌خورد.
#   دو تکه: کارِ پس‌زمینهٔ طولانی در این محیط گاهی نیمه‌کاره کشته می‌شود.
H=$(( N / 2 ))
python3 train_ranker.py extract "$CSV" --n "$H" --skip 0  --spread "$N" \
        --workers 3 --out /tmp/feats_a.npz 2>&1 | tee -a "$LOG" | tail -2
python3 train_ranker.py extract "$CSV" --n "$H" --skip "$H" --spread "$N" \
        --workers 3 --out /tmp/feats_b.npz 2>&1 | tee -a "$LOG" | tail -2
python3 train_ranker.py merge /tmp/feats_a.npz /tmp/feats_b.npz \
        --out /tmp/feats_new.npz 2>&1 | tee -a "$LOG" | tail -2

say "۳/۴ برازشِ شبکه (و مبنای خطی روی همان تفکیک)"
python3 train_mlp.py /tmp/feats_new.npz --hidden 32 \
        --out /tmp/ranker_new.json 2>&1 | tee -a "$LOG" | tail -8

# فقط اگر برازش موفق بود جایگزین کن — وگرنه رتبه‌بندِ کارکنان را خراب می‌کنیم
if [ -s /tmp/ranker_new.json ]; then
  cp ranker.json /tmp/ranker_prev.json
  cp /tmp/ranker_new.json ranker.json
  say "۴/۴ دقتِ کنارگذاشته با رتبه‌بندِ تازه"
  python3 eval_p3r.py "$CSV" --n 400 --workers 4 2>&1 | tee -a "$LOG" | tail -14
else
  say "برازش خروجی نداد — ranker.json دست‌نخورده ماند"
fi
echo -e "\n✅ پایان. گزارشِ کامل: $LOG"
