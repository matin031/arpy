# -*- coding: utf-8 -*-
"""
p3r.py — کارِ مشترک با پیکرهٔ برچسب‌خوردهٔ P3R

P3R.csv  =  VERSE1, VERSE2, PROSODY, PROSODY_ID   (۱.۳۴ میلیون بیتِ برچسب‌خورده)
    https://media.githubusercontent.com/media/m-shahrestani/Prosody-Recognition-in-Persian-Poetry/master/dataset/P3R.csv

★ تفکیکِ آموزش/آزمون
   واژه‌نامه نباید روی بیت‌هایی آموزش ببیند که بعد با آن‌ها محک می‌زنیم،
   وگرنه عددِ دقت خوش‌بینانه و دروغ است. تفکیک بر پایهٔ هشِ متنِ بیت است:
   قطعی، بازتولیدپذیر، و مستقل از ترتیبِ خواندن.
"""
import csv, hashlib, sys

TEST_FRACTION = 20          # یک بیت از هر ۲۰ تا → مجموعهٔ آزمون (۵٪)


def split_of(v1, v2):
    """→ 'test' یا 'train' — قطعی و بازتولیدپذیر."""
    h = hashlib.md5((v1 + "|" + v2).encode("utf-8")).digest()
    return "test" if (h[0] | (h[1] << 8)) % TEST_FRACTION == 0 else "train"


def rows(path, split=None, limit=None):
    """پیمایشِ پیکره → (مصراع۱، مصراع۲، ارکان).
       split=None یعنی همه؛ 'train' / 'test' برای تفکیک."""
    csv.field_size_limit(1 << 24)
    n = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v1 = (row.get("VERSE1") or "").strip()
            v2 = (row.get("VERSE2") or "").strip()
            ark = (row.get("PROSODY") or "").strip()
            if not v1 or not ark:
                continue
            if split and split_of(v1, v2) != split:
                continue
            yield v1, v2, ark
            n += 1
            if limit and n >= limit:
                return
