"""Who is found for clothes words, with sample pictures, and how the score spreads: qa_wear.py VIDEO OUTDIR"""
import ctypes
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)
import numpy as np
from PIL import Image, ImageDraw

from cgm import match, scan

video, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
s = scan.scan_video(video)
chars = match.group_characters([s])
match.describe_characters(chars)
t0 = time.time()
match._track_clip([t for c in chars for _, t in c["members"]])
print("clip on every track: %.0f s, %d tracks" % (time.time() - t0, sum(len(c["members"]) for c in chars)), flush=True)
words = ["nurse outfit", "maid outfit", "sailor uniform", "school uniform", "white dress", "swimsuit", "armor", "straw hat", "glasses", "suit and tie", "kimono"]
tracks = [t for c in chars for _, t in c["members"]]
for w in words:
    p = match.wear_scores(tracks, w)
    print("%-16s tracks>=0.3: %3d  >=0.5: %3d  >=0.7: %3d  top %.2f" % (w, (p >= .3).sum(), (p >= .5).sum(), (p >= .7).sum(), p.max()), flush=True)
rows = []
for w in words:
    r = match.rank_characters(list(chars), w)
    who = [c for c in r if c["wear"]]
    print("%-16s %d characters: %s" % (w, len(who), [(c["why"], round(c["wear_seconds"])) for c in who[:6]]), flush=True)
    pics = [p for c in who[:5] for p in c["show"][:2]]
    rows.append(("%s - %d characters" % (w, len(who)), pics[:10]))
cell = 110
sheet = Image.new("RGB", (cell * 10, (cell + 16) * len(rows)))
dr = ImageDraw.Draw(sheet)
for i, (d, pics) in enumerate(rows):
    y = i * (cell + 16)
    dr.text((3, y + 2), d, fill="white")
    for k, p in enumerate(pics):
        sheet.paste(Image.fromarray(p).resize((cell, cell)), (k * cell, y + 16))
p = os.path.join(out, "wear_sheet.png")
sheet.save(p)
print(p)
print("--- words")
for d, n in [("no male", ""), ("girl, no male", ""), ("", "male"), ("brown hair", "male, glasses"), ("maid outfit, solo", ""), ("", "other people"),
             ("nurse outfit", "no maid outfit"), ("girl not blue hair", "")]:
    print(repr(d), repr(n), "pos", [x["text"] for x in match.split_description(d)], "neg", [x["text"] for x in match.negative_phrases(d, n)], "solo", match.wants_solo(d, n))
print("--- negatives")
for d, n in [("", ""), ("", "male"), ("girl", ""), ("", "female"), ("dark green hair", ""), ("dark green hair", "glasses"), ("", "black hair")]:
    r = match.rank_characters(list(chars), d, n)
    print(repr(d), repr(n), [(c["why"], round(c["seconds"]), c["match"]) for c in r[:4]], flush=True)
    rows.append((d + " / not " + n, [c["thumb"] for c in r[:4]]))
sheet = Image.new("RGB", (cell * 4, (cell + 16) * 7))
dr = ImageDraw.Draw(sheet)
for i, (d, pics) in enumerate(rows[-7:]):
    y = i * (cell + 16)
    dr.text((3, y + 2), d, fill="white")
    for k, pic in enumerate(pics):
        sheet.paste(Image.fromarray(pic).resize((cell, cell)), (k * cell, y + 16))
p = os.path.join(out, "negative_sheet.png")
sheet.save(p)
print(p)
