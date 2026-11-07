"""Shows who wins for some descriptions and what the clothes filter keeps: qa_outfit.py EPISODE OUTDIR"""
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)
import numpy as np
from PIL import Image, ImageDraw

from cgm import match, scan
from cgm.scan import unjpg

video, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
s = scan.scan_video(video)
chars = match.group_characters([s])
rows = []
for d in ["light blue hair, horns, purple eyes", "light blue hair", "blonde man with a hat"]:
    r = match.rank_characters(list(chars), d)
    print(d, [(c["why"], round(c["seconds"]), c["match"]) for c in r[:3]], flush=True)
    rows.append((d, [c["thumb"] for c in r[:3]]))
    print(flush=True)
for d in ["dark green hair, girl, wearing a white dress", "dark green hair, girl, wearing a straw hat", "dark green hair, girl, sailor uniform"]:
    r = match.rank_characters(list(chars), d)
    c = r[0]
    kept = match.outfit_filter(c, d)
    print(d, "tracks", len(c["members"]), "kept", len(kept), flush=True)
    rows.append((d, [unjpg(t["reps"][0]["bust"]) for _, t in kept[:9]]))
cell = 110
sheet = Image.new("RGB", (cell * 10, (cell + 16) * len(rows)))
dr = ImageDraw.Draw(sheet)
for i, (d, pics) in enumerate(rows):
    y = i * (cell + 16)
    dr.text((3, y + 2), d, fill="white")
    for k, p in enumerate(pics):
        sheet.paste(Image.fromarray(p).resize((cell, cell)), (k * cell, y + 16))
p = os.path.join(out, "who_and_clothes.png")
sheet.save(p)
print(p)
