"""Scan one video, group its characters, print the ranking for a description, save a picture sheet."""
import ctypes
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)

import cv2
import numpy as np

cv2.setNumThreads(1)

from cgm import match, scan

video = sys.argv[1]
descriptions = sys.argv[2:]
t0 = time.time()
last = [0]


def prog(p, msg):
    if time.time() - last[0] > 20:
        last[0] = time.time()
        print("%3d%%  %s  (%.0f s)" % (p * 100, msg, time.time() - t0), flush=True)


s = scan.scan_video(video, prog)
print("scan", round(time.time() - t0), "s  tracks", len(s["tracks"]), "cuts", len(s["cuts"]), "bars", s["bars"], s["info"], flush=True)
t0 = time.time()
chars = match.group_characters([s])
print("characters", len(chars), round(time.time() - t0, 1), "s", flush=True)
match.describe_characters(chars)
print("described", round(time.time() - t0, 1), "s", flush=True)

out = os.path.join(ROOT, "_devtest")
os.makedirs(out, exist_ok=True)
cols = 4
rows = min(14, len(chars))
cell = 120
sheet = np.zeros((rows * cell, cols * cell + 330, 3), np.uint8)
for i, c in enumerate(chars[:rows]):
    for k, th in enumerate(c["thumbs"][:cols]):
        sheet[i * cell:(i + 1) * cell, k * cell:(k + 1) * cell] = cv2.resize(th, (cell, cell))
    near = lambda lab: "?" if lab is None else max(match.COLOURS, key=lambda n: match.colour_score(lab, n))
    line = "#%d %.0fs tracks %d" % (i, c["seconds"], len(c["members"]))
    cv2.putText(sheet, line, (cols * cell + 6, i * cell + 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(sheet, "hair %s  eye %s" % (near(c["hair"]), near(c["eye"])), (cols * cell + 6, i * cell + 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(out, "characters.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print("sheet", os.path.join(out, "characters.png"))
ids = {id(c): i for i, c in enumerate(chars)}
for d in descriptions:
    ranked = match.rank_characters(list(chars), d)
    print(repr(d), match.split_description(d))
    for c in ranked[:4]:
        print("   #%d match %d score %.2f  %s" % (ids[id(c)], c["match"], c["score"], c["why"]))
