"""How reliable is Solo? Counts what passes on a real video and saves the frames each shot starts, middles and ends on.
qa_solo.py VIDEO OUTDIR "description" ...
"""
import ctypes
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)
import numpy as np

from cgm import match, render, scan, styles

video, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
s = scan.scan_video(video)
chars = match.group_characters([s])
tiles = []
for n, desc in enumerate(sys.argv[3:]):
    char = match.rank_characters(list(chars), desc)[0]
    opt = render.Options()
    opt.solo, opt.length, opt.height = True, 6.0, 360
    spec = {"close": True, "dur": 0.6}
    own = {id(t) for _, t in char["members"]}
    # every candidate the solo rule could use, with the verdict on its real frames
    cands = render.candidates(char["members"], 0.6)
    rng = random.Random(3)
    rng.shuffle(cands)
    kept, dropped = [], []
    for c in cands[:40]:
        shot = render.load_shot(c, s, spec, opt, 0.6)
        if shot is None:
            continue
        (kept if render.alone(shot) else dropped).append(shot)
    print("%-28s %s, candidates %d, checked %d: alone %d, someone else %d" % (desc, char["why"], len(cands), len(kept) + len(dropped), len(kept), len(dropped)), flush=True)

    def sheet(shots, name):
        rows = []
        for sh in shots[:24]:
            i = len(sh.frames) // 2
            w, h = sh.cw / 2, sh.ch / 2
            x, y = max(0, int(sh.cx[i] - w)), max(0, int(sh.cy[i] - h))
            rows.append(cv2.resize(sh.frames[i][y:int(sh.cy[i] + h), x:int(sh.cx[i] + w)], (150, 166)))
        if not rows:
            return
        cols = 8
        while len(rows) % cols:
            rows.append(np.zeros_like(rows[0]))
        grid = np.vstack([np.hstack(rows[k:k + cols]) for k in range(0, len(rows), cols)])
        cv2.imwrite(os.path.join(out, name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    sheet(kept, "%d_alone.png" % n)
    sheet(dropped, "%d_dropped.png" % n)
    # the whole thing: a real GIF with solo on
    result = render.make_gif(char, match.outfit_filter(char, desc), [s], styles.by_name("Mix"), opt, os.path.join(out, "%d_solo.gif" % n), seed=5, log=lambda m: print("   ", m, flush=True))
    print("    gif", round(result["bytes"] / 1e6, 2), "MB", flush=True)
print("--- suggested tags")
import time
t0 = time.time()
print(match.suggest_tags(chars), "%.1f s" % (time.time() - t0))
