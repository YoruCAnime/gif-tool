"""Same frames, different GIF encodings: sizes plus a 2x crop of each to compare by eye."""
import ctypes
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from cgm import gifenc, match, render, scan, styles

video, description = sys.argv[1:3]
OUT = os.path.join(ROOT, "_devtest", "dither")
os.makedirs(OUT, exist_ok=True)
s = scan.scan_video(video)
char = match.rank_characters(match.group_characters([s]), description)[0]
members = match.core_members(char["members"])
opt = render.Options()
style = styles.by_name("Slide & settle")
rng = random.Random(5)
specs = style.plan(style.count(opt.length), rng)
moments = render.pick_moments(members, len(specs), 0.65, rng)
shots = [render.load_shot(c, s, sp, opt, sp["dur"]) for c, sp in zip(moments, specs)]
frames, segs = render.render(shots, specs, style, opt)
print("frames", len(frames), frames[0].shape, flush=True)
gifenc.DITHER["none"] = "none"
gifenc.DITHER["medium"] = "bayer:bayer_scale=4"
gifenc.DITHER["bayer1"] = "bayer:bayer_scale=1"
gifenc.DITHER["bayer3"] = "bayer:bayer_scale=3"
crops = []
for d in ["fine", "medium", "bayer3", "bayer1", "none"]:
    work = os.path.join(OUT, "w_" + d)
    os.makedirs(work, exist_ok=True)
    groups = []
    for f, sg in zip(frames, segs):
        if not groups or groups[-1][0] != sg:
            groups.append((sg, []))
        groups[-1][1].append(f)
    paths = []
    for i, (_, fr) in enumerate(groups):
        p = os.path.join(work, "%03d.gif" % i)
        gifenc._segment(fr, opt.fps, p, 1.0, d)
        paths.append(p)
    out = os.path.join(OUT, d + ".gif")
    gifenc._join(paths, out, gifenc.delays_for(len(frames), opt.fps))
    shutil.rmtree(work, ignore_errors=True)
    print("%-8s %6.2f MB" % (d, os.path.getsize(out) / 1e6), flush=True)
    fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(out))]
    # a calm frame with a soft gradient, 2x
    pic = fr[12]
    w, h = pic.size
    c = pic.crop((w // 4, h // 6, w // 4 + 200, h // 6 + 240)).resize((400, 480), Image.NEAREST)
    ImageDraw.Draw(c).text((4, 4), d, fill="yellow")
    crops.append(c)
sheet = Image.new("RGB", (400 * len(crops), 480))
for i, c in enumerate(crops):
    sheet.paste(c, (i * 400, 0))
sheet.save(os.path.join(OUT, "compare.png"))
print(os.path.join(OUT, "compare.png"))
