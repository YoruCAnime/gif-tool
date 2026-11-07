"""Make GIFs for a description: test_make.py VIDEO "DESCRIPTION" OUTDIR [style names...] (no names = all + Mix).

Prints the numbers and writes a contact sheet per GIF so the result can be judged by eye.
"""
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

from PIL import Image, ImageDraw, ImageSequence

from cgm import match, render, scan, styles

video, description, outdir = sys.argv[1:4]
names = sys.argv[4:] or [s.name for s in styles.ALL] + [styles.MIX.name]
s = scan.scan_video(video)
chars = match.rank_characters(match.group_characters([s]), description)
char = chars[0]
print("character", char["match"], char["why"], "seconds", round(char["seconds"]), "tracks", len(char["members"]), flush=True)
os.makedirs(outdir, exist_ok=True)
Image.fromarray(char["thumb"]).save(os.path.join(outdir, "_character.png"))
members = match.outfit_filter(char, description)
used = set()
opt = render.Options()
for k, name in enumerate(names, 1):
    t0 = time.time()
    path = os.path.join(outdir, "%02d %s.gif" % (k, name.replace("&", "and")))
    r = render.make_gif(char, members, [s], styles.by_name(name), opt, path, seed=1000 + k, used=used, log=lambda m: print("   ", m, flush=True))
    print("%-16s %5.2f MB %dx%d %d frames %.2f s, took %.0f s, same %s (limit %.2f) at %s" % (
        name, r["bytes"] / 1e6, r["w"], r["h"], r["frames"], r["seconds"], time.time() - t0, r["same"], r["threshold"], r["times"]), flush=True)
    im = Image.open(path)
    fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    n = 21
    idx = [round(i * (len(fr) - 1) / (n - 1)) for i in range(n)]
    tw, th = 150, round(150 * fr[0].height / fr[0].width)
    sheet = Image.new("RGB", (7 * tw, 3 * th))
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(idx):
        x, y = (j % 7) * tw, (j // 7) * th
        sheet.paste(fr[i].resize((tw, th)), (x, y))
        draw.text((x + 3, y + 1), str(i), fill="yellow")
    sheet.save(path[:-4] + " sheet.png")
