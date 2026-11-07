"""Checks what the guide promises: descriptions, several episodes at once, clothes words.

qa_promises.py EPISODE_A EPISODE_B OUTDIR
"""
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)
from PIL import Image, ImageDraw, ImageSequence

from cgm import match, render, scan, styles

ep_a, ep_b, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)


def sheet(path):
    fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(path))]
    idx = [round(i * (len(fr) - 1) / 13) for i in range(14)]
    tw, th = 150, round(150 * fr[0].height / fr[0].width)
    s = Image.new("RGB", (7 * tw, 2 * th))
    d = ImageDraw.Draw(s)
    for j, i in enumerate(idx):
        s.paste(fr[i].resize((tw, th)), ((j % 7) * tw, (j // 7) * th))
        d.text(((j % 7) * tw + 3, (j // 7) * th + 1), str(i), fill="yellow")
    s.save(path[:-4] + " sheet.png")


a = scan.scan_video(ep_a)
chars = match.group_characters([a])
for d in ["light blue hair, horns, purple eyes", "light blue hair", "dark blue hair, glasses", "girl with dark green hair",
          "black hair, stubble, man", "blonde man with a hat", "pink hair girl"]:
    r = match.rank_characters(list(chars), d)
    print("%-38s -> %s (%.0f s on screen), next: %s" % (d, r[0]["why"], r[0]["seconds"], r[1]["why"]), flush=True)
    Image.fromarray(r[0]["thumb"]).save(os.path.join(out, "who - %s.png" % d.replace(",", "")))

print("--- clothes words", flush=True)
r = match.rank_characters(list(chars), "dark green hair, girl, wearing a white dress")
c = r[0]
kept = match.outfit_filter(c, "dark green hair, girl, wearing a white dress")
print("tracks", len(c["members"]), "kept", len(kept), flush=True)
p = os.path.join(out, "white dress.gif")
print(render.make_gif(c, kept, [a], styles.by_name("Slide & settle"), render.Options(), p, seed=3)["times"], flush=True)
sheet(p)

print("--- two episodes", flush=True)
b = scan.scan_video(ep_b, lambda p, m: None)
both = match.group_characters([a, b])
r = match.rank_characters(both, "black hair, stubble, man")
c = r[0]
print("characters", len(both), "top", c["why"], "%.0f s" % c["seconds"], "tracks from A", sum(1 for v, _ in c["members"] if v == 0), "from B", sum(1 for v, _ in c["members"] if v == 1), flush=True)
p = os.path.join(out, "two episodes.gif")
res = render.make_gif(c, c["members"], [a, b], styles.by_name("Mix"), render.Options(), p, seed=4)
print(res["shots"], "shots, same", res["same"], flush=True)
sheet(p)
print("OK", flush=True)
