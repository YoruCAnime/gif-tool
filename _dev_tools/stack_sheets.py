"""Stacks the contact sheets of a test folder into one tall picture per few styles."""
import glob
import os
import sys

from PIL import Image, ImageDraw

folder = sys.argv[1]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
files = sorted(glob.glob(os.path.join(folder, "* sheet.png")))
for part in range(0, len(files), per):
    ims = [Image.open(f).convert("RGB") for f in files[part:part + per]]
    w = max(i.width for i in ims)
    h = sum(i.height + 18 for i in ims)
    out = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(out)
    y = 0
    for f, i in zip(files[part:part + per], ims):
        d.text((4, y + 3), os.path.basename(f)[:-10], fill="white")
        out.paste(i, (0, y + 18))
        y += i.height + 18
    p = os.path.join(folder, "_stack%d.png" % (part // per))
    out.save(p)
    print(p)
