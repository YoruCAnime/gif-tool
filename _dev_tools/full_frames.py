"""Full-size frames from GIFs, to judge the picture itself: full_frames.py OUT.png GIF:frame GIF:frame ..."""
import sys

from PIL import Image, ImageSequence

out = sys.argv[1]
pics = []
for arg in sys.argv[2:]:
    path, _, n = arg.rpartition(":")
    fr = list(ImageSequence.Iterator(Image.open(path)))
    pics.append(fr[int(n)].convert("RGB").copy())
w = sum(p.width for p in pics)
sheet = Image.new("RGB", (w, max(p.height for p in pics)))
x = 0
for p in pics:
    sheet.paste(p, (x, 0))
    x += p.width
sheet.save(out)
print(out, sheet.size)
