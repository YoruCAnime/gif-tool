"""Off-screen test of the tags, the Don't want box, Solo and finding again without a new scan.

Same invisible window as drive_gui.py, on a settings copy WITHOUT the new keys (like an older settings.json).
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "_devtest", "tags")
os.makedirs(OUT, exist_ok=True)
SET = os.path.join(OUT, "settings.json")
video = sys.argv[1]
with open(SET, "w", encoding="utf-8") as f:
    json.dump({"videos": [video], "description": "", "style": "Mix", "out": os.path.join(OUT, "gifs"), "length": 5}, f)
os.environ["CGM_SETTINGS"] = SET
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)

import app as appmod
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
fg_before = user32.GetForegroundWindow()
a = appmod.App()
a.attributes("-alpha", 0.0)
a.geometry("1320x800+30+30")
a.update()
_hwnd = user32.GetAncestor(a.winfo_id(), 2)
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.SetWindowLongPtrW(_hwnd, -20, user32.GetWindowLongPtrW(_hwnd, -20) | 0x80000 | 0x20 | 0x08000000)
user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos(_hwnd, ctypes.c_void_p(1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
user32.SetForegroundWindow(fg_before)
a.update()


def shot(name):
    a.update()
    r = wt.RECT()
    user32.GetWindowRect(_hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    dc = user32.GetWindowDC(_hwnd)
    mem = gdi32.CreateCompatibleDC(dc)
    bmp = gdi32.CreateCompatibleBitmap(dc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(_hwnd, mem, 2)
    buf = ctypes.create_string_buffer(w * h * 4)
    info = (ctypes.c_uint32 * 10)(40, w, -h & 0xFFFFFFFF, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, info, 0)
    Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA").convert("RGB").save(os.path.join(OUT, name))
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(_hwnd, dc)


def wait_idle(what, limit=500):
    t0 = time.time()
    a.update()
    while a.busy and time.time() - t0 < limit:
        a.update()
        time.sleep(0.03)
    for _ in range(12):
        a.update()
        time.sleep(0.02)
    took = time.time() - t0
    print("%-34s %5.1f s | %s" % (what, took, a.status.cget("text")), flush=True)
    return took


def top():
    return [(c["why"], round(c["seconds"]), round(c.get("wear_seconds", 0))) for c in a.chars[:3]]


# --- off by default, also with a settings file from before these options
assert a.neg_var.get() == "", "Don't want must start empty"
assert a.solo_var.get() is False, "Solo must start off"
print("defaults ok: Don't want empty, Solo off", flush=True)

a.on_find()
first = wait_idle("first Find (scan already saved)")
scans = a.scans
print("tags:", a.tags, flush=True)
assert a.tags, "no tags suggested"
print("top", top(), flush=True)
shot("1_found_tags.png")

# --- finding again never scans again
a.desc.delete("1.0", "end")
a.desc.insert("1.0", "dark green hair, girl")
a.on_find()
again = wait_idle("Find again, new description")
assert a.scans is scans, "it scanned again"
print("top", top(), flush=True)

# --- Don't want
a.desc.delete("1.0", "end")
a.neg_var.set("male")
a.on_find()
wait_idle("Don't want: male")
assert a.scans is scans
print("top", top(), flush=True)
a.neg_var.set("")

# --- a tag click adds the word and shows the people in it
a.desc.delete("1.0", "end")
a.use_tag("school uniform")
wait_idle("tag click: school uniform")
assert a.description() == "school uniform", a.description()
assert a.scans is scans
assert a.chars[0]["wear"], "nobody shown in that outfit"
print("top", top(), flush=True)
shot("2_school_uniform.png")
a.use_tag("school uniform")
wait_idle("tag click again takes it out")
assert a.description() == "", a.description()
a.use_tag("glasses", True)
wait_idle("right-click: don't want glasses")
assert a.neg_var.get() == "glasses", a.neg_var.get()
a.neg_var.set("")

# --- an outfit the show does not have: guesses, said as guesses
a.desc.delete("1.0", "end")
a.desc.insert("1.0", "nurse outfit")
a.on_find()
wait_idle("nurse outfit (not in this show)")
print("top", top(), "maybe", a.chars[0]["wear_maybe"], flush=True)
shot("3_nurse_guess.png")

# --- Solo on: real GIF. Then off again: normal GIF
a.desc.delete("1.0", "end")
a.desc.insert("1.0", "dark green hair, girl")
a.solo_var.set(True)
a.on_make()
wait_idle("Make GIF, Solo on")
assert a.job_solo
print("results", [(os.path.basename(r["path"]), round(r["bytes"] / 1e6, 2)) for r in a.results], flush=True)
assert a.results
shot("4_solo_made.png")
n = len(a.results)
a.solo_var.set(False)
a.on_make()
wait_idle("Make GIF, Solo off (several people allowed)")
assert not a.job_solo
assert len(a.results) == n + 1

a.on_close()
saved = json.load(open(SET, encoding="utf-8"))
print("saved: negative", repr(saved["negative"]), "solo", saved["solo"], flush=True)
print("OK", flush=True)
