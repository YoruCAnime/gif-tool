"""Off-screen test of the real window: Find her, change the choice, Make GIF, take pictures of OUR window only.

The window is on screen but invisible (alpha 0), click-through and behind every other window, because
Tk windows far off screen paint nothing for PrintWindow. It uses a settings copy in _devtest.
"""
import ctypes
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "_devtest", "gui")
os.makedirs(OUT, exist_ok=True)
SET = os.path.join(OUT, "settings.json")
video, description = sys.argv[1], sys.argv[2]
with open(SET, "w", encoding="utf-8") as f:
    json.dump({"videos": [video], "description": description, "style": "Mix", "out": os.path.join(OUT, "gifs"), "length": 5}, f)
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
GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE = -20, 0x80000, 0x20, 0x08000000
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.SetWindowLongPtrW(_hwnd, GWL_EXSTYLE, user32.GetWindowLongPtrW(_hwnd, GWL_EXSTYLE) | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos(_hwnd, ctypes.c_void_p(1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_BOTTOM, no size/move/activate
user32.SetForegroundWindow(fg_before)
a.update()


def shot(name):
    a.update()
    hwnd = user32.GetAncestor(a.winfo_id(), 2)
    r = ctypes.wintypes.RECT() if hasattr(ctypes, "wintypes") else None
    import ctypes.wintypes as wt
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    dc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(dc)
    bmp = gdi32.CreateCompatibleBitmap(dc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 2)
    buf = ctypes.create_string_buffer(w * h * 4)
    info = (ctypes.c_uint32 * 10)(40, w, -h & 0xFFFFFFFF, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, info, 0)
    Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA").convert("RGB").save(os.path.join(OUT, name))
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, dc)
    print("picture", name, w, h, flush=True)


def wait_idle(limit=600):
    t0 = time.time()
    a.update()
    while a.busy and time.time() - t0 < limit:
        a.update()
        time.sleep(0.03)
    for _ in range(10):
        a.update()
        time.sleep(0.02)
    print("status:", a.status.cget("text"), "(%.0f s)" % (time.time() - t0), flush=True)


shot("1_start.png")
a.on_find()
wait_idle()
print("characters", len(a.chars), "top match", a.chars[0]["match"], a.chars[0]["why"], flush=True)
shot("2_found.png")
# click the second character, then back to the first
a.pick(a.chars[1])
assert a.selected is a.chars[1]
shot("3_picked_second.png")
a.pick(a.chars[0])
a.on_make()
wait_idle()
print("results", [(os.path.basename(r["path"]), round(r["bytes"] / 1e6, 2)) for r in a.results], flush=True)
assert a.results, "no GIF was made"
t0 = time.time()
while time.time() - t0 < 2.0:
    a.update()
    time.sleep(0.02)
shot("4_made.png")
# Stop must work during a job
a.on_make()
for _ in range(30):
    a.update()
    time.sleep(0.02)
a.stop_flag.set()
wait_idle()
print("after stop:", a.status.cget("text"), flush=True)
# save + close
a.on_close()
print("settings saved:", json.load(open(SET, encoding="utf-8")).get("length"), flush=True)
print("OK", flush=True)
