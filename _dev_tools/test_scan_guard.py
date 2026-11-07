"""One real scan under the watchdog, into a test folder: test_scan_guard.py VIDEO"""
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

from cgm import scan, watchdog

video = sys.argv[1]
scan.CACHE = os.path.join(ROOT, "_devtest", "scan_guard")
path = scan._cache_path(video)
if os.path.isfile(path):
    os.remove(path)
t0 = time.time()
with watchdog.Guard(say=lambda m: print("watchdog:", m, flush=True)) as guard:
    s = scan.scan_video(video)
print("scan %.0f s, %d faces tracks, %d cuts" % (time.time() - t0, len(s["tracks"]), len(s["cuts"])))
print("peaks", {k: round(v, 2) for k, v in guard.peak.items()})
assert s["tracks"], "no faces found"
print("OK")
