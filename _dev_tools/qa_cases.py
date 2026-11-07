"""Edge cases and other characters: qa_cases.py EPISODE SHORTCLIP WORKDIR"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_IDLE"] = "1"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import cv2

cv2.setNumThreads(1)

from cgm import gifenc, match, render, scan, styles

episode, short, work = sys.argv[1:4]
os.makedirs(work, exist_ok=True)


def run(label, video, description, style="Mix", **opts):
    t0 = time.time()
    print("---", label, flush=True)
    try:
        s = scan.scan_video(video)
        chars = match.rank_characters(match.group_characters([s]), description)
        if not chars:
            print("   no characters (message the app shows: No anime faces were found)")
            return
        c = chars[0]
        print("   tracks", len(s["tracks"]), "characters", len(chars), "top:", c["match"], c["why"], "%.0f s" % c["seconds"], flush=True)
        opt = render.Options()
        for k, v in opts.items():
            setattr(opt, k, v)
        out = os.path.join(work, label + ".gif")
        r = render.make_gif(c, match.outfit_filter(c, description), [s], styles.by_name(style), opt, out, seed=7, log=lambda m: print("   ", m, flush=True))
        print("   %.2f MB %dx%d %d frames %.2f s took %.0f s shots %d" % (r["bytes"] / 1e6, r["w"], r["h"], r["frames"], r["seconds"], time.time() - t0, r["shots"]), flush=True)
    except Exception as e:
        print("   ERROR", type(e).__name__, e, flush=True)


# exact timing of the GIF writer
print("delays at 33 fps:", gifenc.delays_for(6, 33), " 25 fps:", gifenc.delays_for(3, 25))
if not os.path.isfile(short):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "300", "-i", episode, "-t", "14", "-c", "copy", short], check=True)
if "--all" in sys.argv:
    run("man black hair", episode, "black hair, stubble, man", "Slide & settle")
    run("green hair girl", episode, "girl with dark green hair", "Zoom punch")
    run("wide square", episode, "girl with dark green hair", "Push slide", aspect=1.7778, length=4.0)
    run("phone tall", episode, "girl with dark green hair", "Dreamy drift", aspect=0.5625, height=600, fps=20, max_mb=8.0)
run("short clip", short, "")
blank = os.path.join(work, "no_faces.mp4")
if not os.path.isfile(blank):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=24", blank], check=True)
run("no faces", blank, "red hair")
