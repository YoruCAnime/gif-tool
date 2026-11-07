"""Reads a video once: scene cuts, every face, face tracks, and what each track looks like.

The result is cached per file (path + size + date), so the same video opens instantly next time.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

from . import models

SAMPLE_FPS = 5.0
SCAN_H = 540
MIN_FACE = 0.07  # of the frame height
SCAN_VERSION = 2
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
BELOW_NORMAL = 0x00004000 if sys.platform == "win32" else 0
IDLE = 0x00000040 if sys.platform == "win32" else 0


def proc_flags():
    return NO_WINDOW | (IDLE if os.environ.get("CGM_IDLE") else BELOW_NORMAL)


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate:format=duration", "-of", "json", path],
        capture_output=True, text=True, errors="replace", creationflags=NO_WINDOW)
    if r.returncode:
        raise RuntimeError("This file could not be read as a video:\n" + r.stderr[-300:])
    d = json.loads(r.stdout)
    s = d["streams"][0]

    def rate(x):
        a, _, b = x.partition("/")
        return float(a) / float(b or 1) if float(b or 1) else 0.0

    fps = rate(s.get("avg_frame_rate", "0/1")) or rate(s.get("r_frame_rate", "0/1")) or 24.0
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": fps, "dur": float(d["format"].get("duration", 0) or 0)}


def _cache_path(path):
    st = os.stat(path)
    key = "%s|%d|%d|%d" % (os.path.abspath(path).lower(), st.st_size, int(st.st_mtime), SCAN_VERSION)
    return os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".scan")


def _jpg(rgb, q=88):
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return b.tobytes()


def unjpg(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def crop_pad(img, x1, y1, x2, y2):
    """Crop that may go past the picture; black fills the outside."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    l, t, r, b = max(0, -x1), max(0, -y1), max(0, x2 - w), max(0, y2 - h)
    c = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if l or t or r or b:
        c = cv2.copyMakeBorder(c, t, b, l, r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return c


def head_box(b, grow=1.6):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(b[2] - b[0], b[3] - b[1]) * grow / 2
    return cx - s, cy - s * 1.05, cx + s, cy + s * 0.95


def bust_box(b):
    cx = (b[0] + b[2]) / 2
    fw, fh = b[2] - b[0], b[3] - b[1]
    s = max(fw, fh) * 1.9
    return cx - s, b[1] - fh * 0.7, cx + s, b[1] - fh * 0.7 + 2 * s


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def _close_scene(buf, scene, tracks, ccip):
    """buf: [(sample index, frame rgb, [(box, score, sharp)...])] of one scene -> face tracks."""
    open_tracks = []
    done = []
    for idx, frame, faces in buf:
        used = set()
        for tr in open_tracks:
            if tr["last"] < idx - 2:
                continue
            best, bi = 0.15, -1
            for k, f in enumerate(faces):
                if k in used:
                    continue
                v = _iou(tr["boxes"][-1], f[0])
                if v > best:
                    best, bi = v, k
            if bi >= 0:
                used.add(bi)
                tr["idx"].append(idx)
                tr["boxes"].append(faces[bi][0])
                tr["score"].append(faces[bi][1])
                tr["sharp"].append(faces[bi][2])
                tr["last"] = idx
        for k, f in enumerate(faces):
            if k not in used:
                t = {"idx": [idx], "boxes": [f[0]], "score": [f[1]], "sharp": [f[2]], "last": idx}
                open_tracks.append(t)
                done.append(t)
    frames = {i: fr for i, fr, _ in buf}
    heads, owners = [], []
    for tr in done:
        if len(tr["idx"]) < 2 and tr["score"][0] < 0.6:
            continue
        n = len(tr["idx"])
        weight = np.array(tr["sharp"]) * np.array([(b[3] - b[1]) for b in tr["boxes"]])
        picks = sorted(set([int(np.argmax(weight)), n // 4, 3 * n // 4]))[:3]
        tr["reps"] = []
        for p in picks:
            fr, b = frames[tr["idx"][p]], tr["boxes"][p]
            h = crop_pad(fr, *head_box(b))
            bust = cv2.resize(crop_pad(fr, *bust_box(b)), (224, 224), interpolation=cv2.INTER_AREA)
            rel_h = (b[3] - b[1]) / fr.shape[0]
            tr["reps"].append({"head": _jpg(cv2.resize(h, (192, 192), interpolation=cv2.INTER_AREA)),
                               "bust": _jpg(bust, 82), "h": rel_h})
            heads.append(h)
            owners.append(tr)
    if not heads:
        return
    feats = ccip.embed(heads)
    for tr, f in zip(owners, feats):
        tr.setdefault("feats", []).append(f)
    fh, fw = buf[0][1].shape[:2]
    for tr in done:
        if "feats" not in tr:
            continue
        tracks.append({
            "scene": scene,
            "t": np.array(tr["idx"], np.float32) / SAMPLE_FPS,
            "boxes": (np.array(tr["boxes"], np.float32) / np.array([fw, fh, fw, fh], np.float32)),
            "score": np.array(tr["score"], np.float32),
            "sharp": np.array(tr["sharp"], np.float32),
            "reps": tr["reps"],
            "feat": np.mean(tr["feats"], axis=0),
        })


def scan_video(path, progress=None, stop=None):
    """Returns {'path', 'info', 'tracks', 'cuts', 'bars'} (cached)."""
    cpath = _cache_path(path)
    if os.path.isfile(cpath):
        with open(cpath, "rb") as f:
            return pickle.load(f)
    info = probe(path)
    sw = int(round(info["w"] * SCAN_H / info["h"] / 2) * 2)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", path, "-map", "0:v:0", "-an", "-sn",
           "-vf", "fps=%g,scale=%d:%d:flags=bilinear" % (SAMPLE_FPS, sw, SCAN_H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=proc_flags(), bufsize=sw * SCAN_H * 3 * 4)
    det = models.get("face")
    ccip = models.get("ccip")
    size = sw * SCAN_H * 3
    idx = 0
    scene = 0
    prev = None
    buf, tracks, cuts = [], [], []
    rows = np.zeros(SCAN_H, np.float32)
    cols = np.zeros(sw, np.float32)
    expected = max(1, int(info["dur"] * SAMPLE_FPS))
    try:
        while True:
            raw = p.stdout.read(size)
            if len(raw) < size:
                break
            if stop is not None and stop.is_set():
                raise InterruptedError
            frame = np.frombuffer(raw, np.uint8).reshape(SCAN_H, sw, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if idx % 5 == 0:
                rows = np.maximum(rows, gray.mean(axis=1))
                cols = np.maximum(cols, gray.mean(axis=0))
            thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None and float(np.abs(thumb - prev).mean()) > 22:
                _close_scene(buf, scene, tracks, ccip)
                buf = []
                scene += 1
                cuts.append(idx / SAMPLE_FPS)
            prev = thumb
            faces = []
            if gray.mean() > 8:
                for x1, y1, x2, y2, s in det.detect(frame):
                    if (y2 - y1) / SCAN_H < MIN_FACE:
                        continue
                    roi = gray[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                    sharp = float(cv2.Laplacian(roi, cv2.CV_32F).var()) if roi.size else 0.0
                    faces.append(((x1, y1, x2, y2), s, sharp))
            if faces:
                buf.append((idx, frame, faces))
            # a scene with no end must not fill the memory
            if len(buf) > 300:
                _close_scene(buf, scene, tracks, ccip)
                buf = []
                scene += 1
            idx += 1
            if progress and idx % 25 == 0:
                progress(min(0.999, idx / expected), "Reading the video: %d faces tracks so far" % len(tracks))
        _close_scene(buf, scene, tracks, ccip)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
    if not idx:
        raise RuntimeError("No frames could be read from this video.")
    # black bars: rows / columns that are never bright
    top = int(np.argmax(rows > 14))
    bot = int(np.argmax(rows[::-1] > 14))
    lef = int(np.argmax(cols > 14))
    rig = int(np.argmax(cols[::-1] > 14))
    res = {"path": path, "info": info, "tracks": tracks, "cuts": cuts,
           "bars": (lef / sw, top / SCAN_H, rig / sw, bot / SCAN_H)}
    os.makedirs(CACHE, exist_ok=True)
    tmp = cpath + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, cpath)
    return res
