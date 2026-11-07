"""Picks the moments and renders the montage: face-following camera + style + motion blur."""
import os
import random
import subprocess
import time

import cv2
import numpy as np

from . import gifenc, match, models, styles
from .scan import CACHE, crop_pad, head_box, proc_flags, unjpg


class Options:
    length = 6.0      # seconds of GIF
    height = 480
    aspect = 0.9      # width / height
    fps = 25
    pace = 1.0        # shot length multiplier
    max_mb = 15.0
    solo = False      # only moments with no other face on screen


# ---------------------------------------------------------------------------- choosing the moments
def _others_on_screen(t, others):
    """For each sample of track t: is another face visible around then? others = tracks of that video and scene."""
    busy = np.zeros(len(t["t"]), bool)
    for o in others:
        busy |= (t["t"] >= o["t"][0] - 0.25) & (t["t"] <= o["t"][-1] + 0.25)
    return busy


def candidates(members, want_dur, company=None):
    """The best stretch of each face track; long tracks give several.

    company(vi, t) -> the other tracks to keep off screen (solo); None = anyone may be there."""
    cands = []
    all_sharp = np.concatenate([t["sharp"] for _, t in members])
    ref = float(np.percentile(all_sharp, 70)) + 1.0
    for vi, t in members:
        times = t["t"]
        if len(times) < 2:
            continue
        span = float(times[-1] - times[0])
        if span < 0.35:
            continue
        hs = t["boxes"][:, 3] - t["boxes"][:, 1]
        # the sliding window with the biggest and sharpest face
        win = max(1, int(round(min(want_dur, span) * 5)))
        q = hs * np.sqrt(np.minimum(t["sharp"] / ref, 1.5)) * t["score"]
        run = np.convolve(q, np.ones(win) / win, "valid")
        if company is not None:
            busy = _others_on_screen(t, company(vi, t))
            run[np.convolve(busy.astype(np.float32), np.ones(win), "valid") > 0] = -1.0
        # a long scene gives one moment per stretch of about 2 shots
        parts = max(1, int(span // max(1.5, want_dur * 2)))
        for part in range(parts):
            lo = part * len(run) // parts
            hi = max(lo + 1, (part + 1) * len(run) // parts)
            i = lo + int(np.argmax(run[lo:hi]))
            if run[i] < 0:
                continue
            cands.append(_candidate(vi, t, i, win, run[i], want_dur, span))
    cands.sort(key=lambda c: -c["score"])
    return cands


def _candidate(vi, t, i, win, quality, want_dur, span):
    t0 = float(t["t"][i])
    b = t["boxes"][i:i + win + 1]
    h = float(np.median(b[:, 3] - b[:, 1]))
    edge = float(np.mean((b[:, 0] < 0.004) | (b[:, 2] > 0.996) | (b[:, 1] < 0.004) | (b[:, 3] > 0.996)))
    drift = float(np.abs(np.diff(b[:, :2], axis=0)).mean()) if len(b) > 1 else 0.0
    conf = float(np.median(t["score"][i:i + win + 1]))
    if "_bright" not in t:
        t["_bright"] = float(np.mean([unjpg(r["head"]).mean() for r in t["reps"]]))
    score = (1.5 * min(1.0, h / 0.32) + 0.7 * min(1.0, float(quality) / max(1e-4, h)) + 0.8 * min(1.0, span / want_dur)
             + 1.0 * conf - 2.0 * edge - 3 * drift - 1.2 * max(0.0, (70 - t["_bright"]) / 70))
    return {"vi": vi, "track": t, "t0": t0, "h": h, "score": score, "scene": (vi, t["scene"]), "time": t0,
            "conf": conf, "edge": edge, "bright": t["_bright"]}


def moment_id(c):
    return (id(c["track"]), int(c["t0"] * 5))


def pick_moments(members, n, want_dur, rng, used=None, company=None):
    """Good moments that are spread out: one per scene first, then allow repeats."""
    used = used if used is not None else set()
    cands = candidates(members, want_dur, company)
    if not cands:
        return []
    for rule in (lambda c: 0.14 <= c["h"] <= 0.62 and c["conf"] >= 0.6 and c["edge"] < 0.3 and c["bright"] >= 55,
                 lambda c: 0.10 <= c["h"] <= 0.7 and c["conf"] >= 0.5 and c["edge"] < 0.5,
                 lambda c: True):
        good = [c for c in cands if rule(c)]
        if len(good) >= n:
            break
    fresh = [c for c in good if moment_id(c) not in used]
    if len(fresh) < n:
        fresh = fresh + [c for c in good if moment_id(c) in used]
    pool = fresh[:max(n * 2, 16)]
    # weighted draw: better moments are likelier, but each run differs
    weights = np.exp((np.array([c["score"] for c in pool]) - pool[0]["score"]) * 2.5)
    chosen, scenes, taken = [], set(), set()
    order = list(range(len(pool)))
    for strict in (True, False):
        left = [i for i in order if i not in taken]
        while left and len(chosen) < n:
            p = weights[left] / weights[left].sum()
            i = rng.choices(left, weights=p.tolist())[0]
            left.remove(i)
            c = pool[i]
            if strict and c["scene"] in scenes:
                continue
            taken.add(i)
            chosen.append(c)
            scenes.add(c["scene"])
    chosen = chosen[:n]
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------- loading a shot
class Shot:
    pass


def _read_frames(path, start, dur, w, h):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % max(0.0, start), "-i", path, "-t", "%.3f" % dur,
           "-map", "0:v:0", "-an", "-sn", "-vf", "scale=%d:%d:flags=lanczos" % (w, h), "-threads", "2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True, creationflags=proc_flags())
    data = p.stdout
    n = len(data) // (w * h * 3)
    return np.frombuffer(data, np.uint8, n * w * h * 3).reshape(n, h, w, 3)


def load_shot(cand, scan, spec, opt, need):
    """Decodes the moment, keeps it inside one scene, and works out the camera path."""
    info, bars = scan["info"], scan["bars"]
    t = cand["track"]
    W, H = info["w"], info["h"]
    # usable picture without black bars (all in source pixels)
    ux1, uy1, ux2, uy2 = bars[0] * W, bars[1] * H, W - bars[2] * W, H - bars[3] * H
    uw, uh = ux2 - ux1, uy2 - uy1
    face_h = cand["h"] * H
    frac = (0.62 if spec["close"] else 0.40)
    ch = float(np.clip(face_h / frac, 0.26 * uh, uh))
    if ch * opt.aspect > uw:
        ch = uw / opt.aspect
    # decode just big enough for this crop
    scale = min(1.0, 1.35 * opt.height / ch)
    dw, dh = int(round(W * scale / 2)) * 2, int(round(H * scale / 2)) * 2
    sx, sy = dw / W, dh / H
    margin = 0.3
    start = max(0.0, cand["t0"] - margin)
    frames = _read_frames(scan["path"], start, need + 2 * margin + 0.2, dw, dh)
    if len(frames) < 2:
        return None
    fps = info["fps"]
    # cuts the 5 fps scan could not see
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    jump = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    cuts = np.flatnonzero(jump > max(18.0, 5 * float(np.median(jump)) + 6))
    anchor = int(np.clip(round((cand["t0"] - start) * fps), 0, len(frames) - 1))
    first, last = 0, len(frames) - 1
    for c in cuts:
        if c < anchor:
            first = int(c) + 1
        elif last == len(frames) - 1:
            last = int(c)
    # use the part around the anchor that best fills the shot
    a = anchor
    b = min(last, a + int(round(need * fps)))
    a = max(first, b - int(round(need * fps)))
    frames = frames[a:b + 1]
    times = start + np.arange(a, b + 1) / fps
    if len(frames) < 2:
        return None
    # the face position, smoothed so the camera floats instead of twitching
    cx = np.interp(times, t["t"], (t["boxes"][:, 0] + t["boxes"][:, 2]) / 2) * dw
    cy = np.interp(times, t["t"], (t["boxes"][:, 1] + t["boxes"][:, 3]) / 2) * dh
    k = max(3, int(fps * 0.5)) | 1
    pad = k // 2
    kern = cv2.getGaussianKernel(k, k / 3.5).ravel()
    cx = np.convolve(np.pad(cx, pad, "edge"), kern, "valid")
    cy = np.convolve(np.pad(cy, pad, "edge"), kern, "valid")
    s = Shot()
    s.spec, s.frames, s.fps, s.cand = spec, frames, fps, cand
    s.ch = ch * sy
    s.cw = s.ch * opt.aspect
    # face a little above the middle; the steady camera stays inside the picture
    lx1, lx2 = ux1 * sx + s.cw / 2, ux2 * sx - s.cw / 2
    ly1, ly2 = uy1 * sy + s.ch / 2, uy2 * sy - s.ch / 2
    s.cx = np.clip(cx, lx1, max(lx1, lx2))
    s.cy = np.clip(cy + 0.05 * s.ch, ly1, max(ly1, ly2))
    # slower than the shot when the moment is short, never a frozen picture if avoidable
    avail = (len(frames) - 1) / fps
    s.speed = min(1.0, avail / max(1e-3, need))
    return s


def alone(shot):
    """Only one person in what the GIF really shows? Looks at real frames across the shot, not at the scan."""
    face, person = models.get("face"), models.get("person")
    for i in np.linspace(0, len(shot.frames) - 1, 5).astype(int):
        f = shot.frames[i]
        # a little wider than the shot: the styles' moves show a bit more than the resting crop
        w, h = shot.cw * 1.2 / 2, shot.ch * 1.2 / 2
        x1, y1 = max(0, int(shot.cx[i] - w)), max(0, int(shot.cy[i] - h))
        crop = f[y1:int(shot.cy[i] + h), x1:int(shot.cx[i] + w)]
        ch, cw = crop.shape[:2]
        if len([b for b in face.detect(crop) if b[3] - b[1] >= 0.05 * ch]) > 1:
            return False
        if person is not None and len([b for b in person.detect(crop, 0.5) if (b[2] - b[0]) * (b[3] - b[1]) >= 0.03 * cw * ch]) > 1:
            return False
    return True


def verify(shot, refs):
    """CCIP check on the frame itself: is this really her? (refs = her track features)"""
    ccip = models.get("ccip")
    t = shot.cand["track"]
    i = len(shot.frames) // 2
    f = shot.frames[i]
    h, w = f.shape[:2]
    k = int(np.argmin(np.abs(t["t"] - shot.cand["t0"])))
    b = t["boxes"][k] * np.array([w, h, w, h])
    feat = ccip.embed([crop_pad(f, *head_box(b))])[0]
    d = ccip.distances(np.vstack([refs, feat[None]]))[-1, :-1]
    return float(np.median(d)), ccip.threshold


# ---------------------------------------------------------------------------- drawing
def _matrix(shot, spec, t, dur, opt, W, H, extra=(0.0, 0.0)):
    """Output pixel -> source pixel at time t of the shot, with the style's camera."""
    dx, dy, zoom, rot = spec["style"].cam(spec, t, dur)
    u = min(len(shot.cx) - 1.0, max(0.0, t * shot.speed * shot.fps))
    i = int(u)
    j = min(i + 1, len(shot.cx) - 1)
    a = u - i
    ccx = shot.cx[i] * (1 - a) + shot.cx[j] * a
    ccy = shot.cy[i] * (1 - a) + shot.cy[j] * a
    s = (shot.ch / H) / max(zoom, 1e-3)
    r = np.deg2rad(rot)
    c, n = np.cos(r) * s, np.sin(r) * s
    # the picture moves by (dx, dy), so the camera looks the other way
    tx = ccx - dx * shot.cw
    ty = ccy - dy * shot.cw
    ox, oy = W / 2 + extra[0], H / 2 + extra[1]
    return np.array([[c, -n, tx - c * ox + n * oy], [n, c, ty - n * ox - c * oy]], np.float32), i


def _draw(shot, spec, t, dur, opt, W, H, extra=(0.0, 0.0)):
    m, _ = _matrix(shot, spec, t, dur, opt, W, H, extra)
    k = int(np.clip(round(t * shot.speed * shot.fps), 0, len(shot.frames) - 1))
    out = cv2.warpAffine(shot.frames[k], m, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    return out.astype(np.float32) / 255.0


def _travel(a, b, W, H):
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1], [W / 2, H / 2, 1]], np.float32)
    return float(np.linalg.norm(corners @ a.T - corners @ b.T, axis=1).max())


def _apply_fx(img, fx):
    blur = fx.get("blur", 0.0)
    if blur > 0.35:
        img = cv2.GaussianBlur(img, (0, 0), blur, borderType=cv2.BORDER_REFLECT_101)
    else:
        # the downscale softens line art a little; give it back
        soft = cv2.GaussianBlur(img, (0, 0), 1.1)
        img = np.clip(img + 0.35 * (img - soft), 0, 1)
    split = fx.get("split", 0.0)
    if split > 0.4:
        h, w = img.shape[:2]
        for ch, sx in ((0, split), (2, -split)):
            m = np.array([[1, 0, sx], [0, 1, 0]], np.float32)
            img[:, :, ch] = cv2.warpAffine(img[:, :, ch], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    glow = fx.get("glow", 0.0)
    if glow:
        g = cv2.GaussianBlur(img, (0, 0), img.shape[0] / 28.0)
        img = 1 - (1 - img) * (1 - np.clip(g * glow, 0, 1))
    con = fx.get("contrast", 1.0)
    if con != 1.0:
        img = np.clip((img - 0.5) * con + 0.5, 0, 1)
    sat = fx.get("sat", 1.0)
    if sat != 1.0:
        y = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img = np.clip(y[..., None] + (img - y[..., None]) * sat, 0, 1)
    warm = fx.get("warm", 0.0)
    if warm:
        img = np.clip(img * np.array([1 + warm, 1.0, 1 - warm], np.float32), 0, 1)
    gain = fx.get("gain", 0.0)
    if gain > 0.01:
        img = np.clip(img * 2 ** gain, 0, 1)
    white = fx.get("white", 0.0)
    if white > 0.005:
        img = img * (1 - white) + white
    return img


def render(shots, specs, style, opt, progress=None, stop=None):
    """Returns (frames, shot number per frame)."""
    H = opt.height
    W = int(round(H * opt.aspect / 2)) * 2
    tr = style.trans * opt.pace if style.transition != "cut" else 0.0
    durs = [s["dur"] * opt.pace for s in specs]
    starts = np.cumsum([0.0] + [d - tr for d in durs])
    total = float(starts[-1])
    starts = starts[:-1]
    n_out = int(round(total * opt.fps))
    frames, segs = [], []

    def active(T):
        act = []
        for i, st in enumerate(starts):
            lt = (T - st) % total
            if lt < durs[i]:
                act.append((i, lt))
        # oldest first: when two overlap, the one that just began comes last
        act.sort(key=lambda x: -x[1])
        return act

    def compose(T):
        act = active(T)
        if len(act) == 1 or style.transition == "cut":
            i, lt = act[-1]
            return _draw(shots[i], specs[i], lt, durs[i], opt, W, H)
        (a, ta), (b, tb) = act[0], act[-1]
        u = styles.ease(tb / max(tr, 1e-3))
        if style.transition == "dissolve":
            return _draw(shots[a], specs[a], ta, durs[a], opt, W, H) * (1 - u) + _draw(shots[b], specs[b], tb, durs[b], opt, W, H) * u
        # push: the new shot arrives from the side it travels from
        d = specs[b]["dir"]
        horizontal = abs(d[0]) >= abs(d[1])
        sign = 1 if (d[0] if horizontal else d[1]) >= 0 else -1
        size = W if horizontal else H
        pos = u * size
        ea = (sign * pos, 0.0) if horizontal else (0.0, sign * pos)
        eb = (sign * (pos - size), 0.0) if horizontal else (0.0, sign * (pos - size))
        ia = _draw(shots[a], specs[a], ta, durs[a], opt, W, H, ea)
        ib = _draw(shots[b], specs[b], tb, durs[b], opt, W, H, eb)
        edge = int(round(pos))
        out = ia
        if horizontal:
            if sign > 0:
                out[:, :edge] = ib[:, :edge]
            elif edge:
                out[:, W - edge:] = ib[:, W - edge:]
        else:
            if sign > 0:
                out[:edge] = ib[:edge]
            elif edge:
                out[H - edge:] = ib[H - edge:]
        return out

    for fi in range(n_out):
        if stop is not None and stop.is_set():
            raise InterruptedError
        T = fi / opt.fps
        act = active(T)
        i, lt = act[-1]
        shutter = specs[i]["style"].shutter / opt.fps
        # motion blur: only as many samples as the movement needs
        ma, _ = _matrix(shots[i], specs[i], lt, durs[i], opt, W, H)
        mb, _ = _matrix(shots[i], specs[i], lt + shutter, durs[i], opt, W, H)
        moved = _travel(ma, mb, W, H) * opt.height / max(1.0, shots[i].ch)
        if style.transition == "push" and len(act) > 1:
            moved = max(moved, abs(styles.ease((lt + shutter) / tr) - styles.ease(lt / tr)) * W)
        n = int(np.clip(1 + moved / 2.5, 1, 28))
        if n == 1:
            img = compose(T)
        else:
            img = np.zeros((H, W, 3), np.float32)
            for k in range(n):
                img += compose((T + shutter * k / n) % total)
            img /= n
        scale = opt.height / 480.0
        fx = dict(specs[i]["style"].fx(specs[i], lt, durs[i]))
        for key in ("blur", "split"):
            if key in fx:
                fx[key] *= scale
        img = _apply_fx(img, fx)
        frames.append((img * 255 + 0.5).astype(np.uint8))
        # palette group: a transition belongs to the shot that takes over at its middle
        seg = act[-1][0] if len(act) == 1 or act[-1][1] > tr / 2 else act[0][0]
        segs.append(seg)
        if progress and fi % 5 == 0:
            progress(fi / max(1, n_out), "Rendering frame %d of %d" % (fi, n_out))
    k = min(len(frames) - 1, int(durs[0] * 0.6 * opt.fps))
    return frames[k:] + frames[:k], segs[k:] + segs[:k]


# ---------------------------------------------------------------------------- the whole job
def make_gif(char, members, scans, style, opt, out_path, seed=None, used=None, log=print, progress=None, stop=None):
    """Renders one GIF. members = the tracks to use. Returns a stats dict."""
    t0 = time.time()
    rng = random.Random(seed)
    specs = style.plan(style.count(opt.length / opt.pace), rng)
    members = match.core_members(members)
    want = max(s["dur"] for s in specs) * opt.pace
    company = None
    if opt.solo:
        own = {id(t) for _, t in char["members"]}
        by_scene = {}
        for vi, sc in enumerate(scans):
            for o in sc["tracks"]:
                if id(o) not in own:
                    by_scene.setdefault((vi, o["scene"]), []).append(o)
        company = lambda vi, t: by_scene.get((vi, t["scene"]), [])
    moments = pick_moments(members, len(specs) * (4 if opt.solo else 2), want, rng, used, company)
    if opt.solo:
        # someone else in the wide picture may still be outside the crop: alone() below decides
        have = {moment_id(c) for c in moments}
        moments += [c for c in pick_moments(members, len(specs) * 4, want, rng, used) if moment_id(c) not in have]
    if not moments:
        raise RuntimeError("This character has no usable moments in the video.")
    feats = np.stack([t["feat"] for _, t in members]).astype(np.float32)
    ccip = models.get("ccip")
    dist = ccip.distances(feats[:400]) if len(feats) > 1 else np.zeros((1, 1), np.float32)
    best = np.argsort(dist.mean(axis=1))[:8]
    refs = feats[best]
    shots, rejected, spare, not_alone = [], 0, [], 0
    for c in moments:
        if len(shots) == len(specs):
            break
        spec = specs[len(shots)]
        if progress:
            progress(len(shots) / len(specs) * 0.3, "Fetching moment %d of %d" % (len(shots) + 1, len(specs)))
        if stop is not None and stop.is_set():
            raise InterruptedError
        s = load_shot(c, scans[c["vi"]], spec, opt, spec["dur"] * opt.pace)
        if s is None:
            continue
        if opt.solo and not alone(s):
            not_alone += 1
            continue
        d, thr = verify(s, refs)
        s.same = d
        if d > thr * 0.55:
            # clean moments measure under 0.1; above that she is often hidden, turned away, or someone
            # else fills the picture. Only used if nothing better turns up.
            rejected += 1
            if d <= thr * 1.2:
                spare.append((d, c, s))
            continue
        shots.append(s)
        if used is not None:
            used.add(moment_id(c))
    spare.sort(key=lambda x: x[0])
    for d, c, s in spare:
        if len(shots) < len(specs):
            s.spec = specs[len(shots)]
            shots.append(s)
            if used is not None:
                used.add(moment_id(c))
    if not shots:
        raise RuntimeError("No solo shots of this character were found. Turn Solo off or try another character." if opt.solo
                           else "No moments of this character could be used. Try another character or a longer video.")
    if opt.solo:
        log("Solo: %d shots kept, %d moments left out because someone else was in the picture." % (len(shots), not_alone))
    # few moments: fewer shots rather than the same picture over and over
    if len(shots) < len(specs):
        log("Only %d good moments found, so this GIF is shorter." % len(shots))
        specs = specs[:len(shots)]
        for i, sp in enumerate(specs):
            sp["next"] = specs[(i + 1) % len(specs)]["dir"]
    while len(shots) < len(specs):  # (kept for safety)
        src = shots[len(shots) % max(1, len(shots))]
        again = Shot()
        again.__dict__.update(src.__dict__)
        again.spec = specs[len(shots)]
        shots.append(again)
    pr = (lambda p, m: progress(0.3 + 0.55 * p, m)) if progress else None
    frames, segs = render(shots, specs, style, opt, pr, stop)
    if progress:
        progress(0.86, "Making the GIF")
    work = os.path.join(CACHE, "work", "%d_%d" % (os.getpid(), int(time.time() * 1000)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    size, scale, dither = gifenc.encode(frames, segs, opt.fps, out_path, int(opt.max_mb * 1e6), work, log)
    H = int(round(opt.height * scale / 2)) * 2
    W = int(round(frames[0].shape[1] * scale / 2)) * 2
    return {"path": out_path, "bytes": size, "frames": len(frames), "seconds": len(frames) / opt.fps, "w": W, "h": H,
            "shots": len(shots), "rejected": rejected, "took": time.time() - t0,
            "same": [round(getattr(s, "same", 0.0), 3) for s in shots], "threshold": ccip.threshold,
            "times": [round(s.cand["t0"], 1) for s in shots]}
