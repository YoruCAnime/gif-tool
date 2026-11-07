"""The edit styles. A style only says how the camera and the picture change over time.

cam(spec, t, dur) -> dx, dy (share of the picture width), zoom, rot (degrees)
fx(spec, t, dur)  -> blur (px at 480p), gain (exposure stops), white, split (px), glow, sat, contrast, warm
t = seconds into the shot, dur = length of the shot. Everything is a smooth function of t, so the
motion stays exact at any frame rate, and the renderer can sample between frames for motion blur.
"""
import math

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7)]


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def decay(t, tau):
    return math.exp(-max(0.0, t) / tau)


class Style:
    name = ""
    blurb = ""
    dur = 0.65          # seconds per shot
    transition = "cut"  # cut | dissolve | push
    trans = 0.0         # seconds the two shots overlap
    close = 0.85        # share of close-up shots (the rest show head and shoulders)
    shutter = 0.5       # part of the frame time the "camera shutter" is open

    def count(self, seconds):
        """How many shots fill that many seconds."""
        return max(3, int(round(seconds / max(0.2, self.dur - self.trans))))

    def plan(self, n, rng):
        out, prev = [], None
        for _ in range(n):
            d = rng.choice([x for x in DIRS if x != prev])
            prev = d
            out.append({"dir": d, "style": self, "side": rng.choice((-1, 1)), "seed": rng.random() * 1000,
                        "close": rng.random() < self.close, "dur": self.dur})
        for i, s in enumerate(out):
            s["next"] = out[(i + 1) % n]["dir"]
        return out

    def cam(self, s, t, dur):
        return 0.0, 0.0, 1.0, 0.0

    def fx(self, s, t, dur):
        return {}


class SlideSettle(Style):
    name = "Slide & settle"
    blurb = "Your GIF: slams in from the side, blurry and bright, then settles sharp"

    def cam(self, s, t, dur):
        off = 0.34 * decay(t, 0.085) - 0.035 * t / dur
        return s["dir"][0] * off, s["dir"][1] * off, 1.0 + 0.03 * decay(t, 0.2), 0.0

    def fx(self, s, t, dur):
        return {"blur": 15 * decay(t, 0.075), "gain": 0.85 * decay(t, 0.09), "white": 0.12 * decay(t, 0.05)}


class ZoomPunch(Style):
    name = "Zoom punch"
    blurb = "Crashes in or out at the face, then a slow push"

    def cam(self, s, t, dur):
        if s["side"] > 0:
            z = 1.0 + 0.42 * decay(t, 0.085) + 0.05 * (1 - t / dur)
        else:
            z = 1.08 - 0.34 * decay(t, 0.085) + 0.06 * t / dur
        return 0.0, 0.0, z, 0.0

    def fx(self, s, t, dur):
        return {"blur": 5 * decay(t, 0.06), "gain": 0.55 * decay(t, 0.08), "contrast": 1.04}


class WhipPan(Style):
    name = "Whip pan"
    blurb = "Fast smeared swipes that carry from one shot into the next"
    dur = 0.8
    shutter = 1.0

    def cam(self, s, t, dur):
        incoming = 0.8 * decay(t, 0.06)
        outgoing = 0.8 * decay(dur - t, 0.06)
        return (s["dir"][0] * incoming - s["next"][0] * outgoing + s["dir"][0] * -0.03 * t / dur,
                s["dir"][1] * incoming - s["next"][1] * outgoing + s["dir"][1] * -0.03 * t / dur, 1.03, 0.0)

    def fx(self, s, t, dur):
        return {"contrast": 1.03}


class DreamyDrift(Style):
    name = "Dreamy drift"
    blurb = "Slow floating moves, soft dissolves and a gentle glow"
    dur = 1.5
    transition = "dissolve"
    trans = 0.4
    close = 0.7

    def cam(self, s, t, dur):
        u = t / dur
        z = 1.02 + 0.09 * u if s["side"] > 0 else 1.11 - 0.09 * u
        return s["dir"][0] * 0.03 * (u - 0.5), s["dir"][1] * 0.03 * (u - 0.5), z, 0.0

    def fx(self, s, t, dur):
        return {"glow": 0.38, "sat": 0.96, "warm": 0.03, "gain": 0.08}


class FlashCuts(Style):
    name = "Flash cuts"
    blurb = "Quick, hard cuts, each with a white flash and a small pop"
    dur = 0.42

    def cam(self, s, t, dur):
        return s["dir"][0] * 0.012 * decay(t, 0.1), s["dir"][1] * 0.012 * decay(t, 0.1), 1.0 + 0.1 * decay(t, 0.07) + 0.02 * t / dur, 0.0

    def fx(self, s, t, dur):
        return {"white": 0.78 * decay(t, 0.03), "gain": 0.35 * decay(t, 0.07), "contrast": 1.06, "sat": 1.05}


class SpinSettle(Style):
    name = "Spin settle"
    blurb = "Twists into place with a zoom and keeps a small tilt"

    def cam(self, s, t, dur):
        sign = s["side"]
        rot = sign * (20 * decay(t, 0.09) + 1.8 * (0.5 - t / dur))
        return 0.0, 0.0, 1.06 + 0.38 * decay(t, 0.09), rot

    def fx(self, s, t, dur):
        return {"blur": 4 * decay(t, 0.06), "gain": 0.5 * decay(t, 0.09)}


class ImpactShake(Style):
    name = "Impact shake"
    blurb = "Hits like a beat drop: camera shake and colour split"

    def cam(self, s, t, dur):
        a = 0.05 * decay(t, 0.14)
        w = 2 * math.pi * 11
        p = s["seed"]
        dx = a * math.sin(w * t + p) * math.cos(0.37 * w * t + p)
        dy = a * math.cos(0.83 * w * t + p * 2)
        return dx, dy, 1.08 + 0.16 * decay(t, 0.06), 1.8 * decay(t, 0.14) * math.sin(w * 0.7 * t + p)

    def fx(self, s, t, dur):
        return {"split": 11 * decay(t, 0.11), "white": 0.45 * decay(t, 0.035), "gain": 0.4 * decay(t, 0.08), "contrast": 1.06}


class ElasticBounce(Style):
    name = "Elastic bounce"
    blurb = "Springs in and wobbles to a stop, playful"

    def cam(self, s, t, dur):
        k = decay(t, 0.17) * math.cos(2 * math.pi * t / 0.3)
        return s["dir"][0] * 0.045 * k, s["dir"][1] * 0.045 * k, 1.07 + 0.27 * k, 0.0

    def fx(self, s, t, dur):
        return {"blur": 3 * decay(t, 0.05), "gain": 0.45 * decay(t, 0.07)}


class StepZoom(Style):
    name = "Step zoom"
    blurb = "Three punch-ins on the same shot, closer each time"
    dur = 1.05
    close = 0.1

    def _step(self, t, dur):
        n = min(2, int(t / (dur / 3)))
        return n, t - n * dur / 3

    def cam(self, s, t, dur):
        n, dt = self._step(t, dur)
        z = (1.0, 1.2, 1.44)[n]
        return 0.0, -0.012 * n, z * (1.0 + 0.09 * decay(dt, 0.05) + 0.02 * dt / (dur / 3)), 0.0

    def fx(self, s, t, dur):
        n, dt = self._step(t, dur)
        return {"blur": 3 * decay(dt, 0.035), "white": 0.35 * decay(dt, 0.03), "contrast": 1.04}


class PushSlide(Style):
    name = "Push slide"
    blurb = "Each shot pushes the last one off screen, smooth and clean"
    dur = 1.0
    transition = "push"
    trans = 0.3
    shutter = 0.75
    close = 0.8

    def cam(self, s, t, dur):
        return s["dir"][0] * 0.03 * (0.5 - t / dur), s["dir"][1] * 0.03 * (0.5 - t / dur), 1.03, 0.0

    def fx(self, s, t, dur):
        return {"contrast": 1.02}


class Mix(Style):
    name = "Mix"
    blurb = "A different move on every shot, so it never repeats"

    def __init__(self):
        self.pool = [SlideSettle(), ZoomPunch(), SpinSettle(), ImpactShake(), ElasticBounce(), FlashCuts()]

    def count(self, seconds):
        return max(3, int(round(seconds / 0.58)))

    def plan(self, n, rng):
        pool = list(self.pool)
        rng.shuffle(pool)
        out, prev = [], None
        for i in range(n):
            sub = pool[i % len(pool)]
            d = rng.choice([x for x in DIRS if x != prev])
            prev = d
            out.append({"dir": d, "style": sub, "side": rng.choice((-1, 1)), "seed": rng.random() * 1000,
                        "close": rng.random() < 0.85, "dur": max(0.5, sub.dur)})
        for i, s in enumerate(out):
            s["next"] = out[(i + 1) % n]["dir"]
        return out


ALL = [SlideSettle(), ZoomPunch(), WhipPan(), DreamyDrift(), FlashCuts(), SpinSettle(), ImpactShake(),
       ElasticBounce(), StepZoom(), PushSlide()]
MIX = Mix()


def by_name(name):
    for s in ALL + [MIX]:
        if s.name == name:
            return s
    raise KeyError(name)
