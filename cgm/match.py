"""Groups the face tracks into characters and ranks them against the typed description."""
import re

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import models
from .scan import unjpg

# CIELAB-style anchors in OpenCV-free units (L 0-100, a, b)
COLOURS = {
    "black": (14, 0, 0), "white": (94, 0, 2), "grey": (62, 0, -2), "gray": (62, 0, -2), "silver": (78, 0, -3),
    "brown": (34, 20, 26), "chestnut": (32, 25, 25), "auburn": (35, 35, 30), "brunette": (32, 16, 20),
    "blonde": (82, 3, 42), "blond": (82, 3, 42), "golden": (76, 8, 58), "gold": (76, 8, 58), "yellow": (88, -6, 68),
    "orange": (66, 38, 60), "ginger": (55, 38, 48), "red": (42, 58, 38), "crimson": (35, 55, 25), "scarlet": (45, 62, 45),
    "pink": (76, 35, 2), "magenta": (55, 70, -15), "purple": (42, 42, -42), "violet": (50, 42, -50), "lavender": (72, 18, -22),
    "blue": (46, 14, -52), "navy": (22, 12, -32), "cyan": (78, -28, -18), "aqua": (78, -30, -10), "teal": (50, -28, -8),
    "turquoise": (72, -35, -5), "green": (52, -44, 34), "emerald": (48, -48, 18), "mint": (84, -28, 12), "amber": (66, 20, 62),
    "hazel": (50, 8, 34), "light blue": (78, -20, -24), "dark blue": (24, 12, -36), "dark brown": (20, 12, 14),
    "light brown": (55, 15, 30), "dark green": (28, -24, 14), "light green": (80, -30, 30), "dark red": (26, 36, 20),
    "platinum": (88, 0, 12), "dark": (16, 2, 2),
}
OUTFIT_HINTS = ("wear", "dress", "shirt", "uniform", "skirt", "jacket", "coat", "kimono", "armor", "armour", "suit", "hoodie",
                "sweater", "blouse", "robe", "cloak", "cape", "swimsuit", "bikini", "apron", "hat", "glasses", "ribbon", "choker",
                "clothes", "outfit", "tie", "scarf", "gloves", "boots", "pants", "shorts", "sailor", "maid", "tshirt", "t-shirt")


FEMALE = r"\b(girl|woman|female|lady|she|her|heroine|princess|queen|maid|waitress)\b"
MALE = r"\b(boy|man|male|guy|he|him|hero|king|prince|dude|gentleman)\b"


def split_description(text):
    """'brown hair, red eye girl, wearing a white dress' -> list of {'text', 'kind', 'colour'}."""
    out = []
    low = (text or "").lower()
    if re.search(FEMALE, low):
        out.append({"text": "girl", "kind": "gender", "colour": None})
    elif re.search(MALE, low):
        out.append({"text": "man", "kind": "gender", "colour": None})
    t = re.sub(r"\b(eyed|eyes)\b", "eye", text.lower())
    t = re.sub(r"\bhaired\b", "hair", t)
    t = re.sub(r"\band\b|\bwith\b|;|/|\n", ",", t)
    names = sorted(COLOURS, key=len, reverse=True)
    for part in [x.strip(" .") for x in t.split(",")]:
        if not part:
            continue
        found = False
        # 'red eye girl' or 'brown hair' or 'hair brown'
        for word, kind in (("hair", "hair"), ("eye", "eyes")):
            if word in part:
                for name in names:
                    if re.search(r"\b%s\b" % re.escape(name), part):
                        out.append({"text": "%s %s" % (name, "hair" if kind == "hair" else "eyes"), "kind": kind, "colour": name})
                        found = True
                        break
        # 'girl' left over from 'red eye girl' adds nothing useful for ranking
        if found and not any(h in part for h in OUTFIT_HINTS):
            continue
        text_part = re.sub(r"^(a|an|the|she is|wearing|in)\s+", "", part)
        text_part = re.sub(FEMALE + "|" + MALE, "", text_part).strip()
        if not text_part:
            continue
        kind = "outfit" if any(h in part for h in OUTFIT_HINTS) else "other"
        out.append({"text": text_part, "kind": kind, "colour": None})
    return out


def _lab(rgb):
    """float Lab in the usual units."""
    x = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return x


def measure_colours(head_rgb):
    """Hair and eye colour of a head crop. The eye colour is None when it cannot be told apart."""
    lab = _lab(head_rgb)
    h, w = lab.shape[:2]
    skin = np.median(lab[int(h * .55):int(h * .72), int(w * .4):int(w * .6)].reshape(-1, 3), axis=0)
    # hair: the band over the forehead, the part of the crop that is nearly always hair
    px = lab[int(h * .07):int(h * .29), int(w * .28):int(w * .72)].reshape(-1, 3)
    px = px[px[:, 0] > 0.5]  # no black padding
    keep = px[np.linalg.norm(px - skin, axis=1) > 16]
    hair = np.median(px, axis=0) if len(px) else skin
    if len(keep) > 40:
        _, labels, cen = cv2.kmeans(keep.astype(np.float32), 2, None, (cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        hair = cen[np.bincount(labels.ravel(), minlength=2).argmax()]
    # eyes: coloured, not white, not skin, not the hair that falls across the eye row
    ey = lab[int(h * .38):int(h * .62), int(w * .26):int(w * .74)].reshape(-1, 3)
    chroma = np.hypot(ey[:, 1], ey[:, 2])
    good = (np.linalg.norm(ey[:, 1:] - skin[1:], axis=1) > 14) & (np.linalg.norm(ey[:, 1:] - hair[1:], axis=1) > 12) & (chroma > 12) & (ey[:, 0] < 88)
    eye = None
    if good.sum() >= 14:
        sel = ey[good]
        c2 = chroma[good]
        eye = np.median(sel[c2 >= np.percentile(c2, 60)], axis=0)
    return hair, eye


def colour_score(lab, name):
    """How well a measured colour fits a colour word, 0-1. Hue decides; shade counts less."""
    if lab is None:
        return 0.5
    la, aa, ba = COLOURS[name]
    l, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    c, ca = float(np.hypot(a, b)), float(np.hypot(aa, ba))
    if ca < 8:  # black, white, grey, silver
        return float(np.exp(-(c / 13.0) ** 2) * np.exp(-((l - la) / 24.0) ** 2))
    dh = abs((np.degrees(np.arctan2(b, a)) - np.degrees(np.arctan2(ba, aa)) + 180) % 360 - 180)
    # 'light blue' and 'dark blue' are about the shade; plain 'blue' is about the hue
    shade = 24.0 if name.startswith(("light ", "dark ")) else 42.0
    return float(np.exp(-(dh / 38.0) ** 2) * min(1.0, c / (0.3 * ca)) * np.exp(-((l - la) / shade) ** 2))


def group_characters(scans, strict=0.85):
    """Tracks of all videos -> characters, biggest first.

    Average linkage on CCIP differences: one odd track cannot chain two characters together.
    """
    ccip = models.get("ccip")
    tracks = [(vi, t) for vi, s in enumerate(scans) for t in s["tracks"]]
    if not tracks:
        return []
    feats = np.stack([t["feat"] for _, t in tracks]).astype(np.float32)
    if len(tracks) == 1:
        labels = np.array([1])
    else:
        dist = np.zeros((len(tracks), len(tracks)), np.float32)
        step = 900
        if len(tracks) <= step:
            dist = ccip.distances(feats)
        else:
            # a very long video: compare against a sampled backbone in chunks
            for i in range(0, len(tracks), step):
                for j in range(i, len(tracks), step):
                    a, b = np.arange(i, min(i + step, len(tracks))), np.arange(j, min(j + step, len(tracks)))
                    idx = np.concatenate([a, b]) if i != j else a
                    d = ccip.distances(feats[idx])
                    if i == j:
                        dist[np.ix_(a, a)] = d
                    else:
                        dist[np.ix_(a, b)] = d[:len(a), len(a):]
                        dist[np.ix_(b, a)] = d[len(a):, :len(a)]
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        z = linkage(squareform(np.clip(dist, 0, None), checks=False), "average")
        labels = fcluster(z, ccip.threshold * strict, "distance")
    chars = []
    for lab in np.unique(labels):
        members = np.flatnonzero(labels == lab)
        secs = sum(float(tracks[i][1]["t"][-1] - tracks[i][1]["t"][0]) + 0.2 for i in members)
        chars.append({"members": [tracks[i] for i in members], "seconds": secs})
    chars.sort(key=lambda c: -c["seconds"])
    return [c for c in chars if len(c["members"]) >= 2 or c["seconds"] >= 2.0]


def _portraits(char, n=8):
    reps = []
    for _, t in char["members"]:
        for r in t["reps"]:
            reps.append((r["h"], r))
    reps.sort(key=lambda x: -x[0])
    step = max(1, len(reps) // (n * 4))
    picked = [r for _, r in reps[::step][:n * 3]]

    # portraits with little black padding first (faces at the edge of the frame look odd)
    def black(r):
        return float((unjpg(r["head"]).max(axis=2) < 6).mean())
    picked.sort(key=lambda r: black(r) > 0.04)
    return picked[:n]


def describe_characters(chars):
    """Fills in each character's measured colours and CLIP picture feature (only once)."""
    clip = models.get("clip")
    for c in chars:
        if "hair" in c:
            continue
        reps = _portraits(c)
        cols = [measure_colours(unjpg(r["head"])) for r in reps]
        c["hair"] = np.median([x[0] for x in cols], axis=0)
        eyes = [x[1] for x in cols if x[1] is not None]
        c["eye"] = np.median(eyes, axis=0) if len(eyes) >= 2 else None
        c["thumb"] = unjpg(reps[0]["head"])
        c["thumbs"] = [unjpg(r["head"]) for r in reps[:4]]
        if clip is not None:
            feats = clip.images([unjpg(r["bust"]) for r in reps])
            c["clip"] = feats.mean(axis=0)
            c["clip"] /= np.linalg.norm(c["clip"]) + 1e-6


SIMPLE = {"black": "black", "dark": "black", "white": "white", "grey": "grey", "gray": "grey", "silver": "silver", "brown": "brown",
          "chestnut": "brown", "auburn": "auburn", "brunette": "brown", "blonde": "blonde", "blond": "blonde", "golden": "blonde",
          "gold": "blonde", "yellow": "yellow", "orange": "orange", "ginger": "orange", "red": "red", "crimson": "red", "scarlet": "red",
          "pink": "pink", "magenta": "pink", "purple": "purple", "violet": "purple", "lavender": "lavender", "blue": "blue", "navy": "dark blue",
          "cyan": "light blue", "aqua": "light blue", "teal": "teal", "turquoise": "teal", "green": "green", "emerald": "green", "mint": "green",
          "amber": "amber", "hazel": "hazel", "light blue": "light blue", "dark blue": "dark blue", "dark brown": "brown", "light brown": "brown",
          "dark green": "dark green", "light green": "green", "dark red": "red", "platinum": "blonde"}


def colour_name(lab):
    return SIMPLE[max(COLOURS, key=lambda n: colour_score(lab, n))]


def looks_like(c):
    """What was actually seen, in words, for the character list."""
    return "%s hair" % colour_name(c["hair"])


def _z(v):
    v = np.asarray(v, np.float32)
    return (v - v.mean()) / (v.std() + 1e-4)


def rank_characters(chars, description):
    """Sorts chars best match first; each gets c['match'] (0-100) and c['why']."""
    describe_characters(chars)
    phrases = split_description(description or "")
    if not chars:
        return chars
    presence = _z(np.log1p([c["seconds"] for c in chars]))
    if not phrases:
        for c, p in zip(chars, presence):
            c["score"], c["why"] = float(p), "on screen %.0f s" % c["seconds"]
        chars.sort(key=lambda c: -c["score"])
        top = max(1e-6, chars[0]["seconds"])
        for c in chars:
            c["match"] = int(round(100 * c["seconds"] / top))
            c["why"] = looks_like(c)
        return chars
    clip = models.get("clip")
    total = 0.3 * presence - np.array([1.5 if c["seconds"] < 3 else (0.5 if c["seconds"] < 12 else 0.0) for c in chars], np.float32)
    raw_col = np.zeros((len(chars), max(1, len(phrases))), np.float32)
    n_col = 0
    for pi, ph in enumerate(phrases):
        parts = []
        if ph["colour"]:
            v = [colour_score(c["hair"] if ph["kind"] == "hair" else c["eye"], ph["colour"]) for c in chars]
            raw_col[:, pi] = v
            n_col += 1
            parts.append((1.3 if ph["kind"] == "hair" else 0.9) * _z(v))
        if clip is not None and "clip" in chars[0] and ph["kind"] != "eyes_only":
            if ph["kind"] == "gender":
                # both sides of the question, and it counts as much as all the rest together
                other = "man" if ph["text"] == "girl" else "girl"
                txt = clip.texts(["portrait of an anime " + ph["text"], "portrait of an anime " + other])
                v = [float(c["clip"] @ txt[0]) - float(c["clip"] @ txt[1]) for c in chars]
                parts.append(1.6 * len(phrases) / max(1, len(phrases) - 1 or 1) * _z(v))
            else:
                txt = clip.texts(["anime character with " + ph["text"], "anime character"])
                v = [float(c["clip"] @ txt[0]) - float(c["clip"] @ txt[1]) for c in chars]
                parts.append((0.7 if ph["colour"] else 1.0) * _z(v))
        for p in parts:
            total = total + p / len(phrases)
    for c, s in zip(chars, total):
        c["score"] = float(s)
    order = np.argsort(-total)
    conf = np.exp(total - total.max())
    for i, c in enumerate(chars):
        c["match"] = int(round(100 * conf[i]))
        c["why"] = looks_like(c)
    chars[:] = [chars[i] for i in order]
    return chars


def core_members(members):
    """Only the tracks that clearly match the middle of the group, so a stray look-alike is left out."""
    if len(members) < 6:
        return members
    ccip = models.get("ccip")
    step = max(1, len(members) // 400)
    feats = np.stack([t["feat"] for _, t in members[::step]]).astype(np.float32)
    dist = ccip.distances(feats)
    refs = feats[np.argsort(dist.mean(axis=1))[:8]]
    keep = []
    for i in range(0, len(members), 200):
        part = members[i:i + 200]
        f = np.stack([t["feat"] for _, t in part]).astype(np.float32)
        d = ccip.distances(np.vstack([refs, f]))[len(refs):, :len(refs)]
        keep += [m for m, v in zip(part, np.median(d, axis=1)) if v < ccip.threshold]
    return keep if len(keep) >= max(4, len(members) // 3) else members


def outfit_filter(char, description):
    """Only the tracks where she wears what the description says, if enough of them exist."""
    phrases = [p for p in split_description(description or "") if p["kind"] == "outfit"]
    clip = models.get("clip")
    members = char["members"]
    if not phrases or clip is None or len(members) < 16:
        return members
    feats = clip.images([unjpg(t["reps"][-1]["bust"]) for _, t in members])
    txt = clip.texts(["anime character wearing " + p["text"] for p in phrases] + ["anime character"])
    score = (feats @ txt[:-1].T).mean(axis=1) - feats @ txt[-1]
    keep = np.argsort(-score)[:max(10, len(members) // 3)]
    return [members[i] for i in keep]
