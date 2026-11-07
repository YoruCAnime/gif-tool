"""Frames -> GIF: one palette per shot, exact delays, fitted under a size limit."""
import os
import shutil
import struct
import subprocess

from .scan import NO_WINDOW, proc_flags

TRAILER = 0x3B


def _skip(b, i):
    while True:
        n = b[i]
        i += 1
        if not n:
            return i
        i += n


def _parse(path):
    with open(path, "rb") as f:
        b = f.read()
    w, h, flags = struct.unpack("<HHB", b[6:11])
    i = 13
    gct = b""
    if flags & 0x80:
        n = 3 * (2 << (flags & 7))
        gct = b[i:i + n]
        i += n
    frames, gce = [], b""
    while i < len(b):
        c = b[i]
        if c == TRAILER:
            break
        if c == 0x21:
            j = _skip(b, i + 2)
            if b[i + 1] == 0xF9:
                gce = b[i:j]
            i = j
        elif c == 0x2C:
            ff = b[i + 9]
            j = i + 10
            lct = b""
            if ff & 0x80:
                n = 3 * (2 << (ff & 7))
                lct = b[j:j + n]
                j += n
            start = j
            j = _skip(b, j + 1)
            desc = bytearray(b[i:i + 10])
            if not lct:
                # the frame moves to its own colour table, so segments can be joined
                desc[9] = (ff | 0x80) & ~0x07 | (flags & 7)
                lct = gct
            frames.append((gce, bytes(desc), lct, b[start:j]))
            gce = b""
            i = j
        else:
            raise ValueError("Unexpected GIF block")
    return w, h, frames


def delays_for(n, fps):
    """GIF delays are in 1/100 s; rounding the running total keeps the exact speed."""
    return [round((i + 1) * 100 / fps) - round(i * 100 / fps) for i in range(n)]


def _join(paths, out, delays):
    w, h, _ = _parse(paths[0])
    k = 0
    with open(out, "wb") as f:
        f.write(b"GIF89a" + struct.pack("<HHBBB", w, h, 0x70, 0, 0))
        f.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")
        for p in paths:
            _, _, frames = _parse(p)
            for gce, desc, lct, data in frames:
                if len(gce) < 8:
                    gce = b"\x21\xF9\x04\x00\x00\x00\x00\x00"
                gce = gce[:4] + struct.pack("<H", max(2, delays[min(k, len(delays) - 1)])) + gce[6:]
                f.write(gce + desc + lct + data)
                k += 1
        f.write(bytes([TRAILER]))
    return k


# measured on anime: no dither is about 20 % smaller than the fine one and stays clean
DITHER = {"fine": "sierra2_4a", "clean": "none"}


def _segment(frames, fps, path, scale, dither):
    h, w = frames[0].shape[:2]
    nw, nh = int(round(w * scale / 2)) * 2, int(round(h * scale / 2)) * 2
    size = "scale=%d:%d:flags=lanczos," % (nw, nh) if scale != 1 else ""
    vf = (size + "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
          "[b][p]paletteuse=dither=%s:diff_mode=rectangle" % DITHER[dither])
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-", "-filter_complex", vf, "-fps_mode", "passthrough", "-threads", "1", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         creationflags=proc_flags())
    try:
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
    except OSError:
        pass
    err = p.stderr.read()
    p.wait()
    if p.returncode:
        raise RuntimeError("GIF encoding failed: " + err.decode(errors="replace")[-300:])


def encode(frames, segs, fps, out, max_bytes, work, log=print):
    """frames: uint8 RGB list, segs: shot number per frame. Returns (bytes, scale, dither)."""
    os.makedirs(work, exist_ok=True)
    groups = []
    for f, s in zip(frames, segs):
        if not groups or groups[-1][0] != s:
            groups.append((s, []))
        groups[-1][1].append(f)
    delays = delays_for(len(frames), fps)

    def attempt(scale, dither):
        paths = []
        for i, (_, fr) in enumerate(groups):
            p = os.path.join(work, "seg%03d.gif" % i)
            _segment(fr, fps, p, scale, dither)
            paths.append(p)
        n = _join(paths, out, delays)
        if n != len(frames):
            raise RuntimeError("GIF has %d frames, expected %d" % (n, len(frames)))
        return os.path.getsize(out)

    scale, dither = 1.0, "fine"
    try:
        size = attempt(scale, dither)
        tries = 0
        while max_bytes and size > max_bytes and tries < 6:
            tries += 1
            over = size / max_bytes
            if dither == "fine":
                dither = "clean"        # costs little picture, saves a lot
            elif scale > 0.55:
                scale = max(0.55, scale * min(0.95, (1 / over) ** 0.55))
            else:
                log("Could not get under the size limit without ruining the picture.")
                break
            log("%.1f MB is over the limit, making it smaller (%d%% size)" % (size / 1e6, scale * 100))
            size = attempt(scale, dither)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return size, scale, dither
