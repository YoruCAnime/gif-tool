"""Keeps a job from taking the PC down: watches memory, the app's own size, and the video card.

If something gets tight the job waits at its next check() and carries on once it clears. If it does
not clear, or the card stops answering, the job ends with a message instead of pushing on. All its
work (reading a video, making a GIF) passes through check() every frame.
"""
import ctypes
import gc
import subprocess
import sys
import threading
import time

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
GB = 1024.0 ** 3

RAM_PAUSE = 1.5    # GB free on the PC: wait below this
RAM_STOP = 0.7     # GB free on the PC: end the job at once
APP_MAX = 7.0      # GB this app may hold
HEAT_PAUSE = 83    # C on the card: wait from here (the crash of 2026-11-07 came at 85-87)
HEAT_RESUME = 75   # C: carry on again
VRAM_PAUSE = 0.90  # share of the card's memory
PATIENCE = 300.0   # seconds a job may wait before it gives up
LOOK_EVERY = 3.0   # seconds between looks at the PC
CARD_EVERY = 6.0   # seconds between looks at the card (it starts a small program)
CARD_MISSES = 2    # the card failing to answer this many times in a row counts as lost


class Stopped(RuntimeError):
    """The watchdog ended the job; the text says why, in plain words."""


class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def read_ram():
    """GB free on the PC."""
    m = _Mem()
    m.dwLength = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GB


def read_app():
    """GB this app really holds in memory."""
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    proc = ctypes.windll.kernel32.GetCurrentProcess
    proc.restype = ctypes.c_void_p
    get = ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
    get(proc(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / GB


def read_card():
    """(temperature C, share of its memory in use), or None when nvidia-smi gives no answer."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        temp, used, total = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return temp, used / max(1.0, total)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Guard:
    """Runs while a job runs. check() is called by the job and is the only place it waits or ends."""

    def __init__(self, say=None, ram=read_ram, app=read_app, card=read_card, clock=time.monotonic, look_every=LOOK_EVERY, card_every=CARD_EVERY, patience=PATIENCE):
        self.say, self.ram, self.app, self.card, self.clock = say, ram, app, card, clock
        self.look_every, self.card_every, self.patience = look_every, card_every, patience
        self.waiting = None      # why the job should wait
        self.ended = None        # why the job should end
        self.since = None
        self.hot = False
        self.has_card = False
        self.misses = 0
        self.last_card = None
        self.next_card = 0.0
        self.quit = threading.Event()
        self.thread = None
        self.peak = {"ram_low": 99.0, "app": 0.0, "heat": 0.0, "vram": 0.0, "waits": 0, "waited": 0.0}
        self.last_look = None

    def look(self):
        """One look at the PC; sets waiting / ended."""
        free, mine = self.ram(), self.app()
        now = self.clock()
        if self.waiting is not None and self.last_look is not None:
            self.peak["waited"] += now - self.last_look
        self.last_look = now
        self.peak["ram_low"] = min(self.peak["ram_low"], free)
        self.peak["app"] = max(self.peak["app"], mine)
        if now >= self.next_card:
            self.next_card = now + self.card_every
            reading = self.card()
            if reading is None:
                self.misses += 1
                if self.has_card and self.misses >= CARD_MISSES:
                    self.ended = "The video card stopped responding. Save your work and restart the PC if it stays that way."
                    return
            else:
                self.has_card, self.misses, self.last_card = True, 0, reading
                self.peak["heat"] = max(self.peak["heat"], reading[0])
                self.peak["vram"] = max(self.peak["vram"], reading[1])
        if free < RAM_STOP:
            self.ended = "The PC is almost out of memory (%.1f GB free), so this job stopped. Close something and try again." % free
            return
        if mine > APP_MAX:
            self.ended = "This job grew too big (%.1f GB), so it stopped. Try a shorter GIF or a lower height." % mine
            return
        why = None
        if free < RAM_PAUSE:
            why = "Waiting: only %.1f GB of memory free" % free
        elif mine > APP_MAX * 0.85:
            why = "Waiting: this job is using %.1f GB of memory" % mine
        elif self.last_card is not None:
            temp, vram = self.last_card
            self.hot = temp >= (HEAT_RESUME if self.hot else HEAT_PAUSE)
            if self.hot:
                why = "Cooling down: the video card is at %d C" % temp
            elif vram >= VRAM_PAUSE:
                why = "Waiting: the video card's memory is %d%% full" % (vram * 100)
        if why is None:
            if self.waiting is not None and self.say:
                self.say("Carrying on")
            self.waiting, self.since = None, None
            return
        if self.waiting is None:
            self.since = now
            self.peak["waits"] += 1
            gc.collect()
        self.waiting = why
        if self.say:
            self.say(why)
        if now - self.since > self.patience:
            self.ended = "Waited too long (%s), so this job stopped." % why.split(":", 1)[-1].strip()

    def _run(self):
        while not self.quit.wait(self.look_every):
            try:
                self.look()
            except Exception as e:  # a watchdog must never be what breaks the job
                print("watchdog:", e, file=sys.stderr)
            if self.ended:
                return

    def check(self, stop=None):
        while True:
            if self.ended:
                raise Stopped(self.ended)
            if self.waiting is None:
                return
            if stop is not None and stop.is_set():
                raise InterruptedError
            time.sleep(0.25)

    def __enter__(self):
        global _active
        _active = self
        self.look()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        global _active
        _active = None
        self.quit.set()
        self.thread.join(15)
        return False


_active = None


def check(stop=None):
    """Called by the work every frame. Does nothing when no job is being guarded."""
    guard = _active
    if guard is not None:
        guard.check(stop)
