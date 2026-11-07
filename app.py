"""Character GIF Maker: pick a video, describe a character, get a smooth montage GIF of her."""
import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog

ROOT = os.path.dirname(os.path.abspath(__file__))
if sys.stdout is None or sys.stderr is None:  # pythonw has no console
    sys.stdout = sys.stderr = open(os.path.join(ROOT, "app.log"), "a", encoding="utf-8", buffering=1)

import customtkinter as ctk
from PIL import Image, ImageSequence, ImageTk

SETTINGS = os.environ.get("CGM_SETTINGS") or os.path.join(ROOT, "settings.json")
DEFAULTS = {"videos": [], "description": "", "style": "Slide & settle", "length": 6, "pace": "Normal", "shape": "Portrait",
            "fps": "25", "height": "480", "max_mb": "15", "out": os.path.join(ROOT, "output")}
PACES = {"Fast": 0.75, "Normal": 1.0, "Slow": 1.35}
SHAPES = {"Portrait": 0.9, "Square": 1.0, "Phone": 0.5625, "Wide": 1.7778}
ACCENT = "#d0578f"
ACCENT_HOVER = "#b4447a"
CARD = "#2b2b31"
CARD_ON = "#5a2f46"
ALL_STYLES = "All 10 styles"
PLACEHOLDER = "brown hair, red eyes, girl, wearing a white dress"


def safe_name(text, limit=46):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:limit].strip(" .") or "video"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Character GIF Maker")
        self.geometry("1320x800")
        self.minsize(1120, 680)
        ctk.set_appearance_mode("dark")
        self.cfg = dict(DEFAULTS)
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                self.cfg.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = False
        self.scans = None
        self.scan_key = None
        self.chars = []
        self.ranked_for = None
        self.selected = None
        self.results = []
        self.play_job = None
        self.play_frames = []
        self.cards = []
        self.grid_columnconfigure(0, weight=0, minsize=350)
        self.grid_columnconfigure(1, weight=1, minsize=330)
        self.grid_columnconfigure(2, weight=2, minsize=430)
        self.grid_rowconfigure(0, weight=1)
        self._left()
        self._middle()
        self._right()
        self._bottom()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(60, self._pump)

    # ------------------------------------------------------------------ layout
    def _left(self):
        f = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color="#222227")
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)
        self.left = f
        r = 0

        def head(text):
            nonlocal r
            ctk.CTkLabel(f, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(row=r, column=0, sticky="ew", padx=12, pady=(12, 3))
            r += 1

        head("1. Video")
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.grid(row=r, column=0, sticky="ew", padx=12)
        row.grid_columnconfigure(0, weight=1)
        r += 1
        self.video_var = tk.StringVar(value="; ".join(self.cfg["videos"]))
        ctk.CTkEntry(row, textvariable=self.video_var, placeholder_text="Choose one or more episodes").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(row, text="Browse", width=72, fg_color="#44444c", hover_color="#55555e", command=self.browse).grid(row=0, column=1)

        head("2. Who should it be?")
        self.desc = ctk.CTkTextbox(f, height=84, wrap="word")
        self.desc.grid(row=r, column=0, sticky="ew", padx=12)
        r += 1
        self.desc.insert("1.0", self.cfg["description"])
        ctk.CTkLabel(f, text="For example: " + PLACEHOLDER, text_color="#9a9aa2", font=ctk.CTkFont(size=11), anchor="w", justify="left", wraplength=310).grid(row=r, column=0, sticky="ew", padx=12)
        r += 1
        self.find_btn = ctk.CTkButton(f, text="Find her", fg_color="#44444c", hover_color="#55555e", command=self.on_find)
        self.find_btn.grid(row=r, column=0, sticky="ew", padx=12, pady=(6, 0))
        r += 1

        head("3. Style")
        from cgm import styles
        self.styles = styles
        names = [s.name for s in styles.ALL] + [styles.MIX.name, ALL_STYLES]
        self.style_var = tk.StringVar(value=self.cfg["style"] if self.cfg["style"] in names else names[0])
        ctk.CTkOptionMenu(f, variable=self.style_var, values=names, command=lambda _=None: self._blurb(), fg_color="#44444c", button_color="#55555e", button_hover_color="#66666f").grid(row=r, column=0, sticky="ew", padx=12)
        r += 1
        self.blurb = ctk.CTkLabel(f, text="", text_color="#b8b8c0", font=ctk.CTkFont(size=12), anchor="w", justify="left", wraplength=310)
        self.blurb.grid(row=r, column=0, sticky="ew", padx=12, pady=(3, 0))
        r += 1
        self._blurb()

        head("Settings")
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.grid(row=r, column=0, sticky="ew", padx=12)
        grid.grid_columnconfigure(1, weight=1)
        r += 1
        self.length_var = tk.IntVar(value=int(self.cfg["length"]))
        self.pace_var = tk.StringVar(value=self.cfg["pace"])
        self.shape_var = tk.StringVar(value=self.cfg["shape"])
        self.fps_var = tk.StringVar(value=str(self.cfg["fps"]))
        self.height_var = tk.StringVar(value=str(self.cfg["height"]))
        self.mb_var = tk.StringVar(value=str(self.cfg["max_mb"]))
        for i, (label, var, values) in enumerate([
                ("Length (s)", None, None),
                ("Pace", self.pace_var, list(PACES)),
                ("Shape", self.shape_var, list(SHAPES)),
                ("Smoothness", self.fps_var, ["15", "20", "25", "33", "50"]),
                ("Height (px)", self.height_var, ["360", "480", "600", "720"]),
                ("Max size (MB)", self.mb_var, ["5", "10", "15", "25", "50"])]):
            ctk.CTkLabel(grid, text=label, anchor="w").grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
            if var is None:
                box = ctk.CTkFrame(grid, fg_color="transparent")
                box.grid(row=i, column=1, sticky="ew")
                box.grid_columnconfigure(0, weight=1)
                self.length_label = ctk.CTkLabel(box, text=str(self.length_var.get()), width=22)
                self.length_label.grid(row=0, column=1)
                ctk.CTkSlider(box, from_=3, to=15, number_of_steps=12, variable=self.length_var, command=lambda v: self.length_label.configure(text=str(int(v))), button_color=ACCENT, button_hover_color=ACCENT_HOVER).grid(row=0, column=0, sticky="ew")
            else:
                ctk.CTkOptionMenu(grid, variable=var, values=values, height=26, fg_color="#44444c", button_color="#55555e", button_hover_color="#66666f").grid(row=i, column=1, sticky="ew", pady=2)

        head("Save GIFs in")
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.grid(row=r, column=0, sticky="ew", padx=12)
        row.grid_columnconfigure(0, weight=1)
        r += 1
        self.out_var = tk.StringVar(value=self.cfg["out"])
        ctk.CTkEntry(row, textvariable=self.out_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(row, text="...", width=34, fg_color="#44444c", hover_color="#55555e", command=self.browse_out).grid(row=0, column=1)
        self.make_btn = ctk.CTkButton(f, text="Make GIF", height=42, font=ctk.CTkFont(size=15, weight="bold"), fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.on_make)
        self.make_btn.grid(row=r, column=0, sticky="ew", padx=12, pady=(14, 12))

    def _middle(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=(8, 4), pady=8)
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f, text="Who I found (click to choose)", font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(row=0, column=0, sticky="ew", pady=(2, 4))
        self.gallery = ctk.CTkScrollableFrame(f, fg_color="#1d1d21")
        self.gallery.grid(row=1, column=0, sticky="nsew")
        self.gallery.grid_columnconfigure(0, weight=1)
        self.empty = ctk.CTkLabel(self.gallery, text="Choose a video and press Find her,\nor just press Make GIF.", text_color="#9a9aa2")
        self.empty.grid(row=0, column=0, pady=40)

    def _right(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(f, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(2, 4))
        top.grid_columnconfigure(0, weight=1)
        self.preview_title = ctk.CTkLabel(top, text="Preview", font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        self.preview_title.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(top, text="Open folder", width=90, height=26, fg_color="#44444c", hover_color="#55555e", command=self.open_folder).grid(row=0, column=1)
        self.stage = tk.Label(f, bg="#141417", bd=0)
        self.stage.grid(row=1, column=0, sticky="nsew")
        self.stage.bind("<Configure>", lambda e: self._refit())
        self.result_box = ctk.CTkScrollableFrame(f, height=112, fg_color="#1d1d21", label_text="")
        self.result_box.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.result_box.grid_columnconfigure(0, weight=1)

    def _bottom(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color="#18181b", height=40)
        f.grid(row=1, column=0, columnspan=3, sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        self.status = ctk.CTkLabel(f, text="Ready", anchor="w", width=620)
        self.status.grid(row=0, column=0, sticky="w", padx=12, pady=7)
        self.bar = ctk.CTkProgressBar(f, progress_color=ACCENT)
        self.bar.set(0)
        self.bar.grid(row=0, column=1, sticky="ew", padx=8)
        self.stop_btn = ctk.CTkButton(f, text="Stop", width=70, height=26, state="disabled", fg_color="#44444c", hover_color="#55555e", command=self.stop_flag.set)
        self.stop_btn.grid(row=0, column=2, padx=10)

    # ------------------------------------------------------------------ small helpers
    def _blurb(self):
        name = self.style_var.get()
        if name == ALL_STYLES:
            text = "One GIF in each of the 10 styles, each with different moments"
        else:
            text = self.styles.by_name(name).blurb
        self.blurb.configure(text=text)

    def browse(self):
        paths = filedialog.askopenfilenames(parent=self, title="Choose video(s)", filetypes=[("Videos", "*.mkv *.mp4 *.webm *.mov *.avi *.m4v *.ts"), ("All files", "*.*")])
        if paths:
            self.video_var.set("; ".join(os.path.normpath(p) for p in paths))

    def browse_out(self):
        p = filedialog.askdirectory(parent=self, title="Save GIFs in")
        if p:
            self.out_var.set(os.path.normpath(p))

    def open_folder(self):
        p = self.out_var.get()
        if os.path.isdir(p):
            os.startfile(p)
        else:
            self.say("That folder does not exist yet: it appears after the first GIF.")

    def videos(self):
        return [p.strip().strip('"') for p in self.video_var.get().split(";") if p.strip()]

    def description(self):
        return self.desc.get("1.0", "end").strip()

    def save_settings(self):
        self.cfg.update(videos=self.videos(), description=self.description(), style=self.style_var.get(), length=int(self.length_var.get()),
                        pace=self.pace_var.get(), shape=self.shape_var.get(), fps=self.fps_var.get(), height=self.height_var.get(),
                        max_mb=self.mb_var.get(), out=self.out_var.get())
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=1, ensure_ascii=False)
        except OSError:
            pass

    def say(self, text, progress=None):
        self.status.configure(text=text[:96])
        if progress is not None:
            self.bar.set(progress)

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.make_btn.configure(state=state)
        self.find_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if busy else "disabled")

    def on_close(self):
        self.stop_flag.set()
        self.save_settings()
        self.destroy()

    # ------------------------------------------------------------------ background work
    def _pump(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "say":
                    self.say(*data)
                elif kind == "chars":
                    self.show_chars()
                elif kind == "result":
                    self.add_result(data)
                elif kind == "done":
                    self.set_busy(False)
                    self.say(data, 1.0 if not data.startswith("Stopped") else 0)
        except queue.Empty:
            pass
        self.after(60, self._pump)

    def _start(self, target):
        if self.busy:
            return
        if not self.videos():
            self.say("Choose a video first.")
            return
        for p in self.videos():
            if not os.path.isfile(p):
                self.say("This file was not found: " + p)
                return
        self.save_settings()
        self.stop_flag.clear()
        self.job_videos = self.videos()
        self.job_desc = self.description()
        self.set_busy(True)

        def run():
            try:
                self.q.put(("done", target()))
            except InterruptedError:
                self.q.put(("done", "Stopped."))
            except Exception as e:
                traceback.print_exc()
                self.q.put(("done", "Problem: %s" % str(e).replace("\n", " ")))
        threading.Thread(target=run, daemon=True).start()

    def _find(self):
        """Scan (or reuse), group, rank. Runs in the background thread."""
        from cgm import match, scan
        videos = self.job_videos
        key = tuple(videos)
        if key != self.scan_key:
            scans = []
            for i, path in enumerate(videos):
                name = os.path.basename(path)

                def prog(p, msg, i=i, name=name):
                    self.q.put(("say", ("%s  %s" % (msg, name[:30]), (i + p) / len(videos) * 0.9)))
                self.q.put(("say", ("Loading the models", 0.0)))
                scans.append(scan.scan_video(path, prog, self.stop_flag))
            self.q.put(("say", ("Grouping the characters", 0.92)))
            self.chars = match.group_characters(scans)
            self.scans, self.scan_key, self.ranked_for = scans, key, None
            self.selected = None
        desc = self.job_desc
        if desc != self.ranked_for or self.selected is None:
            self.q.put(("say", ("Comparing with your description", 0.95)))
            self.chars = match.rank_characters(self.chars, desc)
            self.ranked_for = desc
            self.selected = self.chars[0] if self.chars else None
            self.q.put(("chars", None))
        if not self.chars:
            raise RuntimeError("No anime faces were found in this video.")

    def on_find(self):
        def job():
            self._find()
            return "Found %d characters. The best match is highlighted; click another if it is wrong." % len(self.chars)
        self._start(job)

    def on_make(self):
        name = self.style_var.get()
        opt_values = dict(length=float(self.length_var.get()), pace=PACES[self.pace_var.get()], aspect=SHAPES[self.shape_var.get()],
                          fps=int(self.fps_var.get()), height=int(self.height_var.get()))
        try:
            opt_values["max_mb"] = float(self.mb_var.get())
        except ValueError:
            opt_values["max_mb"] = 15.0
        outdir = self.out_var.get()

        def job():
            from cgm import match, render
            self._find()
            char = self.selected
            members = match.outfit_filter(char, self.job_desc)
            opt = render.Options()
            for k, v in opt_values.items():
                setattr(opt, k, v)
            todo = [s for s in self.styles.ALL] if name == ALL_STYLES else [self.styles.by_name(name)]
            stem = safe_name(os.path.splitext(os.path.basename(self.job_videos[0]))[0])
            used = set()
            last = None
            for i, style in enumerate(todo):
                if self.stop_flag.is_set():
                    raise InterruptedError
                out = self._free_name(outdir, "%s - %s" % (stem, style.name.replace("&", "and")))

                def prog(p, msg, i=i, style=style):
                    self.q.put(("say", ("%s: %s" % (style.name, msg), (i + p) / len(todo))))
                last = render.make_gif(char, members, self.scans, style, opt, out, used=used, log=lambda m: self.q.put(("say", (m, None))), progress=prog, stop=self.stop_flag)
                last["style"] = style.name
                self.q.put(("result", last))
            return "Done: %d GIF%s in %s" % (len(todo), "" if len(todo) == 1 else "s", outdir)
        self._start(job)

    @staticmethod
    def _free_name(folder, stem):
        i = 1
        while True:
            p = os.path.join(folder, "%s %02d.gif" % (stem, i))
            if not os.path.exists(p):
                return p
            i += 1

    # ------------------------------------------------------------------ gallery
    def show_chars(self):
        for w in self.gallery.winfo_children():
            w.destroy()
        self.cards = []
        for i, c in enumerate(self.chars[:24]):
            on = c is self.selected
            card = ctk.CTkFrame(self.gallery, fg_color=CARD_ON if on else CARD, corner_radius=8)
            card.grid(row=i, column=0, sticky="ew", padx=4, pady=3)
            card.grid_columnconfigure(1, weight=1)
            pics = ctk.CTkFrame(card, fg_color="transparent")
            pics.grid(row=0, column=0, rowspan=2, padx=6, pady=6)
            clicks = [card, pics]
            for k, th in enumerate(c["thumbs"][:3]):
                img = ctk.CTkImage(Image.fromarray(th), size=(66, 66))
                lab = ctk.CTkLabel(pics, image=img, text="")
                lab.grid(row=0, column=k, padx=1)
                clicks.append(lab)
            top = ctk.CTkLabel(card, text="Match %d%%" % c.get("match", 0), font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
            top.grid(row=0, column=1, sticky="ew", padx=4, pady=(6, 0))
            sub = ctk.CTkLabel(card, text="%s\non screen %d s" % (c.get("why", ""), c["seconds"]), text_color="#b8b8c0", font=ctk.CTkFont(size=11), anchor="w", justify="left", wraplength=135)
            sub.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 6))
            clicks += [top, sub]
            for w in clicks:
                w.bind("<Button-1>", lambda e, ch=c: self.pick(ch))
            self.cards.append((card, c))

    def pick(self, char):
        self.selected = char
        for card, c in self.cards:
            card.configure(fg_color=CARD_ON if c is char else CARD)
        self.say("Chosen. Press Make GIF.")

    # ------------------------------------------------------------------ results and preview
    def add_result(self, r):
        self.results.append(r)
        row = len(self.results)
        text = "%s   %.1f MB, %dx%d, %.1f s" % (r["style"], r["bytes"] / 1e6, r["w"], r["h"], r["seconds"])
        b = ctk.CTkButton(self.result_box, text=text, anchor="w", height=26, fg_color=CARD, hover_color="#3b3b43", command=lambda p=r["path"]: self.play(p))
        b.grid(row=row, column=0, sticky="ew", pady=1)
        self.play(r["path"])

    def play(self, path):
        if self.play_job:
            self.after_cancel(self.play_job)
            self.play_job = None
        try:
            im = Image.open(path)
            self.play_src = [(f.convert("RGB").copy(), max(20, f.info.get("duration", 40))) for f in ImageSequence.Iterator(im)]
        except OSError as e:
            self.say("Could not open the GIF: %s" % e)
            return
        self.preview_title.configure(text=os.path.basename(path))
        self._refit()

    def _refit(self):
        src = getattr(self, "play_src", None)
        if not src:
            return
        w, h = max(50, self.stage.winfo_width()), max(50, self.stage.winfo_height())
        iw, ih = src[0][0].size
        s = min(w / iw, h / ih, 1.5)
        size = (max(1, int(iw * s)), max(1, int(ih * s)))
        if getattr(self, "_play_size", None) == (size, id(src)):
            return
        self._play_size = (size, id(src))
        if self.play_job:
            self.after_cancel(self.play_job)
        self.play_frames = [(ImageTk.PhotoImage(f.resize(size, Image.BILINEAR)), d) for f, d in src]
        self.play_i = 0
        self.play_next = time.perf_counter()
        self._tick()

    def _tick(self):
        if not self.play_frames:
            return
        n = len(self.play_frames)
        now = time.perf_counter()
        # skip frames instead of slowing down, so the preview keeps the true speed
        while self.play_next + self.play_frames[self.play_i % n][1] / 1000 < now:
            self.play_next += self.play_frames[self.play_i % n][1] / 1000
            self.play_i += 1
        img, d = self.play_frames[self.play_i % n]
        self.stage.configure(image=img)
        wait = self.play_next + d / 1000 - now
        self.play_next += d / 1000
        self.play_i += 1
        self.play_job = self.after(max(1, int(wait * 1000)), self._tick)


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CharacterGifMaker")
        except OSError:
            pass
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import cv2
    cv2.setNumThreads(2)
    App().mainloop()


if __name__ == "__main__":
    main()
