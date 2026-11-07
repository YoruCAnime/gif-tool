"""Run with the launcher's pythonw: the app module must import with no console, and find ffmpeg and the GPU."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["CGM_SETTINGS"] = os.path.join(ROOT, "_devtest", "gui", "settings.json")
result = os.path.join(ROOT, "_devtest", "pythonw_check.txt")
lines = ["console stdout was None: %s" % (sys.stdout is None)]
try:
    import app  # noqa: F401
    from cgm import models
    import onnxruntime
    det = models.get("face")
    lines += ["import ok", "ort " + onnxruntime.__version__, "face provider " + det.s.get_providers()[0], "ffmpeg " + str(shutil.which("ffmpeg")),
              "clip " + str(models.get("clip") is not None)]
except Exception as e:
    lines.append("FAILED %r" % e)
with open(result, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
