"""Local anime models: face detector, CCIP (same character?), CLIP (text against picture).

All three are read from the Hugging Face cache, the app downloads nothing (see README.txt). ONNX
runs on the GPU when onnxruntime can use CUDA (a version built for another CUDA silently falls back
to the CPU). The app also works, slower, on the CPU.
"""
import glob
import json
import os
import threading

import cv2
import numpy as np

HUB = os.path.expanduser(r"~\.cache\huggingface\hub")
_lock = threading.Lock()
_cache = {}


def hub_file(pattern):
    hits = glob.glob(os.path.join(HUB, pattern))
    if not hits:
        raise FileNotFoundError("Model not found in the Hugging Face cache: " + pattern)
    return hits[0]


def _session(path, threads=2):
    try:
        import torch  # noqa: F401  (puts the CUDA 12 dlls on the path for onnxruntime)
    except Exception:
        pass
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
    return ort.InferenceSession(path, opts, providers=providers)


def _nms(boxes, scores, iou=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (area_i + area_o - inter + 1e-6) < iou]
    return keep


class FaceDetector:
    def __init__(self):
        self.s = _session(hub_file("models--deepghs--anime_face_detection/snapshots/*/face_detect_v1.4_s/model.onnx"))

    def detect(self, rgb, conf=0.4):
        """Returns [(x1, y1, x2, y2, score)] in the pixels of rgb."""
        h, w = rgb.shape[:2]
        nw, nh = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
        img = rgb if (nw, nh) == (w, h) else cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        data = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.s.run(None, {"images": data})[0][0]
        if out.shape[0] < out.shape[1]:
            out = out.T
        scores = out[:, 4:].max(axis=1)
        out, scores = out[scores > conf], scores[scores > conf]
        if not len(out):
            return []
        xy, wh = out[:, :2], out[:, 2:4]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        boxes[:, [0, 2]] *= w / nw
        boxes[:, [1, 3]] *= h / nh
        return [(*boxes[i].tolist(), float(scores[i])) for i in _nms(boxes, scores)]


class CCIP:
    """Says whether two anime pictures show the same character."""

    def __init__(self):
        base = "models--deepghs--ccip_onnx/snapshots/*/ccip-caformer-24-randaug-pruned/"
        self.feat = _session(hub_file(base + "model_feat.onnx"))
        self.metric = None
        self._metric_path = hub_file(base + "model_metrics.onnx")
        with open(hub_file(base + "metrics.json"), encoding="utf-8") as f:
            self.threshold = float(json.load(f)["threshold"])

    def embed(self, rgb_list, batch=16):
        out = []
        for i in range(0, len(rgb_list), batch):
            arr = np.stack([cv2.resize(x, (384, 384), interpolation=cv2.INTER_LINEAR) for x in rgb_list[i:i + batch]])
            arr = ((arr.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2)
            out.append(self.feat.run(None, {"input": arr})[0])
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def distances(self, feats):
        """All-against-all difference matrix (below self.threshold = same character)."""
        if self.metric is None:
            self.metric = _session(self._metric_path)
        return self.metric.run(None, {"input": feats.astype(np.float32)})[0]


class Clip:
    """OpenAI CLIP through transformers. None when unavailable."""

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        path = glob.glob(os.path.join(HUB, "models--openai--clip-vit-base-patch32", "snapshots", "*"))[0]
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(path, local_files_only=True)

    def images(self, rgb_list, batch=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(rgb_list), batch):
                inp = self.proc(images=list(rgb_list[i:i + batch]), return_tensors="pt").to(self.device)
                f = self.model.get_image_features(**inp)
                out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def texts(self, texts):
        with self.torch.no_grad():
            inp = self.proc(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = self.model.get_text_features(**inp)
            return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()


def get(name):
    """Loads each model once. get('clip') returns None when CLIP cannot run here."""
    with _lock:
        if name not in _cache:
            if name == "face":
                _cache[name] = FaceDetector()
            elif name == "ccip":
                _cache[name] = CCIP()
            else:
                try:
                    _cache[name] = Clip()
                except Exception:
                    _cache[name] = None
        return _cache[name]
