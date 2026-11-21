import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MODELS_DIR = Path("models")

SWAPPER_URLS = {
    "inswapper_128_fp16": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128_fp16.onnx",
    ],
    "inswapper_128": [
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    ],
    "hyperswap_1a_256": [
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/hyperswap_1a_256.onnx",
    ],
    "alphaface_256": [
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/alphaface_256.onnx",
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/alphaface_256.onnx",
    ],
    "ghost_1_256": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx",
    ],
    "simswap_256": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/simswap_256.onnx",
    ],
}

AUX_MODEL_URLS = {
    "xseg.onnx": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/xseg_1.onnx",
    ],
    "face_parser.onnx": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/face_parser.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/face_parser.onnx",
    ],
    "restoreformer_plus_plus.onnx": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
    ],
    "GPEN-BFR-512.onnx": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
    ],
    "CodeFormerv0.1.onnx": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx",
    ],
}

_CACHE: Dict[str, object] = {
    "analyser": None,
    "det_size": None,
    "swapper_name": None,
    "swapper_obj": None,
    "enhancer_name": None,
    "enhancer_obj": None,
    "xseg_obj": None,
    "parser_obj": None,
    "source_key": None,
    "source_face": None,
}

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def run_cmd(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def ensure_packages(use_gradio: bool) -> None:
    pkgs = ["insightface==0.7.3", "onnxruntime-gpu", "opencv-python", "scikit-image"]
    if use_gradio:
        pkgs.extend(["gradio==5.13.0", "pydantic==2.10.6"])
    print("[SETUP] Memastikan paket ringan & CUDA ONNX terpasang...", flush=True)
    run_cmd(f"{sys.executable} -m pip install -q " + " ".join(pkgs))


def download_from_list(filename: str, urls: List[str], min_bytes: int = 5_000_000) -> Optional[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    for url in urls:
        print(f"[UNDUH] {url}", flush=True)
        rc = run_cmd(f'wget -q --show-progress "{url}" -O "{dest}"')
        if rc == 0 and dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"[SUKSES] Tersimpan: {dest}", flush=True)
            return dest
        if dest.exists():
            dest.unlink()
    return None


def _rotate_points_back(pts: np.ndarray, angle: int, orig_w: int, orig_h: int) -> np.ndarray:
    """Mengembalikan koordinat landmark/bbox dari gambar terputar (90/180/270) ke orientasi asli."""
    if angle == 0:
        return pts
    out = pts.copy()
    if angle == 90:
        # cv2.ROTATE_90_CLOCKWISE: (x_rot, y_rot) -> (y_rot, orig_h - 1 - x_rot)
        out[:, 0] = pts[:, 1]
        out[:, 1] = (orig_h - 1) - pts[:, 0]
    elif angle == 270:
        # cv2.ROTATE_90_COUNTERCLOCKWISE: (x_rot, y_rot) -> (orig_w - 1 - y_rot, x_rot)
        out[:, 0] = (orig_w - 1) - pts[:, 1]
        out[:, 1] = pts[:, 0]
    elif angle == 180:
        out[:, 0] = (orig_w - 1) - pts[:, 0]
        out[:, 1] = (orig_h - 1) - pts[:, 1]
    return out


def detect_faces_angle_aware(app, frame: np.ndarray, enable_rotation: bool = True):
    """
    Deteksi wajah dengan rotasi otomatis (0°, 90°, 270°) agar wajah miring/rebah
    tetap terdeteksi tegak dan landmark mata/mulut tidak meleset.
    """
    faces = app.get(frame)
    if faces or not enable_rotation:
        return faces

    h, w = frame.shape[:2]
    rot_configs = [
        (90, cv2.ROTATE_90_CLOCKWISE),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for angle, rot_flag in rot_configs:
        rot_img = cv2.rotate(frame, rot_flag)
        rot_faces = app.get(rot_img)
        if rot_faces:
            restored_faces = []
            for f in rot_faces:
                x1, y1, x2, y2 = f.bbox[:4]
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                orig_corners = _rotate_points_back(corners, angle, w, h)
                new_bbox = np.array(
                    [
                        np.min(orig_corners[:, 0]),
                        np.min(orig_corners[:, 1]),
                        np.max(orig_corners[:, 0]),
                        np.max(orig_corners[:, 1]),
                    ],
                    dtype=np.float32,
                )
                new_kps = _rotate_points_back(np.array(f.kps, dtype=np.float32), angle, w, h)
                restored_faces.append(
                    SimpleNamespace(
                        bbox=new_bbox,
                        kps=new_kps,
                        det_score=getattr(f, "det_score", 0.9),
                        embedding=getattr(f, "embedding", None),
                        normed_embedding=getattr(f, "normed_embedding", None),
                    )
                )
            return restored_faces
    return []


def align_face_crop(image: np.ndarray, kps: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    scale = crop_size / 112.0
    dst = ARCFACE_DST * scale
    if crop_size == 256:
        dst = (ARCFACE_DST - 56.0) * (crop_size / 128.0) + (crop_size / 2.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_CUBIC, borderValue=0.0)
    return aligned, M


def match_color_lab(source_crop: np.ndarray, target_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(source_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    valid = mask > 0.35
    if np.count_nonzero(valid) < 50:
        return source_crop

    for ch in range(3):
        s_vals = src_lab[:, :, ch][valid]
        t_vals = tgt_lab[:, :, ch][valid]
        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals)) + 1e-5
        t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals)) + 1e-5
        src_lab[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

    matched = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


def restore_eye_expression(swapped_crop: np.ndarray, target_crop: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Memindahkan dinamika terang-gelap kelopak mata & arah bayangan bola mata dari wajah target
    ke wajah hasil swap agar tatapan mata lebih hidup dan sinkron dengan ekspresi asli.
    """
    if strength <= 0.0:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    out = swapped_crop.copy()

    # ROI kedua mata pada koordinat aligned crop
    y1, y2 = int(h * 0.32), int(h * 0.54)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    swap_roi = out[y1:y2, x1:x2]
    tgt_roi = target_crop[y1:y2, x1:x2]
    if swap_roi.size == 0:
        return out

    s_lab = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Ekstrak peta bayangan/kelopak mata frekuensi tinggi dari target
    t_l = t_lab[:, :, 0]
    t_blur = cv2.GaussianBlur(t_l, (0, 0), 2.5)
    t_detail = t_l - t_blur

    s_lab[:, :, 0] = np.clip(s_lab[:, :, 0] + t_detail * float(np.clip(strength, 0.0, 0.8)), 0, 255)
    restored_roi = cv2.cvtColor(s_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Buat gradien halus di pinggir kotak mata
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=((x2 - x1) // 2, (y2 - y1) // 2),
        axes=(int((x2 - x1) * 0.45), int((y2 - y1) * 0.42)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )
    mask = cv2.GaussianBlur(mask, (9, 9), 0)[..., None]
    blended = restored_roi.astype(np.float32) * mask + swap_roi.astype(np.float32) * (1.0 - mask)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


class UniversalOnnxSwapper:
    def __init__(self, model_path: Path, model_name: str):
        import onnx
        import onnxruntime as ort

        self.model_name = model_name
        self.model_path = model_path
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.is_inswapper = "inswapper" in model_name.lower()
        self.emap = None
        if self.is_inswapper:
            try:
                from onnx import numpy_helper
                model_proto = onnx.load(str(model_path))
                self.emap = numpy_helper.to_array(model_proto.graph.initializer[-1])
            except Exception:
                self.emap = None

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.inputs = self.session.get_inputs()

        self.crop_size = 128
        for inp in self.inputs:
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int):
                self.crop_size = int(inp.shape[2])
                break

    def swap_crop(self, aligned_bgr: np.ndarray, src_face) -> np.ndarray:
        img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if not self.is_inswapper:
            img_rgb = (img_rgb - 0.5) / 0.5
        img_tensor = np.transpose(img_rgb, (2, 0, 1))[None, ...].astype(np.float32)

        emb = src_face.normed_embedding.reshape(1, -1).astype(np.float32)
        if self.is_inswapper and self.emap is not None:
            latent = np.dot(emb, self.emap)
            latent /= np.linalg.norm(latent) + 1e-6
        else:
            latent = emb

        feed = {}
        for inp in self.inputs:
            if len(inp.shape) == 4:
                feed[inp.name] = img_tensor
            else:
                feed[inp.name] = latent.astype(np.float32)

        out = self.session.run(None, feed)[0][0]
        out = np.transpose(out, (1, 2, 0))
        if not self.is_inswapper:
            out = out * 0.5 + 0.5
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class XSegMasker:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 256
        self.nhwc = len(shape) == 4 and shape[-1] == 3

    def predict_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        img = cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = img_rgb[None, ...] if self.nhwc else np.transpose(img_rgb, (2, 0, 1))[None, ...]
        out = self.session.run(None, {self.input_name: tensor.astype(np.float32)})[0][0]
        if out.ndim == 3:
            out = out[:, :, 0] if out.shape[-1] == 1 else out[0]
        mask = np.clip(out, 0.0, 1.0).astype(np.float32)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


class BiSeNetFaceParser:
    """
    Face Parser (BiSeNet) untuk memisahkan area wajah (kulit, mata, alis, hidung, bibir)
    dari rambut/poni sehingga poni yang jatuh di dahi/mata tidak tertimpa.
    """

    # Kelas CelebAMask-HQ standar: 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 10=nose, 11=mouth, 12=u_lip, 13=l_lip
    FACE_CLASSES = {1, 2, 3, 4, 5, 10, 11, 12, 13}

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 512

    def predict_face_region_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        img = cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (img_rgb - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

        out = self.session.run(None, {self.input_name: tensor})[0][0]
        if out.ndim == 3:
            cls_map = np.argmax(out, axis=0)
        else:
            cls_map = out.astype(np.int32)

        mask = np.isin(cls_map, list(self.FACE_CLASSES)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


class OnnxFaceEnhancer:
    def __init__(self, model_path: Path, model_type: str):
        import onnxruntime as ort

        self.model_type = model_type
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.size = int(self.input_shape[2]) if isinstance(self.input_shape[2], int) else 512
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def enhance_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        img = cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - 0.5) / 0.5
        tensor = np.transpose(img_norm, (2, 0, 1))[None, ...].astype(np.float32)

        feed = {self.input_name: tensor}
        if len(self.input_names) > 1:
            feed[self.input_names[1]] = np.array([0.5], dtype=np.float64)

        out = self.session.run(None, feed)[0][0]
        out = np.transpose(out, (1, 2, 0))
        out = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def get_face_analyser(det_size: int = 640):
    if _CACHE["analyser"] is not None and _CACHE["det_size"] == det_size:
        return _CACHE["analyser"]

    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    _CACHE["analyser"] = app
    _CACHE["det_size"] = det_size
    return app


def get_swapper(model_name: str) -> UniversalOnnxSwapper:
    if _CACHE["swapper_name"] == model_name and _CACHE["swapper_obj"] is not None:
        return _CACHE["swapper_obj"]

    urls = SWAPPER_URLS.get(model_name, [])
    mpath = download_from_list(f"{model_name}.onnx", urls, min_bytes=20_000_000)
    if mpath is None and model_name != "inswapper_128_fp16":
        model_name = "inswapper_128_fp16"
        mpath = download_from_list("inswapper_128_fp16.onnx", SWAPPER_URLS["inswapper_128_fp16"], min_bytes=20_000_000)

    if mpath is None:
        raise RuntimeError("Gagal mengunduh model Face Swapper ONNX.")

    obj = UniversalOnnxSwapper(mpath, model_name)
    _CACHE["swapper_name"] = model_name
    _CACHE["swapper_obj"] = obj
    return obj


def get_xseg() -> Optional[XSegMasker]:
    if _CACHE["xseg_obj"] is not None:
        return _CACHE["xseg_obj"]
    xpath = download_from_list("xseg.onnx", AUX_MODEL_URLS["xseg.onnx"], min_bytes=10_000_000)
    if xpath is None:
        return None
    obj = XSegMasker(xpath)
    _CACHE["xseg_obj"] = obj
    return obj


def get_face_parser() -> Optional[BiSeNetFaceParser]:
    if _CACHE["parser_obj"] is not None:
        return _CACHE["parser_obj"]
    ppath = download_from_list("face_parser.onnx", AUX_MODEL_URLS["face_parser.onnx"], min_bytes=10_000_000)
    if ppath is None:
        return None
    obj = BiSeNetFaceParser(ppath)
    _CACHE["parser_obj"] = obj
    return obj


def get_enhancer(enhancer_choice: str) -> Optional[OnnxFaceEnhancer]:
    if enhancer_choice == "none":
        return None
    if _CACHE["enhancer_name"] == enhancer_choice and _CACHE["enhancer_obj"] is not None:
        return _CACHE["enhancer_obj"]

    fname_map = {
        "restoreformer++": "restoreformer_plus_plus.onnx",
        "gpen-512": "GPEN-BFR-512.onnx",
        "codeformer": "CodeFormerv0.1.onnx",
    }
    fname = fname_map[enhancer_choice]
    mpath = download_from_list(fname, AUX_MODEL_URLS[fname], min_bytes=50_000_000)
    if mpath is None:
        return None
    obj = OnnxFaceEnhancer(mpath, enhancer_choice)
    _CACHE["enhancer_name"] = enhancer_choice
    _CACHE["enhancer_obj"] = obj
    return obj


def load_source_face(source_file: str, app):
    src_path = Path(source_file)
    cache_key = f"{src_path.resolve()}_{src_path.stat().st_mtime}"
    if _CACHE["source_key"] == cache_key and _CACHE["source_face"] is not None:
        return _CACHE["source_face"]

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    detected_faces = []

    if src_path.suffix.lower() in {".fsz", ".zip"}:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(src_path, "r") as zf:
                zf.extractall(tmpdir)
            all_imgs = [
                Path(root) / fname
                for root, _, files in os.walk(tmpdir)
                for fname in sorted(files)
                if (Path(root) / fname).suffix.lower() in img_exts
            ]
            if len(all_imgs) > 30:
                step = len(all_imgs) / 30.0
                all_imgs = [all_imgs[int(i * step)] for i in range(30)]

            for fpath in all_imgs:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                faces = detect_faces_angle_aware(app, img, enable_rotation=True)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    detected_faces.append(best)

        gc.collect()
        if not detected_faces:
            raise RuntimeError("Tidak ada wajah yang terdeteksi di dalam faceset (.fsz).")

        raw_embs = [f.normed_embedding for f in detected_faces if getattr(f, "normed_embedding", None) is not None]
        mean_emb = np.mean(np.stack(raw_embs, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_emb))
        normed = (mean_emb / norm) if norm > 0 else mean_emb
        merged_face = SimpleNamespace(
            embedding=mean_emb.astype(np.float32),
            normed_embedding=normed.astype(np.float32),
        )
        _CACHE["source_key"] = cache_key
        _CACHE["source_face"] = merged_face
        return merged_face

    src_img = cv2.imread(str(src_path))
    if src_img is None:
        raise FileNotFoundError(f"Gagal membaca file sumber: {src_path}")
    faces = detect_faces_angle_aware(app, src_img, enable_rotation=True)
    if not faces:
        raise RuntimeError("Wajah tidak terdeteksi pada gambar sumber.")
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    _CACHE["source_key"] = cache_key
    _CACHE["source_face"] = best_face
    return best_face


def process_face_advanced(
    frame: np.ndarray,
    face,
    src_face,
    swapper: UniversalOnnxSwapper,
    enhancer_obj: Optional[OnnxFaceEnhancer],
    xseg_obj: Optional[XSegMasker],
    parser_obj: Optional[BiSeNetFaceParser],
    swap_weight: float = 1.0,
    enhancer_blend: float = 0.30,
    eye_expr_strength: float = 0.35,
    use_color_match: bool = True,
    use_xseg: bool = True,
    use_face_parser: bool = True,
    protect_mouth: bool = True,
    mask_blur: int = 13,
) -> np.ndarray:
    h, w = frame.shape[:2]
    crop_size = swapper.crop_size

    aligned_tgt, M = align_face_crop(frame, face.kps, crop_size=crop_size)
    swapped_crop = swapper.swap_crop(aligned_tgt, src_face)

    # 1. Buat mask oval dasar wajah
    base_mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    cv2.ellipse(
        base_mask,
        center=(crop_size // 2, int(crop_size * 0.55)),
        axes=(int(crop_size * 0.44), int(crop_size * 0.50)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )
    top_pad = int(crop_size * 0.05)
    bot_pad = int(crop_size * 0.03)
    base_mask[:top_pad, :] = 0.0
    base_mask[crop_size - bot_pad :, :] = 0.0

    combined_mask = base_mask.copy()

    # 2. BiSeNet Face Parser: memisahkan wajah dari poni rambut agar poni di dahi/mata tidak tertimpa
    if use_face_parser and parser_obj is not None:
        region_mask = parser_obj.predict_face_region_mask(aligned_tgt)
        combined_mask = combined_mask * region_mask

    # 3. XSeg Occlusion Mask + proteksi khusus zona mulut
    if use_xseg and xseg_obj is not None:
        occ_mask = xseg_obj.predict_mask(aligned_tgt)
        if protect_mouth:
            mouth_zone = np.zeros((crop_size, crop_size), dtype=np.float32)
            cv2.ellipse(
                mouth_zone,
                center=(crop_size // 2, int(crop_size * 0.78)),
                axes=(int(crop_size * 0.24), int(crop_size * 0.14)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1.0,
                thickness=-1,
            )
            occ_mask = np.maximum(occ_mask, mouth_zone)
        combined_mask = combined_mask * occ_mask

    # 4. Samakan warna kulit dengan pencahayaan target
    if use_color_match:
        swapped_crop = match_color_lab(swapped_crop, aligned_tgt, combined_mask)

    # 5. Expression Restorer pada mata (mempertahankan arah bayangan & kelopak mata target)
    if eye_expr_strength > 0.0:
        swapped_crop = restore_eye_expression(swapped_crop, aligned_tgt, strength=float(eye_expr_strength))

    # 6. Face Enhancer (RestoreFormer++ / GPEN-512)
    if enhancer_obj is not None and enhancer_blend > 0.0:
        enhanced = enhancer_obj.enhance_crop(swapped_crop)
        eb = float(np.clip(enhancer_blend, 0.0, 1.0))
        swapped_crop = np.clip(
            enhanced.astype(np.float32) * eb + swapped_crop.astype(np.float32) * (1.0 - eb),
            0,
            255,
        ).astype(np.uint8)

    k = max(5, int(mask_blur) | 1)
    soft_mask = cv2.GaussianBlur(combined_mask, (k, k), 0) * float(np.clip(swap_weight, 0.0, 1.0))

    inv_M = cv2.invertAffineTransform(M)
    warped_face = cv2.warpAffine(swapped_crop, inv_M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.warpAffine(soft_mask, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]

    out = warped_face.astype(np.float32) * warped_mask + frame.astype(np.float32) * (1.0 - warped_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def select_target_faces(faces, mode: str = "largest"):
    if not faces:
        return []
    if mode == "all":
        return faces
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return [largest]


def get_video_frame_count(target_file: Optional[str]) -> int:
    if not target_file:
        return 1
    p = Path(target_file)
    if p.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return 1
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, total)


def preview_single_frame(
    source_file: Optional[str],
    target_file: Optional[str],
    frame_index: int,
    swapper_model: str,
    swap_weight: float,
    enhancer: str,
    enhancer_blend: float,
    eye_expr_strength: float,
    use_angle_rot: bool,
    use_face_parser: bool,
    use_xseg: bool,
    protect_mouth: bool,
    use_color_match: bool,
    face_mode: str,
    det_size: int,
    mask_blur: int,
) -> Tuple[Optional[np.ndarray], str]:
    try:
        if not source_file or not target_file:
            return None, "Unggah Sumber Wajah (.jpg/.png/.fsz) dan Target, lalu klik Cek Preview."

        app = get_face_analyser(det_size=int(det_size))
        swapper = get_swapper(swapper_model)
        enhancer_obj = get_enhancer(enhancer)
        xseg_obj = get_xseg() if use_xseg else None
        parser_obj = get_face_parser() if use_face_parser else None
        src_face = load_source_face(source_file, app)

        target_path = Path(target_file)
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if target_path.suffix.lower() in video_exts:
            cap = cv2.VideoCapture(str(target_path))
            if not cap.isOpened():
                return None, "Gagal membuka file video target."
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            safe_idx = max(0, min(total - 1, int(frame_index)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None, f"Gagal membaca frame ke-{safe_idx}."
            status = f"Preview Frame {safe_idx + 1}/{total} | Model: {swapper.model_name}"
        else:
            frame = cv2.imread(str(target_path))
            if frame is None:
                return None, "Gagal membaca gambar target."
            status = f"Preview Gambar | Model: {swapper.model_name}"

        out_frame = frame.copy()
        raw_faces = detect_faces_angle_aware(app, frame, enable_rotation=bool(use_angle_rot))
        faces = select_target_faces(raw_faces, mode=face_mode)
        if not faces:
            status += " (Wajah tidak terdeteksi pada frame ini)"
        for face in faces:
            out_frame = process_face_advanced(
                out_frame,
                face,
                src_face,
                swapper=swapper,
                enhancer_obj=enhancer_obj,
                xseg_obj=xseg_obj,
                parser_obj=parser_obj,
                swap_weight=float(swap_weight),
                enhancer_blend=float(enhancer_blend),
                eye_expr_strength=float(eye_expr_strength),
                use_color_match=bool(use_color_match),
                use_xseg=bool(use_xseg),
                use_face_parser=bool(use_face_parser),
                protect_mouth=bool(protect_mouth),
                mask_blur=int(mask_blur),
            )

        return cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), status
    except Exception as exc:
        return None, f"Error Preview: {exc}"


def run_render_full(
    source_file: str,
    target_file: str,
    swapper_model: str,
    swap_weight: float,
    enhancer: str,
    enhancer_blend: float,
    eye_expr_strength: float,
    use_angle_rot: bool,
    use_face_parser: bool,
    use_xseg: bool,
    protect_mouth: bool,
    use_color_match: bool,
    face_mode: str,
    det_size: int,
    mask_blur: int,
    progress=None,
) -> Tuple[str, str]:
    if not source_file or not target_file:
        raise ValueError("Sumber wajah dan target wajib diisi.")

    app = get_face_analyser(det_size=int(det_size))
    swapper = get_swapper(swapper_model)
    enhancer_obj = get_enhancer(enhancer)
    xseg_obj = get_xseg() if use_xseg else None
    parser_obj = get_face_parser() if use_face_parser else None
    src_face = load_source_face(source_file, app)

    target_path = Path(target_file)
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if target_path.suffix.lower() in video_exts:
        out_path = out_dir / "hasil_swap.mp4"
        temp_video = out_dir / "hasil_swap.raw.mp4"

        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Gagal membuka video target: {target_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            out_frame = frame.copy()
            raw_faces = detect_faces_angle_aware(app, frame, enable_rotation=bool(use_angle_rot))
            faces = select_target_faces(raw_faces, mode=face_mode)
            for face in faces:
                out_frame = process_face_advanced(
                    out_frame,
                    face,
                    src_face,
                    swapper=swapper,
                    enhancer_obj=enhancer_obj,
                    xseg_obj=xseg_obj,
                    parser_obj=parser_obj,
                    swap_weight=float(swap_weight),
                    enhancer_blend=float(enhancer_blend),
                    eye_expr_strength=float(eye_expr_strength),
                    use_color_match=bool(use_color_match),
                    use_xseg=bool(use_xseg),
                    use_face_parser=bool(use_face_parser),
                    protect_mouth=bool(protect_mouth),
                    mask_blur=int(mask_blur),
                )
            writer.write(out_frame)

            if idx % 20 == 0 or idx == total:
                gc.collect()
                if progress is not None:
                    progress((idx, total), desc=f"Merender frame {idx}/{total}")
                print(f"[CUDA RENDER] Frame {idx}/{total}", flush=True)

        cap.release()
        writer.release()
        gc.collect()

        mux_cmd = (
            f'ffmpeg -y -i "{temp_video}" -i "{target_path}" '
            f'-c:v libx264 -preset fast -crf 18 -c:a aac -map 0:v:0 -map 1:a? -shortest "{out_path}"'
        )
        rc = run_cmd(mux_cmd)
        if rc == 0 and out_path.exists():
            temp_video.unlink(missing_ok=True)
        else:
            shutil.move(str(temp_video), str(out_path))
        return str(out_path), f"Selesai ({total} frame) -> {out_path}"

    tgt_img = cv2.imread(str(target_path))
    if tgt_img is None:
        raise FileNotFoundError(f"Gagal membaca foto target: {target_path}")
    out = tgt_img.copy()
    raw_faces = detect_faces_angle_aware(app, tgt_img, enable_rotation=bool(use_angle_rot))
    faces = select_target_faces(raw_faces, mode=face_mode)
    for face in faces:
        out = process_face_advanced(
            out,
            face,
            src_face,
            swapper=swapper,
            enhancer_obj=enhancer_obj,
            xseg_obj=xseg_obj,
            parser_obj=parser_obj,
            swap_weight=float(swap_weight),
            enhancer_blend=float(enhancer_blend),
            eye_expr_strength=float(eye_expr_strength),
            use_color_match=bool(use_color_match),
            use_xseg=bool(use_xseg),
            use_face_parser=bool(use_face_parser),
            protect_mouth=bool(protect_mouth),
            mask_blur=int(mask_blur),
        )
    img_out = out_dir / "hasil_swap.png"
    cv2.imwrite(str(img_out), out)
    return str(img_out), f"Selesai -> {img_out}"


def launch_fast_ui(share: bool = True) -> None:
    import gradio as gr
    import onnxruntime as ort

    providers = ort.get_available_providers()
    gpu_status = "CUDA GPU AKTIF" if "CUDAExecutionProvider" in providers else "CPU MODE (Aktifkan T4 GPU!)"

    with gr.Blocks(title="Fast FaceSwap CUDA UI") as demo:
        gr.Markdown(f"### Fast FaceSwap Pro UI — **{gpu_status}**")

        with gr.Row():
            source_in = gr.File(
                label="Sumber Wajah (.jpg / .png / .fsz)",
                type="filepath",
            )
            target_in = gr.File(
                label="Video atau Foto Target",
                type="filepath",
            )

        with gr.Row():
            swapper_in = gr.Dropdown(
                choices=list(SWAPPER_URLS.keys()),
                value="inswapper_128_fp16",
                label="Face Swapper Model",
                info="inswapper_128_fp16 paling konsisten untuk sudut miring & ekspresi.",
            )
            swap_weight_in = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=1.0,
                label="Face Swapper Weight",
                info="1.0 penuh agar hasil swap tidak transparan/tercampur.",
            )

        with gr.Row():
            enhancer_in = gr.Dropdown(
                choices=["restoreformer++", "gpen-512", "codeformer", "none"],
                value="restoreformer++",
                label="Face Enhancer",
                info="Mempertajam detail kulit, bibir & mata setelah swap.",
            )
            enhancer_blend_in = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
                label="Enhancer Blend",
                info="0.25–0.35 menjaga wajah & mata tetap natural.",
            )
            eye_expr_in = gr.Slider(
                minimum=0.0,
                maximum=0.8,
                step=0.05,
                value=0.35,
                label="Eye Expression Restorer",
                info="Mempertahankan arah bayangan kelopak & tatapan mata asli target.",
            )

        with gr.Row():
            use_angle_rot_in = gr.Checkbox(
                value=True,
                label="Auto Rotasi Deteksi (90°/270°)",
                info="Wajib aktif untuk kepala rebah/miring agar titik mata tidak miss.",
            )
            use_parser_in = gr.Checkbox(
                value=True,
                label="Face Parser (BiSeNet)",
                info="Memisahkan poni rambut agar tidak tertimpa kulit wajah.",
            )
            use_xseg_in = gr.Checkbox(
                value=True,
                label="XSeg Occlusion Mask",
                info="Menjaga tangan/rambut/kacamata di depan wajah tetap utuh.",
            )

        with gr.Row():
            protect_mouth_in = gr.Checkbox(
                value=True,
                label="Kunci Area Mulut",
                info="Mencegah bibir terpotong saat mulut terbuka/bicara.",
            )
            use_color_in = gr.Checkbox(
                value=True,
                label="Color Match (LAB)",
                info="Menyamakan warna kulit dengan pencahayaan video.",
            )
            face_mode_in = gr.Dropdown(
                choices=["largest", "all"],
                value="largest",
                label="Target Wajah",
                info="largest = wajah utama; all = semua wajah.",
            )

        with gr.Row():
            det_size_in = gr.Slider(
                minimum=320,
                maximum=960,
                step=64,
                value=640,
                label="Det Size (Presisi Landmark)",
                info="640 mengunci koordinat sudut mata & bibir dengan presisi.",
            )
            mask_blur_in = gr.Slider(
                minimum=5,
                maximum=35,
                step=2,
                value=13,
                label="Mask Blur (Kehalusan Tepi)",
                info="11–15 menghaluskan batas pipi tanpa mengaburkan mata/mulut.",
            )

        gr.Markdown("---")
        preview_img = gr.Image(label="Live Preview Hasil Per Frame", interactive=False)
        frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Geser Frame (Live Preview)")
        with gr.Row():
            preview_btn = gr.Button("Cek Preview Frame", variant="secondary")
            run_btn = gr.Button("Render Video Lengkap", variant="primary")

        preview_status = gr.Textbox(label="Status", interactive=False)
        output_file_box = gr.File(label="Download Hasil")

        all_params = [
            source_in,
            target_in,
            frame_slider,
            swapper_in,
            swap_weight_in,
            enhancer_in,
            enhancer_blend_in,
            eye_expr_in,
            use_angle_rot_in,
            use_parser_in,
            use_xseg_in,
            protect_mouth_in,
            use_color_in,
            face_mode_in,
            det_size_in,
            mask_blur_in,
        ]

        def on_target_uploaded(tgt):
            total = get_video_frame_count(tgt)
            return gr.update(minimum=0, maximum=max(0, total - 1), value=0), f"Video siap ({total} frame)."

        target_in.change(fn=on_target_uploaded, inputs=[target_in], outputs=[frame_slider, preview_status])
        frame_slider.release(fn=preview_single_frame, inputs=all_params, outputs=[preview_img, preview_status])
        preview_btn.click(fn=preview_single_frame, inputs=all_params, outputs=[preview_img, preview_status])

        render_params = [
            source_in,
            target_in,
            swapper_in,
            swap_weight_in,
            enhancer_in,
            enhancer_blend_in,
            eye_expr_in,
            use_angle_rot_in,
            use_parser_in,
            use_xseg_in,
            protect_mouth_in,
            use_color_in,
            face_mode_in,
            det_size_in,
            mask_blur_in,
        ]
        run_btn.click(fn=run_render_full, inputs=render_params, outputs=[output_file_box, preview_status])

    demo.queue(max_size=4).launch(share=share, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast FaceSwap CUDA UI")
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    if not args.skip_install:
        ensure_packages(use_gradio=True)
    launch_fast_ui(share=not args.no_share)


if __name__ == "__main__":
    main()
