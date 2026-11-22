import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

MODELS_DIR = Path("models")
FACESET_WORK_DIR = Path("faceset_workspace")
DEFAULT_DRIVE_WORKSPACE = Path("/content/drive/MyDrive/faceswap_workspace")

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
    "source_count": 0,
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


def release_mem(clean_cuda: bool = False) -> None:
    gc.collect()
    if clean_cuda and torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


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


def compute_job_id(source_file: str, target_file: str) -> str:
    sp = Path(source_file) if source_file else None
    tp = Path(target_file) if target_file else None
    s_size = sp.stat().st_size if (sp and sp.exists()) else 0
    t_size = tp.stat().st_size if (tp and tp.exists()) else 0
    raw = f"src_size={s_size}|tgt_size={t_size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def inspect_valid_mp4(video_path: Path) -> Tuple[bool, int]:
    """Memastikan file MP4 memiliki moov atom valid dan bisa dibaca minimal 1 frame."""
    if not video_path.exists() or video_path.stat().st_size < 1024:
        return False, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, 0
    f_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _ = cap.read()
    cap.release()
    if not ret or f_cnt <= 0:
        return False, 0
    return True, f_cnt


class DriveSegmentCheckpoint:
    """Menyimpan input persisten, segmen sementara & progres di Google Drive secara atomik agar bebas moov atom error & hilang input saat disconnect."""

    def __init__(self, workspace_root: Optional[Path] = None):
        root = workspace_root or DEFAULT_DRIVE_WORKSPACE
        if not root.parent.exists() and Path("/content/drive/MyDrive").exists():
            root = DEFAULT_DRIVE_WORKSPACE
        self.root = root
        self.temp_root = self.root / "temp_segments"
        self.output_dir = self.root / "output"
        self.inputs_dir = self.root / "persisted_inputs"
        try:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
            self.enabled = True
        except Exception:
            self.temp_root = Path("output/temp_segments")
            self.output_dir = Path("output")
            self.inputs_dir = Path("output/persisted_inputs")
            self.temp_root.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
            self.enabled = False

    def persist_input_file(self, file_path: Optional[str], prefix: str = "input") -> Optional[str]:
        if not file_path:
            return None
        src = Path(file_path)
        if not src.exists() or not src.is_file():
            return None
        try:
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
            if src.resolve().parent == self.inputs_dir.resolve():
                os.utime(str(src), None)
                return str(src.resolve())
            stat = src.stat()
            ext = src.suffix.lower()
            sig = hashlib.md5(f"{prefix}|{stat.st_size}".encode("utf-8")).hexdigest()[:10]
            dst = self.inputs_dir / f"{prefix}_{sig}{ext}"
            if not dst.exists() or dst.stat().st_size != stat.st_size:
                tmp_dst = self.inputs_dir / f".tmp_{prefix}_{sig}{ext}"
                shutil.copy2(str(src), str(tmp_dst))
                os.replace(str(tmp_dst), str(dst))
            else:
                os.utime(str(dst), None)
            return str(dst.resolve())
        except Exception:
            return str(src.resolve()) if src.exists() else None

    def find_latest_persisted_inputs(self) -> Tuple[Optional[str], Optional[str]]:
        """Mengambil file src_* dan tgt_* terbaru dari persisted_inputs jika state belum sempat tertulis."""
        if not self.inputs_dir.exists():
            return None, None
        srcs = sorted(
            [p for p in self.inputs_dir.glob("src_*") if p.is_file() and not p.name.startswith(".tmp_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        tgts = sorted(
            [p for p in self.inputs_dir.glob("tgt_*") if p.is_file() and not p.name.startswith(".tmp_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        s_out = str(srcs[0].resolve()) if srcs else None
        t_out = str(tgts[0].resolve()) if tgts else None
        return s_out, t_out

    def job_dir(self, job_id: str) -> Path:
        d = self.temp_root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "render_state.json"

    def get_clean_valid_segments(self, job_id: str) -> Tuple[List[Path], int]:
        d = self.job_dir(job_id)
        for junk in d.glob("_tmp_*"):
            try:
                junk.unlink()
            except Exception:
                pass

        valid_segs: List[Path] = []
        total_valid_frames = 0
        for p in sorted(d.glob("part_*.mp4")):
            ok, f_cnt = inspect_valid_mp4(p)
            if ok:
                valid_segs.append(p)
                total_valid_frames += f_cnt
            else:
                print(f"[CLEANUP] Menghapus segmen terpotong/korup akibat disconnect: {p.name}", flush=True)
                try:
                    p.unlink()
                except Exception:
                    pass
        return valid_segs, total_valid_frames

    def load_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        p = self.state_path(job_id)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def find_latest_unfinished(self) -> Optional[Dict[str, Any]]:
        if not self.temp_root.exists():
            return None
        found: List[Dict[str, Any]] = []
        for sub in self.temp_root.iterdir():
            if sub.is_dir():
                p = sub / "render_state.json"
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if not data.get("completed", False):
                            data["_mtime"] = p.stat().st_mtime
                            found.append(data)
                    except Exception:
                        pass
        if not found:
            return None
        found.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
        return found[0]

    def save_job_state(self, job_id: str, payload: Dict[str, Any]) -> None:
        p = self.state_path(job_id)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def clear_job(self, job_id: str) -> None:
        d = self.temp_root / job_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        release_mem(clean_cuda=True)

    def clear_all_workspace_files(self) -> None:
        """Menghapus semua segmen sementara, file hasil render di output, dan salinan input persisten."""
        for folder in (self.temp_root, self.output_dir, self.inputs_dir):
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
        release_mem(clean_cuda=True)

    def finalize_video(self, job_id: str, target_path: Path, out_name: str = "hasil_swap.mp4") -> Tuple[bool, str]:
        segs, _ = self.get_clean_valid_segments(job_id)
        if not segs:
            return False, "Tidak ada segmen valid ditemukan."

        local_tmp_dir = Path(tempfile.mkdtemp(prefix="faceswap_merge_"))
        try:
            concat_txt = local_tmp_dir / "concat_list.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for s in segs:
                    f.write(f"file '{s.resolve().as_posix()}'\n")

            merged_raw = local_tmp_dir / "_merged_raw.mp4"
            rc_concat = run_cmd(
                f'ffmpeg -y -f concat -safe 0 -i "{concat_txt}" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "{merged_raw}"'
            )
            if rc_concat != 0 or not merged_raw.exists() or merged_raw.stat().st_size == 0:
                return False, "Gagal menggabungkan potongan segmen."

            local_final = local_tmp_dir / out_name
            mux_cmd = (
                f'ffmpeg -y -i "{merged_raw}" -i "{target_path}" '
                f'-c:v copy -c:a aac -map 0:v:0 -map 1:a? -shortest "{local_final}"'
            )
            rc_mux = run_cmd(mux_cmd)
            if rc_mux != 0 or not local_final.exists() or local_final.stat().st_size == 0:
                shutil.move(str(merged_raw), str(local_final))

            final_out = self.output_dir / out_name
            tmp_drive_out = self.output_dir / f".tmp_{out_name}"
            shutil.copy2(str(local_final), str(tmp_drive_out))
            os.replace(str(tmp_drive_out), str(final_out))

            if final_out.exists() and final_out.stat().st_size > 0:
                self.clear_job(job_id)
                return True, str(final_out.resolve())

            return False, "Hasil akhir gagal disimpan ke Drive."
        finally:
            shutil.rmtree(local_tmp_dir, ignore_errors=True)


_CHECKPOINT_MGR: Optional[DriveSegmentCheckpoint] = None


def get_checkpoint_mgr(workspace_dir: Optional[str] = None) -> DriveSegmentCheckpoint:
    global _CHECKPOINT_MGR
    if _CHECKPOINT_MGR is None or workspace_dir is not None:
        _CHECKPOINT_MGR = DriveSegmentCheckpoint(Path(workspace_dir) if workspace_dir else None)
    return _CHECKPOINT_MGR


def _rotate_points_back(pts: np.ndarray, angle: int, orig_w: int, orig_h: int) -> np.ndarray:
    if angle == 0:
        return pts
    out = pts.copy()
    if angle == 90:
        out[:, 0] = pts[:, 1]
        out[:, 1] = (orig_h - 1) - pts[:, 0]
    elif angle == 270:
        out[:, 0] = (orig_w - 1) - pts[:, 1]
        out[:, 1] = pts[:, 0]
    elif angle == 180:
        out[:, 0] = (orig_w - 1) - pts[:, 0]
        out[:, 1] = (orig_h - 1) - pts[:, 1]
    return out


def detect_faces_angle_aware(app, frame: np.ndarray, enable_rotation: bool = True):
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


class TemporalFaceTracker:
    def __init__(self, alpha: float = 0.65, max_hold_frames: int = 3):
        self.alpha = float(np.clip(alpha, 0.2, 1.0))
        self.max_hold = int(max_hold_frames)
        self.prev_kps: Optional[np.ndarray] = None
        self.prev_bbox: Optional[np.ndarray] = None
        self.missed_count: int = 0

    def update(self, faces: List) -> List:
        if faces:
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            curr_kps = np.array(largest.kps, dtype=np.float32)
            curr_bbox = np.array(largest.bbox[:4], dtype=np.float32)

            if self.prev_kps is not None:
                dist = float(np.mean(np.linalg.norm(curr_kps - self.prev_kps, axis=1)))
                face_w = max(20.0, float(curr_bbox[2] - curr_bbox[0]))
                if dist < face_w * 0.45:
                    curr_kps = self.alpha * curr_kps + (1.0 - self.alpha) * self.prev_kps
                    curr_bbox = self.alpha * curr_bbox + (1.0 - self.alpha) * self.prev_bbox

            self.prev_kps = curr_kps
            self.prev_bbox = curr_bbox
            self.missed_count = 0

            return [
                SimpleNamespace(
                    bbox=curr_bbox,
                    kps=curr_kps,
                    det_score=getattr(largest, "det_score", 0.9),
                    embedding=getattr(largest, "embedding", None),
                    normed_embedding=getattr(largest, "normed_embedding", None),
                )
            ]

        if self.prev_kps is not None and self.missed_count < self.max_hold:
            self.missed_count += 1
            return [
                SimpleNamespace(
                    bbox=self.prev_bbox,
                    kps=self.prev_kps,
                    det_score=0.5,
                    embedding=None,
                    normed_embedding=None,
                )
            ]

        self.prev_kps = None
        self.prev_bbox = None
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
    if strength <= 0.0:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    out = swapped_crop.copy()

    y1, y2 = int(h * 0.32), int(h * 0.54)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    swap_roi = out[y1:y2, x1:x2]
    tgt_roi = target_crop[y1:y2, x1:x2]
    if swap_roi.size == 0:
        return out

    s_lab = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    t_l = t_lab[:, :, 0]
    t_blur = cv2.GaussianBlur(t_l, (0, 0), 2.5)
    t_detail = t_l - t_blur

    s_lab[:, :, 0] = np.clip(s_lab[:, :, 0] + t_detail * float(np.clip(strength, 0.0, 0.8)), 0, 255)
    restored_roi = cv2.cvtColor(s_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

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


def inspect_source_gallery(source_file: Optional[str]) -> Tuple[List[str], str]:
    """Mengekstrak dan menampilkan seluruh isi wajah di dalam .fsz atau foto tunggal untuk verifikasi."""
    if not source_file:
        return [], "Belum ada sumber wajah dipilih."

    src_path = Path(source_file)
    if not src_path.exists():
        return [], "File sumber tidak ditemukan."

    FACESET_WORK_DIR.mkdir(parents=True, exist_ok=True)
    inspect_dir = FACESET_WORK_DIR / "active_preview"
    if inspect_dir.exists():
        shutil.rmtree(inspect_dir)
    inspect_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    saved_paths: List[str] = []

    if src_path.suffix.lower() in {".fsz", ".zip"}:
        with zipfile.ZipFile(src_path, "r") as zf:
            idx = 0
            for member in sorted(zf.infolist(), key=lambda m: m.filename):
                if member.is_dir():
                    continue
                ext = Path(member.filename).suffix.lower()
                if ext in img_exts:
                    idx += 1
                    out_f = inspect_dir / f"face_{idx:03d}{ext}"
                    with zf.open(member) as s, open(out_f, "wb") as d:
                        shutil.copyfileobj(s, d)
                    saved_paths.append(str(out_f.resolve()))
        return saved_paths, f"Faceset terverifikasi: {len(saved_paths)} gambar wajah siap digabungkan."

    out_f = inspect_dir / src_path.name
    shutil.copy2(src_path, out_f)
    return [str(out_f.resolve())], f"1 foto sumber aktif: {src_path.name}"


def load_source_face(source_file: str, app, top_sharp_only: bool = False):
    src_path = Path(source_file)
    cache_key = f"{src_path.resolve()}_{src_path.stat().st_size}_{top_sharp_only}"
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

        if top_sharp_only and len(detected_faces) > 6:
            detected_faces = sorted(
                detected_faces,
                key=lambda f: float(getattr(f, "det_score", 0.0)),
                reverse=True,
            )[:6]

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
        _CACHE["source_count"] = len(raw_embs)
        print(f"[FACESET AKTIF] Menggabungkan {len(raw_embs)} embedding wajah dari {src_path.name}", flush=True)
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
    _CACHE["source_count"] = 1
    return best_face


# ================= TOOLS PEMBUAT FACESET (.FSZ) =================


def _crop_padded_face(img_bgr: np.ndarray, face, out_size: int = 256) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(bw, bh) * 1.45

    nx1 = max(0, int(cx - side * 0.5))
    ny1 = max(0, int(cy - side * 0.5))
    nx2 = min(w, int(cx + side * 0.5))
    ny2 = min(h, int(cy + side * 0.5))

    crop = img_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return cv2.resize(img_bgr, (out_size, out_size))
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def fsz_add_images(
    uploaded_files: Optional[List[str]],
    current_gallery: Optional[List[str]],
    det_size: int,
) -> Tuple[List[str], str]:
    if not uploaded_files:
        return current_gallery or [], "Pilih satu atau beberapa foto terlebih dahulu."

    app = get_face_analyser(det_size=int(det_size))
    builder_dir = FACESET_WORK_DIR / "builder_items"
    builder_dir.mkdir(parents=True, exist_ok=True)

    items = list(current_gallery) if current_gallery else []
    added = 0
    for fpath in uploaded_files:
        p = Path(fpath)
        if p.suffix.lower() in {".fsz", ".zip"}:
            with zipfile.ZipFile(p, "r") as zf:
                for member in sorted(zf.infolist(), key=lambda m: m.filename):
                    if member.is_dir():
                        continue
                    ext = Path(member.filename).suffix.lower()
                    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                        added += 1
                        out_f = builder_dir / f"crop_{len(items) + 1:03d}{ext}"
                        with zf.open(member) as s, open(out_f, "wb") as d:
                            shutil.copyfileobj(s, d)
                        items.append(str(out_f.resolve()))
            continue

        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = detect_faces_angle_aware(app, img, enable_rotation=True)
        if not faces:
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        crop = _crop_padded_face(img, best, out_size=256)
        added += 1
        out_f = builder_dir / f"crop_{len(items) + 1:03d}.png"
        cv2.imwrite(str(out_f), crop)
        items.append(str(out_f.resolve()))

    return items, f"Berhasil menambahkan {added} potongan wajah. Total dalam daftar: {len(items)} wajah."


def fsz_cut_from_video(
    video_file: Optional[str],
    frame_num: int,
    current_gallery: Optional[List[str]],
    det_size: int,
) -> Tuple[List[str], str]:
    if not video_file:
        return current_gallery or [], "Unggah video terlebih dahulu untuk memotong wajah dari frame."

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return current_gallery or [], "Gagal membuka video."
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    safe_idx = max(0, min(total - 1, int(frame_num)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return current_gallery or [], f"Gagal membaca frame ke-{safe_idx}."

    app = get_face_analyser(det_size=int(det_size))
    faces = detect_faces_angle_aware(app, frame, enable_rotation=True)
    if not faces:
        return current_gallery or [], f"Tidak ada wajah terdeteksi pada frame {safe_idx + 1}."

    builder_dir = FACESET_WORK_DIR / "builder_items"
    builder_dir.mkdir(parents=True, exist_ok=True)
    items = list(current_gallery) if current_gallery else []

    for face in faces:
        crop = _crop_padded_face(frame, face, out_size=256)
        out_f = builder_dir / f"vid_f{safe_idx + 1}_{len(items) + 1:03d}.png"
        cv2.imwrite(str(out_f), crop)
        items.append(str(out_f.resolve()))

    return items, f"Berhasil mengambil {len(faces)} wajah dari frame {safe_idx + 1}. Total: {len(items)} wajah."


def fsz_remove_selected(current_gallery: Optional[List[str]], selected_idx: Optional[int]) -> Tuple[List[str], str]:
    items = list(current_gallery) if current_gallery else []
    if not items:
        return [], "Daftar faceset sudah kosong."
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(items):
        items.pop()
    else:
        items.pop(selected_idx)
    return items, f"1 wajah dihapus. Sisa dalam daftar: {len(items)} wajah."


def fsz_clear_all() -> Tuple[List[str], str]:
    builder_dir = FACESET_WORK_DIR / "builder_items"
    if builder_dir.exists():
        shutil.rmtree(builder_dir)
    return [], "Semua wajah di daftar faceset telah dibersihkan."


def fsz_build_and_apply(
    current_gallery: Optional[List[str]],
    faceset_name: str,
) -> Tuple[Optional[str], Optional[str], List[str], str]:
    items = list(current_gallery) if current_gallery else []
    if not items:
        return None, None, [], "Daftar wajah kosong. Tambahkan minimal 1 foto/frame terlebih dahulu."

    clean_name = "".join(c for c in (faceset_name or "custom_faceset") if c.isalnum() or c in ("_", "-")).strip()
    if not clean_name:
        clean_name = "custom_faceset"

    ckpt_mgr = get_checkpoint_mgr()
    out_dir = ckpt_mgr.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fsz_path = out_dir / f"{clean_name}.fsz"

    with zipfile.ZipFile(fsz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, item_path in enumerate(items, start=1):
            p = Path(item_path)
            if p.exists():
                zf.write(p, arcname=f"face_{i:03d}{p.suffix.lower()}")

    persisted_fsz = ckpt_mgr.persist_input_file(str(fsz_path), prefix="src") or str(fsz_path.resolve())
    preview_list, _ = inspect_source_gallery(persisted_fsz)
    return (
        persisted_fsz,
        persisted_fsz,
        preview_list,
        f"Faceset '{fsz_path.name}' selesai dibuat ({len(items)} wajah) di Drive & langsung dipasang sebagai Sumber Wajah!",
    )


# ================= PROSES FACE SWAP =================


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

    if use_face_parser and parser_obj is not None:
        region_mask = parser_obj.predict_face_region_mask(aligned_tgt)
        combined_mask = combined_mask * region_mask

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

    if use_color_match:
        swapped_crop = match_color_lab(swapped_crop, aligned_tgt, combined_mask)

    if eye_expr_strength > 0.0:
        swapped_crop = restore_eye_expression(swapped_crop, aligned_tgt, strength=float(eye_expr_strength))

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
    use_temporal_smooth: bool,
    top_sharp_fsz: bool,
    face_mode: str,
    det_size: int,
    mask_blur: int,
) -> Tuple[Optional[np.ndarray], str]:
    try:
        if not source_file or not target_file:
            return None, "Unggah / buat Sumber Wajah (.jpg/.png/.fsz) dan Target, lalu klik Cek Preview."

        ckpt_mgr = get_checkpoint_mgr()
        source_file = ckpt_mgr.persist_input_file(source_file, prefix="src") or source_file
        target_file = ckpt_mgr.persist_input_file(target_file, prefix="tgt") or target_file

        app = get_face_analyser(det_size=int(det_size))
        swapper = get_swapper(swapper_model)
        enhancer_obj = get_enhancer(enhancer)
        xseg_obj = get_xseg() if use_xseg else None
        parser_obj = get_face_parser() if use_face_parser else None
        src_face = load_source_face(source_file, app, top_sharp_only=bool(top_sharp_fsz))
        src_cnt = int(_CACHE.get("source_count", 1))

        target_path = Path(target_file)
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if target_path.suffix.lower() in video_exts:
            cap = cv2.VideoCapture(str(target_path))
            if not cap.isOpened():
                return None, "Gagal membuka file video target."
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            safe_idx = max(0, min(total - 1, int(frame_index)))

            tracker = TemporalFaceTracker() if use_temporal_smooth else None
            start_idx = max(0, safe_idx - 2) if use_temporal_smooth else safe_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            frame = None
            faces = []
            for _ in range(start_idx, safe_idx + 1):
                ret, f_img = cap.read()
                if not ret or f_img is None:
                    break
                frame = f_img
                raw_f = detect_faces_angle_aware(app, frame, enable_rotation=bool(use_angle_rot))
                if tracker is not None:
                    faces = tracker.update(raw_f)
                else:
                    faces = select_target_faces(raw_f, mode=face_mode)

            cap.release()
            if frame is None:
                return None, f"Gagal membaca frame ke-{safe_idx}."
            status = f"Frame {safe_idx + 1}/{total} | {src_cnt} wajah sumber digabung | Model: {swapper.model_name}"
        else:
            frame = cv2.imread(str(target_path))
            if frame is None:
                return None, "Gagal membaca gambar target."
            raw_f = detect_faces_angle_aware(app, frame, enable_rotation=bool(use_angle_rot))
            faces = select_target_faces(raw_f, mode=face_mode)
            status = f"Preview Gambar | {src_cnt} wajah sumber digabung | Model: {swapper.model_name}"

        out_frame = frame.copy()
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
    use_temporal_smooth: bool,
    top_sharp_fsz: bool,
    face_mode: str,
    det_size: int,
    mask_blur: int,
    segment_frames: int = 50,
    fresh_start: bool = False,
    progress=None,
):
    ckpt_mgr = get_checkpoint_mgr()
    latest_unfinished = ckpt_mgr.find_latest_unfinished() if not fresh_start else None

    if not source_file and latest_unfinished:
        source_file = latest_unfinished.get("settings", {}).get("source_file", "")
    if not target_file and latest_unfinished:
        target_file = latest_unfinished.get("settings", {}).get("target_file", "")

    if not source_file or not target_file:
        fb_src, fb_tgt = ckpt_mgr.find_latest_persisted_inputs()
        if not source_file and fb_src:
            source_file = fb_src
        if not target_file and fb_tgt:
            target_file = fb_tgt

    if not source_file or not target_file:
        raise ValueError("Sumber wajah dan target wajib diisi.")

    persisted_src = ckpt_mgr.persist_input_file(source_file, prefix="src") or source_file
    persisted_tgt = ckpt_mgr.persist_input_file(target_file, prefix="tgt") or target_file

    app = get_face_analyser(det_size=int(det_size))
    swapper = get_swapper(swapper_model)
    enhancer_obj = get_enhancer(enhancer)
    xseg_obj = get_xseg() if use_xseg else None
    parser_obj = get_face_parser() if use_face_parser else None
    src_face = load_source_face(persisted_src, app, top_sharp_only=bool(top_sharp_fsz))
    src_cnt = int(_CACHE.get("source_count", 1))

    target_path = Path(persisted_tgt)
    out_dir = ckpt_mgr.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if target_path.suffix.lower() in video_exts:
        job_id = compute_job_id(persisted_src, persisted_tgt)

        if fresh_start:
            ckpt_mgr.clear_job(job_id)
            prev_state = None
        else:
            prev_state = ckpt_mgr.load_job_state(job_id)
            if prev_state is None and latest_unfinished is not None:
                cand_id = latest_unfinished.get("job_id")
                cand_tot = int(latest_unfinished.get("total_frames", 0))
                cap_chk = cv2.VideoCapture(str(target_path))
                cur_tot = int(cap_chk.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_chk.isOpened() else 0
                cap_chk.release()
                if cand_id and cand_tot > 0 and cand_tot == cur_tot:
                    job_id = cand_id
                    prev_state = latest_unfinished

        job_dir = ckpt_mgr.job_dir(job_id)

        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Gagal membuka video target: {target_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

        settings_snapshot = {
            "source_file": persisted_src,
            "target_file": persisted_tgt,
            "swapper_model": swapper_model,
            "swap_weight": float(swap_weight),
            "enhancer": enhancer,
            "enhancer_blend": float(enhancer_blend),
            "eye_expr_strength": float(eye_expr_strength),
            "use_angle_rot": bool(use_angle_rot),
            "use_face_parser": bool(use_face_parser),
            "use_xseg": bool(use_xseg),
            "protect_mouth": bool(protect_mouth),
            "use_color_match": bool(use_color_match),
            "use_temporal_smooth": bool(use_temporal_smooth),
            "top_sharp_fsz": bool(top_sharp_fsz),
            "face_mode": face_mode,
            "det_size": int(det_size),
            "mask_blur": int(mask_blur),
            "segment_frames": int(segment_frames),
        }

        if prev_state and not prev_state.get("completed", False):
            valid_segs, counted_frames = ckpt_mgr.get_clean_valid_segments(job_id)
            start_idx = counted_frames
            seg_num = len(valid_segs)
            print(
                f"[AUTO-RESUME] Melanjutkan otomatis dari frame {start_idx + 1}/{total} ({seg_num} segmen valid tersimpan di Drive)...",
                flush=True,
            )
        else:
            ckpt_mgr.clear_job(job_id)
            job_dir = ckpt_mgr.job_dir(job_id)
            start_idx = 0
            seg_num = 0

        if start_idx < 0 or start_idx >= total:
            start_idx = 0
            seg_num = 0

        ckpt_mgr.save_job_state(
            job_id,
            {
                "job_id": job_id,
                "last_completed_idx": start_idx,
                "total_frames": total,
                "next_seg_num": seg_num,
                "completed": False,
                "settings": settings_snapshot,
            },
        )

        seg_step = max(25, int(segment_frames))
        tracker = TemporalFaceTracker() if use_temporal_smooth else None
        if use_temporal_smooth and start_idx > 0:
            cap_warm = cv2.VideoCapture(str(target_path))
            warm_from = max(0, start_idx - 3)
            cap_warm.set(cv2.CAP_PROP_POS_FRAMES, warm_from)
            for _ in range(warm_from, start_idx):
                r_w, f_w = cap_warm.read()
                if r_w and f_w is not None:
                    tracker.update(detect_faces_angle_aware(app, f_w, enable_rotation=bool(use_angle_rot)))
            cap_warm.release()

        curr_start = start_idx
        while curr_start < total:
            curr_end = min(total, curr_start + seg_step)
            final_seg_file = job_dir / f"part_{seg_num:04d}.mp4"
            if final_seg_file.exists():
                final_seg_file.unlink()

            local_seg_fd, local_seg_path_str = tempfile.mkstemp(prefix=f"seg_{seg_num:04d}_", suffix=".mp4")
            os.close(local_seg_fd)
            local_seg_file = Path(local_seg_path_str)

            writer = cv2.VideoWriter(str(local_seg_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            cap = cv2.VideoCapture(str(target_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_start)

            processed_this_seg = 0
            last_done_idx = curr_start
            try:
                for idx in range(curr_start, curr_end):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    out_frame = frame.copy()
                    raw_faces = detect_faces_angle_aware(app, frame, enable_rotation=bool(use_angle_rot))
                    if tracker is not None:
                        faces = tracker.update(raw_faces)
                    else:
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
                    processed_this_seg += 1
                    last_done_idx = idx + 1

                    if processed_this_seg % 10 == 0 or last_done_idx == curr_end:
                        release_mem(clean_cuda=(processed_this_seg % 50 == 0))
                        if progress is not None:
                            progress((last_done_idx, total), desc=f"Merender frame {last_done_idx}/{total}")
                        print(
                            f"[CUDA RENDER] Frame {last_done_idx}/{total} | Segmen #{seg_num + 1} ({src_cnt} wajah sumber)",
                            flush=True,
                        )
            finally:
                cap.release()
                writer.release()

            ok_seg, _ = inspect_valid_mp4(local_seg_file)
            if processed_this_seg > 0 and ok_seg:
                drive_tmp_seg = job_dir / f"_tmp_part_{seg_num:04d}.mp4"
                shutil.copy2(str(local_seg_file), str(drive_tmp_seg))
                os.replace(str(drive_tmp_seg), str(final_seg_file))
                local_seg_file.unlink(missing_ok=True)

                ckpt_mgr.save_job_state(
                    job_id,
                    {
                        "job_id": job_id,
                        "last_completed_idx": last_done_idx,
                        "total_frames": total,
                        "next_seg_num": seg_num + 1,
                        "completed": False,
                        "settings": settings_snapshot,
                    },
                )
            else:
                local_seg_file.unlink(missing_ok=True)
                raise RuntimeError("Segmen gagal ditutup secara valid.")

            release_mem(clean_cuda=True)

            curr_start = last_done_idx
            seg_num += 1
            if processed_this_seg == 0:
                break

        ok, final_path_str = ckpt_mgr.finalize_video(job_id, target_path, out_name="hasil_swap.mp4")
        if ok:
            return final_path_str, f"Selesai ({total} frame, {src_cnt} wajah digabung) -> {final_path_str} (segmen temp dibersihkan)"

        out_path = out_dir / "hasil_swap.mp4"
        return str(out_path.resolve()), f"Selesai sebagian/gagal gabung ({final_path_str})"

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
    return str(img_out.resolve()), f"Selesai ({src_cnt} wajah digabung) -> {img_out}"


def launch_fast_ui(share: bool = True) -> None:
    import gradio as gr
    import onnxruntime as ort

    providers = ort.get_available_providers()
    gpu_status = "CUDA GPU AKTIF" if "CUDAExecutionProvider" in providers else "CPU MODE (Aktifkan T4 GPU!)"

    ckpt_mgr = get_checkpoint_mgr()
    saved_job = ckpt_mgr.find_latest_unfinished()
    saved_cfg = saved_job.get("settings", {}) if saved_job else {}

    def _valid_existing_file(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        try:
            if Path(p).is_file():
                return str(Path(p).resolve())
        except Exception:
            pass
        return None

    init_source_file = _valid_existing_file(saved_cfg.get("source_file"))
    init_target_file = _valid_existing_file(saved_cfg.get("target_file"))

    if not init_source_file or not init_target_file:
        fb_src, fb_tgt = ckpt_mgr.find_latest_persisted_inputs()
        if not init_source_file:
            init_source_file = _valid_existing_file(fb_src)
        if not init_target_file:
            init_target_file = _valid_existing_file(fb_tgt)

    def initial_resume_banner() -> str:
        latest = ckpt_mgr.find_latest_unfinished()
        if latest:
            jid = latest.get("job_id", "")
            _, valid_cnt = ckpt_mgr.get_clean_valid_segments(jid) if jid else ([], int(latest.get("last_completed_idx", 0)))
            tot_f = int(latest.get("total_frames", 0))
            return (
                f"Auto-Resume aktif: {valid_cnt}/{tot_f} frame valid tersimpan di Google Drive. "
                f"Klik 'Render Video Lengkap' untuk menyambung, atau klik 'Hapus History & File Hasil' untuk bersihkan semua."
            )
        if init_source_file or init_target_file:
            return "File sebelumnya ditemukan di Google Drive dan dimuat otomatis. Klik 'Render Video Lengkap' untuk melanjutkan."
        return "Siap. File input, segmen sementara & hasil akhir disimpan ke Drive; segmen terhapus otomatis saat selesai."

    def reset_all_history_and_outputs():
        ckpt_mgr.clear_all_workspace_files()
        _CACHE["source_key"] = None
        _CACHE["source_face"] = None
        _CACHE["source_count"] = 0
        return (
            None,
            None,
            [],
            None,
            "Semua history progres, segmen temp & file hasil di Google Drive telah dihapus.",
            "Siap untuk unggah file baru.",
        )

    with gr.Blocks(title="Fast FaceSwap CUDA UI + Faceset Studio") as demo:
        gr.Markdown(f"### Fast FaceSwap Pro UI & Faceset Studio — **{gpu_status}**")
        with gr.Row():
            resume_info_box = gr.Textbox(
                label="Status Auto-Resume & Segmen",
                value=initial_resume_banner(),
                interactive=False,
                scale=4,
            )
            clear_history_btn = gr.Button("Hapus History & File Hasil", variant="stop", scale=1)

        with gr.Row():
            source_in = gr.File(
                label="Sumber Wajah Aktif (.jpg / .png / .fsz)",
                type="filepath",
                value=init_source_file,
            )
            target_in = gr.File(
                label="Video atau Foto Target",
                type="filepath",
                value=init_target_file,
            )

        active_faces_gallery = gr.Gallery(
            label="Verifikasi Isi Wajah Aktif (Semua foto di bawah ini digabung)",
            columns=5,
            height="auto",
            object_fit="contain",
        )
        active_faces_info = gr.Textbox(label="Info Wajah Sumber", interactive=False)

        with gr.Accordion("Tools: Buat / Edit Faceset (.fsz) & Potong Wajah dari Video", open=False):
            gr.Markdown("Gabungkan beberapa foto sudut wajah atau potong langsung dari frame video menjadi satu file `.fsz`.")
            fsz_input_files = gr.File(
                label="1. Input Files (Pilih banyak foto sekaligus atau file .fsz lama)",
                file_count="multiple",
                type="filepath",
            )
            fsz_add_btn = gr.Button("Tambahkan Foto ke Daftar Faceset", variant="secondary")

            with gr.Row():
                fsz_video_in = gr.File(label="2. Cut Face from Video Frame (Opsional)", type="filepath")
                fsz_frame_num = gr.Number(value=0, precision=0, label="Nomor Frame Video")
            fsz_cut_btn = gr.Button("Use Faces from This Frame (Potong Wajah dari Frame)", variant="secondary")

            fsz_builder_gallery = gr.Gallery(
                label="Faces in this Faceset (Klik gambar untuk memilih sebelum dihapus)",
                columns=4,
                height="auto",
                object_fit="contain",
            )
            fsz_selected_idx = gr.State(value=None)
            fsz_name_in = gr.Textbox(value="faceset_multi_sudut", label="Nama File Output (.fsz)")

            with gr.Row():
                fsz_remove_btn = gr.Button("Remove Selected", variant="secondary")
                fsz_clear_btn = gr.Button("Clear All", variant="stop")
            fsz_create_btn = gr.Button("Create / Update Faceset & Pakai Langsung", variant="primary")
            fsz_status = gr.Textbox(label="Status Faceset Studio", interactive=False)
            fsz_download_box = gr.File(label="Download File .fsz Hasil")

        with gr.Row():
            swapper_in = gr.Dropdown(
                choices=list(SWAPPER_URLS.keys()),
                value=saved_cfg.get("swapper_model", "inswapper_128_fp16"),
                label="Face Swapper Model",
                info="inswapper_128_fp16 paling konsisten untuk sudut miring & ekspresi.",
            )
            swap_weight_in = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=float(saved_cfg.get("swap_weight", 1.0)),
                label="Face Swapper Weight",
                info="1.0 penuh agar hasil swap tidak transparan/tercampur.",
            )

        with gr.Row():
            enhancer_in = gr.Dropdown(
                choices=["restoreformer++", "gpen-512", "codeformer", "none"],
                value=saved_cfg.get("enhancer", "restoreformer++"),
                label="Face Enhancer",
                info="Mempertajam detail kulit, bibir & mata setelah swap.",
            )
            enhancer_blend_in = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=float(saved_cfg.get("enhancer_blend", 0.30)),
                label="Enhancer Blend",
                info="0.25–0.35 menjaga wajah & mata tetap natural.",
            )
            eye_expr_in = gr.Slider(
                minimum=0.0,
                maximum=0.8,
                step=0.05,
                value=float(saved_cfg.get("eye_expr_strength", 0.35)),
                label="Eye Expression Restorer",
                info="Mempertahankan arah bayangan kelopak & tatapan mata asli target.",
            )

        with gr.Row():
            use_angle_rot_in = gr.Checkbox(
                value=bool(saved_cfg.get("use_angle_rot", True)),
                label="Auto Rotasi Deteksi (90°/270°)",
                info="Wajib aktif untuk kepala rebah/miring agar titik mata tidak miss.",
            )
            use_parser_in = gr.Checkbox(
                value=bool(saved_cfg.get("use_face_parser", True)),
                label="Face Parser (BiSeNet)",
                info="Memisahkan poni rambut agar tidak tertimpa kulit wajah.",
            )
            use_xseg_in = gr.Checkbox(
                value=bool(saved_cfg.get("use_xseg", True)),
                label="XSeg Occlusion Mask",
                info="Menjaga tangan/rambut/kacamata di depan wajah tetap utuh.",
            )

        with gr.Row():
            protect_mouth_in = gr.Checkbox(
                value=bool(saved_cfg.get("protect_mouth", True)),
                label="Kunci Area Mulut",
                info="Mencegah bibir terpotong saat mulut terbuka/bicara.",
            )
            use_color_in = gr.Checkbox(
                value=bool(saved_cfg.get("use_color_match", True)),
                label="Color Match (LAB)",
                info="Menyamakan warna kulit dengan pencahayaan video.",
            )
            use_temporal_in = gr.Checkbox(
                value=bool(saved_cfg.get("use_temporal_smooth", False)),
                label="Temporal Smoothing & Anti-Miss (Opsional)",
                info="Default OFF. Centang untuk menghaluskan getaran & mencegah miss saat gerakan cepat.",
            )
            top_sharp_in = gr.Checkbox(
                value=bool(saved_cfg.get("top_sharp_fsz", False)),
                label="Filter Top-6 Foto Tajam (.fsz)",
                info="Default OFF. Centang jika .fsz berisi banyak sudut acak agar embedding lebih tajam.",
            )

        with gr.Row():
            face_mode_in = gr.Dropdown(
                choices=["largest", "all"],
                value=saved_cfg.get("face_mode", "largest"),
                label="Target Wajah",
                info="largest = wajah utama; all = semua wajah.",
            )
            det_size_in = gr.Slider(
                minimum=320,
                maximum=960,
                step=64,
                value=int(saved_cfg.get("det_size", 640)),
                label="Det Size (Presisi Landmark)",
                info="640 mengunci koordinat sudut mata & bibir dengan presisi.",
            )
            mask_blur_in = gr.Slider(
                minimum=5,
                maximum=35,
                step=2,
                value=int(saved_cfg.get("mask_blur", 13)),
                label="Mask Blur (Kehalusan Tepi)",
                info="11–15 menghaluskan batas pipi tanpa mengaburkan mata/mulut.",
            )

        with gr.Row():
            segment_frames_in = gr.Slider(
                minimum=25,
                maximum=400,
                step=25,
                value=int(saved_cfg.get("segment_frames", 50)),
                label="Ukuran Segmen Sementara (Frame)",
                info="50 frame (~1,5–2 detik) cepat tersimpan atomik ke Drive agar tahan disconnect.",
            )
            fresh_start_in = gr.Checkbox(
                value=False,
                label="Mulai dari Frame 0 (Hapus Progres Lama)",
                info="Centang jika ingin mengulang video baru dari awal.",
            )

        gr.Markdown("---")
        preview_img = gr.Image(label="Live Preview Hasil Per Frame", interactive=False)
        frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Geser Frame (Live Preview)")
        with gr.Row():
            preview_btn = gr.Button("Cek Preview Frame", variant="secondary")
            run_btn = gr.Button("Render Video Lengkap", variant="primary")

        preview_status = gr.Textbox(label="Status Render / Preview", interactive=False)
        output_file_box = gr.File(label="Download Hasil")

        clear_history_btn.click(
            fn=reset_all_history_and_outputs,
            outputs=[
                source_in,
                target_in,
                active_faces_gallery,
                output_file_box,
                resume_info_box,
                preview_status,
            ],
        )

        def on_source_changed(src_path):
            if not src_path:
                return [], "Belum ada sumber wajah dipilih."
            persisted = ckpt_mgr.persist_input_file(src_path, prefix="src") or src_path
            return inspect_source_gallery(persisted)

        source_in.change(
            fn=on_source_changed,
            inputs=[source_in],
            outputs=[active_faces_gallery, active_faces_info],
        )

        def on_gallery_select(evt: gr.SelectData):
            return evt.index

        fsz_builder_gallery.select(fn=on_gallery_select, outputs=[fsz_selected_idx])
        fsz_add_btn.click(
            fn=fsz_add_images,
            inputs=[fsz_input_files, fsz_builder_gallery, det_size_in],
            outputs=[fsz_builder_gallery, fsz_status],
        )
        fsz_cut_btn.click(
            fn=fsz_cut_from_video,
            inputs=[fsz_video_in, fsz_frame_num, fsz_builder_gallery, det_size_in],
            outputs=[fsz_builder_gallery, fsz_status],
        )
        fsz_remove_btn.click(
            fn=fsz_remove_selected,
            inputs=[fsz_builder_gallery, fsz_selected_idx],
            outputs=[fsz_builder_gallery, fsz_status],
        )
        fsz_clear_btn.click(
            fn=fsz_clear_all,
            outputs=[fsz_builder_gallery, fsz_status],
        )
        fsz_create_btn.click(
            fn=fsz_build_and_apply,
            inputs=[fsz_builder_gallery, fsz_name_in],
            outputs=[fsz_download_box, source_in, active_faces_gallery, fsz_status],
        )

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
            use_temporal_in,
            top_sharp_in,
            face_mode_in,
            det_size_in,
            mask_blur_in,
        ]

        def on_target_uploaded(tgt):
            if not tgt:
                return gr.update(minimum=0, maximum=100, value=0), "Belum ada target."
            persisted = ckpt_mgr.persist_input_file(tgt, prefix="tgt") or tgt
            total = get_video_frame_count(persisted)
            return gr.update(minimum=0, maximum=max(0, total - 1), value=0), f"Video siap ({total} frame) & tersimpan di Drive."

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
            use_temporal_in,
            top_sharp_fsz_in := top_sharp_in,
            face_mode_in,
            det_size_in,
            mask_blur_in,
            segment_frames_in,
            fresh_start_in,
        ]
        run_btn.click(fn=run_render_full, inputs=render_params, outputs=[output_file_box, preview_status])

    demo.queue(max_size=4).launch(
        share=share,
        debug=False,
        allowed_paths=["/content", "/content/drive", str(ckpt_mgr.root)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast FaceSwap CUDA UI & CLI")
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--no-gradio", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--drive-workspace", default=str(DEFAULT_DRIVE_WORKSPACE))
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--swapper", default="inswapper_128_fp16")
    parser.add_argument("--enhancer", default="restoreformer++")
    parser.add_argument("--enhancer-blend", type=float, default=0.30)
    parser.add_argument("--eye-expr", type=float, default=0.35)
    parser.add_argument("--temporal-smooth", action="store_true")
    parser.add_argument("--top-sharp-fsz", action="store_true")
    parser.add_argument("--segment-frames", type=int, default=50)
    parser.add_argument("--fresh-start", action="store_true")
    args = parser.parse_args()

    get_checkpoint_mgr(args.drive_workspace)

    if not args.skip_install:
        ensure_packages(use_gradio=not args.no_gradio)

    if args.no_gradio:
        out_path, msg = run_render_full(
            source_file=args.source,
            target_file=args.target,
            swapper_model=args.swapper,
            swap_weight=1.0,
            enhancer=args.enhancer,
            enhancer_blend=args.enhancer_blend,
            eye_expr_strength=args.eye_expr,
            use_angle_rot=True,
            use_face_parser=True,
            use_xseg=True,
            protect_mouth=True,
            use_color_match=True,
            use_temporal_smooth=args.temporal_smooth,
            top_sharp_fsz=args.top_sharp_fsz,
            face_mode="largest",
            det_size=640,
            mask_blur=13,
            segment_frames=args.segment_frames,
            fresh_start=args.fresh_start,
        )
        print(msg, flush=True)
        return

    launch_fast_ui(share=not args.no_share)


if __name__ == "__main__":
    main()
