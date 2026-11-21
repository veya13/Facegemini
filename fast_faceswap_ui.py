import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None


def free_memory(deep_cuda: bool = False):
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            if deep_cuda:
                torch.cuda.empty_cache()
        except Exception:
            pass


def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 30.0, 1920, 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    return total_frames, fps, width, height


def read_frame_at(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0:
        frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def compute_job_id(target_video: str, source_path_str: str) -> str:
    vp = Path(target_video)
    sp = Path(source_path_str) if source_path_str else None
    v_size = vp.stat().st_size if vp.exists() else 0
    s_size = sp.stat().st_size if (sp and sp.exists()) else 0
    raw = f"{vp.name}|{v_size}|{sp.name if sp else ''}|{s_size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def load_source_images(source_input: Any, work_dir: Path) -> List[np.ndarray]:
    """Membaca foto tunggal atau kumpulan foto di dalam file .fsz (zip)."""
    if not source_input:
        return []
    path_str = source_input[0] if isinstance(source_input, (list, tuple)) else str(source_input)
    p = Path(path_str)
    if not p.exists():
        return []

    imgs_bgr: List[np.ndarray] = []
    if p.suffix.lower() == ".fsz":
        extract_dir = work_dir / f"fsz_extract_{p.stem}"
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(str(p), "r") as zf:
                zf.extractall(str(extract_dir))
            for f in sorted(extract_dir.rglob("*")):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    im = cv2.imread(str(f))
                    if im is not None:
                        imgs_bgr.append(im)
        except Exception:
            pass
    else:
        im = cv2.imread(str(p))
        if im is not None:
            imgs_bgr.append(im)
    return imgs_bgr


def match_color_lab(swapped_bgr: np.ndarray, target_bgr: np.ndarray, strength: float = 0.75) -> np.ndarray:
    if strength <= 0.001:
        return swapped_bgr
    try:
        src_lab = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        dst_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        for c in range(3):
            s_mean, s_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
            d_mean, d_std = dst_lab[:, :, c].mean(), dst_lab[:, :, c].std() + 1e-6
            matched = (src_lab[:, :, c] - s_mean) * (d_std / s_std) + d_mean
            src_lab[:, :, c] = np.clip(matched, 0, 255)

        matched_bgr = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(matched_bgr, strength, swapped_bgr, 1.0 - strength, 0)
    except Exception:
        return swapped_bgr


class AutoResumeSegmentManager:
    def __init__(self, drive_root: str = "/content/drive/MyDrive/faceswap_workspace"):
        self.drive_root = Path(drive_root)
        self.final_dir = self.drive_root / "final_output"
        self.temp_root = self.drive_root / "temp_segments"
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def get_job_dir(self, job_id: str) -> Path:
        d = self.temp_root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_state_file(self, job_id: str) -> Path:
        return self.get_job_dir(job_id) / "render_state.json"

    def load_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        sp = self.get_state_file(job_id)
        if not sp.exists():
            return None
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def find_latest_incomplete_state(self) -> Optional[Dict[str, Any]]:
        if not self.temp_root.exists():
            return None
        items: List[Dict[str, Any]] = []
        for sub in self.temp_root.iterdir():
            if sub.is_dir():
                sp = sub / "render_state.json"
                if sp.exists():
                    try:
                        with open(sp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if not data.get("completed", False):
                            data["_mtime"] = sp.stat().st_mtime
                            items.append(data)
                    except Exception:
                        pass
        if not items:
            return None
        items.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
        return items[0]

    def save_state(self, job_id: str, state: Dict[str, Any]):
        sp = self.get_state_file(job_id)
        tmp = sp.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, sp)

    def cleanup_temp_job(self, job_id: str):
        d = self.temp_root / job_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        free_memory(deep_cuda=True)

    def merge_segments_to_final(
        self,
        job_id: str,
        target_video: str,
        final_filename: str = "hasil_swap_final.mp4",
    ) -> Tuple[bool, str]:
        job_dir = self.get_job_dir(job_id)
        segs = [p for p in sorted(job_dir.glob("part_*.mp4")) if p.exists() and p.stat().st_size > 0]
        if not segs:
            return False, "Tidak ada file segmen yang valid."

        concat_file = job_dir / "concat.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in segs:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        merged_no_audio = job_dir / "_merged_video.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(merged_no_audio),
        ]
        r1 = subprocess.run(cmd_concat, capture_output=True, text=True)
        if r1.returncode != 0 or not merged_no_audio.exists():
            return False, f"Gagal menggabungkan segmen: {r1.stderr[-250:]}"

        final_path = self.final_dir / final_filename
        cmd_mux = [
            "ffmpeg",
            "-y",
            "-i",
            str(merged_no_audio),
            "-i",
            target_video,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_path),
        ]
        r2 = subprocess.run(cmd_mux, capture_output=True, text=True)
        if r2.returncode != 0 or not final_path.exists() or final_path.stat().st_size == 0:
            shutil.copy2(merged_no_audio, final_path)

        if final_path.exists() and final_path.stat().st_size > 0:
            self.cleanup_temp_job(job_id)
            return True, str(final_path)

        return False, "File hasil akhir gagal dibuat."


class FaceSwapPipeline:
    def __init__(self, models_dir: str = "/content/fast_fs/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.face_app: Optional[FaceAnalysis] = None
        self.current_det_size: int = 512
        self.swapper = None
        self.enhancer_session = None
        self.current_enhancer_name: str = "none"

    def _providers(self) -> List[str]:
        if ort is not None:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get_analyser(self, det_size: int = 512) -> FaceAnalysis:
        if self.face_app is None or self.current_det_size != det_size:
            self.face_app = None
            free_memory(deep_cuda=True)
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(self.models_dir),
                providers=self._providers(),
            )
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.face_app = app
            self.current_det_size = det_size
        return self.face_app

    def get_swapper(self):
        if self.swapper is None:
            model_path = self.models_dir / "inswapper_128.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_path} tidak ditemukan.")
            self.swapper = insightface.model_zoo.get_model(
                str(model_path), providers=self._providers()
            )
        return self.swapper

    def get_enhancer(self, enhancer_name: str):
        if not enhancer_name or enhancer_name.lower() == "none":
            self.enhancer_session = None
            self.current_enhancer_name = "none"
            return None

        if self.current_enhancer_name == enhancer_name and self.enhancer_session is not None:
            return self.enhancer_session

        file_map = {
            "gpen-512": "GPEN-BFR-512.onnx",
            "restoreformer++": "restoreformer_plus_plus.onnx",
            "gfpgan-1.4": "GFPGANv1.4.onnx",
        }
        fname = file_map.get(enhancer_name.lower())
        if not fname:
            return None
        mpath = self.models_dir / fname
        if not mpath.exists() or ort is None:
            return None

        self.enhancer_session = None
        free_memory(deep_cuda=True)
        self.enhancer_session = ort.InferenceSession(str(mpath), providers=self._providers())
        self.current_enhancer_name = enhancer_name
        return self.enhancer_session

    def _enhance_face_region(self, face_bgr: np.ndarray, session) -> np.ndarray:
        if session is None or face_bgr.size == 0:
            return face_bgr
        try:
            inp = session.get_inputs()[0]
            shape = inp.shape
            h_in = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 512
            w_in = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 512

            orig_h, orig_w = face_bgr.shape[:2]
            resized = cv2.resize(face_bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - 0.5) / 0.5
            chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

            out = session.run(None, {inp.name: chw})[0][0]
            out_hwc = np.transpose(out, (1, 2, 0))
            out_hwc = np.clip((out_hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_hwc, cv2.COLOR_RGB2BGR)
            return cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        except Exception:
            return face_bgr

    def _select_source_face(self, analyser: FaceAnalysis, src_images: List[np.ndarray], tgt_face):
        candidates = []
        for im in src_images:
            faces = analyser.get(im)
            if faces:
                best = sorted(
                    faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True,
                )[0]
                candidates.append(best)
        if not candidates:
            return None
        if len(candidates) == 1 or getattr(tgt_face, "normed_embedding", None) is None:
            return candidates[0]

        # Pilih referensi yang paling selaras secara pose/embedding jika tersedia
        best_face = candidates[0]
        best_score = -1.0
        t_emb = tgt_face.normed_embedding
        for c in candidates:
            if getattr(c, "normed_embedding", None) is not None:
                score = float(np.dot(c.normed_embedding, t_emb))
                if score > best_score:
                    best_score = score
                    best_face = c
        return best_face

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        src_images: List[np.ndarray],
        det_size: int = 512,
        blend_strength: float = 0.85,
        color_match_strength: float = 0.75,
        enhancer_name: str = "none",
        enhance_strength: float = 0.7,
        face_mode: str = "largest",
    ) -> np.ndarray:
        if not src_images:
            return frame_bgr

        analyser = self.get_analyser(det_size=det_size)
        swapper = self.get_swapper()
        enh_sess = self.get_enhancer(enhancer_name)

        tgt_faces = analyser.get(frame_bgr)
        if not tgt_faces:
            return frame_bgr

        if face_mode == "largest":
            targets = [
                sorted(
                    tgt_faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True,
                )[0]
            ]
        else:
            targets = tgt_faces

        out = frame_bgr.copy()
        for tf in targets:
            sf = self._select_source_face(analyser, src_images, tf)
            if sf is None:
                continue

            swapped = swapper.get(out, tf, sf, paste_back=True)

            x1, y1, x2, y2 = [int(v) for v in tf.bbox]
            h, w = out.shape[:2]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            if cx2 > cx1 and cy2 > cy1:
                crop_swap = swapped[cy1:cy2, cx1:cx2]
                crop_orig = out[cy1:cy2, cx1:cx2]

                crop_swap = match_color_lab(crop_swap, crop_orig, strength=color_match_strength)

                if enh_sess is not None and enhance_strength > 0.01:
                    enh_crop = self._enhance_face_region(crop_swap, enh_sess)
                    crop_swap = cv2.addWeighted(
                        enh_crop, enhance_strength, crop_swap, 1.0 - enhance_strength, 0
                    )

                if blend_strength < 0.999:
                    blended_crop = cv2.addWeighted(
                        crop_swap, blend_strength, crop_orig, 1.0 - blend_strength, 0
                    )
                else:
                    blended_crop = crop_swap

                out[cy1:cy2, cx1:cx2] = blended_crop

        return out


def build_faceset_file(gallery_files: List[Any], name_str: str, out_dir: Path) -> Tuple[Optional[str], str]:
    if not gallery_files:
        return None, "Tambahkan minimal 1 foto sudut wajah ke galeri terlebih dahulu."
    clean = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (name_str or "faceset_multi_sudut")).strip("_") or "faceset_multi_sudut"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / f"_tmp_{clean}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, item in enumerate(gallery_files):
        p = item[0] if isinstance(item, (list, tuple)) else (item.get("name") if isinstance(item, dict) else str(item))
        if p and Path(p).exists():
            ext = Path(p).suffix or ".jpg"
            shutil.copy2(p, tmp_dir / f"sudut_{idx+1:02d}{ext}")
            count += 1

    if count == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, "Tidak ada gambar valid yang ditemukan."

    fsz_target = out_dir / f"{clean}.fsz"
    if fsz_target.exists():
        fsz_target.unlink()

    zip_path = shutil.make_archive(str(out_dir / clean), "zip", root_dir=str(tmp_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.replace(zip_path, str(fsz_target))
    return str(fsz_target), f"Faceset '{fsz_target.name}' ({count} sudut) tersimpan di {fsz_target}."


def build_ui(
    drive_workspace: str = "/content/drive/MyDrive/faceswap_workspace",
    models_dir: str = "/content/fast_fs/models",
):
    seg_mgr = AutoResumeSegmentManager(drive_root=drive_workspace)
    pipeline = FaceSwapPipeline(models_dir=models_dir)
    work_tmp = Path("/content/fast_fs/tmp_work")
    work_tmp.mkdir(parents=True, exist_ok=True)

    latest_state = seg_mgr.find_latest_incomplete_state()
    init_cfg = latest_state.get("settings", {}) if latest_state else {}

    def initial_status() -> str:
        if latest_state:
            done_f = int(latest_state.get("last_completed_frame", -1)) + 1
            tot_f = int(latest_state.get("total_frames", 0))
            return (
                f"Progres sebelumnya terdeteksi di Google Drive ({done_f}/{tot_f} frame selesai). "
                f"Setelan dimuat otomatis; klik 'Render / Lanjutkan Otomatis' untuk menyambung."
            )
        return "Siap. Gunakan Preview untuk cek frame, lalu Render Otomatis per segmen ke Google Drive."

    def on_load_video(video_path: str):
        if not video_path or not Path(video_path).exists():
            return gr.update(maximum=1, value=0), "Video belum dipilih atau path tidak ditemukan."
        total_f, fps, w, h = get_video_info(video_path)
        max_idx = max(0, total_f - 1)
        return gr.update(maximum=max_idx, value=0), f"Video siap: {w}x{h}, {fps:.2f} FPS, total {total_f} frame."

    def on_preview_frame(
        video_path: str,
        source_path: str,
        frame_idx: int,
        det_size: int,
        blend_val: float,
        color_val: float,
        enh_name: str,
        enh_val: float,
        face_mode: str,
    ):
        if not video_path or not Path(video_path).exists():
            return None, "Pilih video target terlebih dahulu."
        if not source_path or not Path(source_path).exists():
            return None, "Pilih foto atau file .fsz sumber wajah terlebih dahulu."

        frame_bgr = read_frame_at(video_path, int(frame_idx))
        if frame_bgr is None:
            return None, f"Gagal membaca frame ke-{int(frame_idx)}."

        src_imgs = load_source_images(source_path, work_tmp)
        if not src_imgs:
            return None, "Gagal memuat gambar sumber wajah."

        res_bgr = pipeline.process_frame(
            frame_bgr=frame_bgr,
            src_images=src_imgs,
            det_size=int(det_size),
            blend_strength=float(blend_val),
            color_match_strength=float(color_val),
            enhancer_name=enh_name,
            enhance_strength=float(enh_val),
            face_mode=face_mode,
        )
        rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), f"Preview frame #{int(frame_idx)} berhasil."

    def on_render_auto(
        video_path: str,
        source_path: str,
        segment_size: int,
        det_size: int,
        blend_val: float,
        color_val: float,
        enh_name: str,
        enh_val: float,
        face_mode: str,
        final_name: str,
        force_restart: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not video_path or not Path(video_path).exists():
            yield None, "Path video target belum valid."
            return
        if not source_path or not Path(source_path).exists():
            yield None, "Path foto / .fsz sumber wajah belum valid."
            return

        src_imgs = load_source_images(source_path, work_tmp)
        if not src_imgs:
            yield None, "Tidak dapat membaca gambar wajah sumber."
            return

        total_frames, fps, width, height = get_video_info(video_path)
        if total_frames <= 0:
            yield None, "Tidak dapat membaca frame video target."
            return

        job_id = compute_job_id(video_path, source_path)
        job_dir = seg_mgr.get_job_dir(job_id)

        settings_snapshot = {
            "target_video": video_path,
            "source_file": source_path,
            "segment_size": int(segment_size),
            "det_size": int(det_size),
            "blend_strength": float(blend_val),
            "color_match_strength": float(color_val),
            "enhancer_name": enh_name,
            "enhance_strength": float(enh_val),
            "face_mode": face_mode,
            "final_name": final_name or "hasil_swap_final.mp4",
        }

        if force_restart:
            seg_mgr.cleanup_temp_job(job_id)
            job_dir = seg_mgr.get_job_dir(job_id)
            state = None
        else:
            state = seg_mgr.load_state(job_id)

        if state and not state.get("completed", False):
            start_frame = int(state.get("last_completed_frame", -1)) + 1
            seg_idx = int(state.get("next_seg_index", 0))
        else:
            start_frame = 0
            seg_idx = 0

        if start_frame < 0 or start_frame >= total_frames:
            start_frame = 0
            seg_idx = 0

        seg_len = max(50, int(segment_size))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while start_frame < total_frames:
            end_frame = min(total_frames, start_frame + seg_len)
            seg_file = job_dir / f"part_{seg_idx:04d}.mp4"
            if seg_file.exists():
                seg_file.unlink()

            writer = cv2.VideoWriter(str(seg_file), fourcc, fps, (width, height))
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            step_count = 0
            for f_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                out_frame = pipeline.process_frame(
                    frame_bgr=frame,
                    src_images=src_imgs,
                    det_size=int(det_size),
                    blend_strength=float(blend_val),
                    color_match_strength=float(color_val),
                    enhancer_name=enh_name,
                    enhance_strength=float(enh_val),
                    face_mode=face_mode,
                )
                writer.write(out_frame)
                step_count += 1

                if step_count % 15 == 0 or f_idx == end_frame - 1:
                    seg_mgr.save_state(
                        job_id,
                        {
                            "job_id": job_id,
                            "last_completed_frame": f_idx,
                            "total_frames": total_frames,
                            "next_seg_index": seg_idx + (1 if f_idx == end_frame - 1 else 0),
                            "completed": False,
                            "settings": settings_snapshot,
                        },
                    )
                    free_memory(deep_cuda=(step_count % 45 == 0))
                    progress((f_idx + 1) / max(1, total_frames), desc=f"Frame {f_idx + 1}/{total_frames}")
                    yield None, (
                        f"Rendering frame {f_idx + 1}/{total_frames} "
                        f"(Segmen #{seg_idx + 1} tersimpan sementara di Drive)."
                    )

            cap.release()
            writer.release()
            free_memory(deep_cuda=True)

            start_frame = end_frame
            seg_idx += 1

        clean_name = (final_name or "hasil_swap_final.mp4").strip()
        if not clean_name.lower().endswith(".mp4"):
            clean_name += ".mp4"

        ok, res_path_or_err = seg_mgr.merge_segments_to_final(
            job_id=job_id,
            target_video=video_path,
            final_filename=clean_name,
        )
        if ok:
            yield res_path_or_err, (
                f"Selesai! Hasil akhir tersimpan di {res_path_or_err}. "
                f"File segmen sementara sudah otomatis dihapus."
            )
        else:
            yield None, f"Penggabungan akhir bermasalah: {res_path_or_err}"

    with gr.Blocks(title="FaceSwap Studio + Auto-Resume") as demo:
        gr.Markdown("## FaceSwap Studio (Preview + Enhancer + Faceset .fsz + Auto-Resume)")
        status_box = gr.Textbox(label="Status", value=initial_status(), interactive=False)

        with gr.Row():
            target_video_in = gr.Textbox(
                label="Path Video Target",
                value=init_cfg.get("target_video", ""),
                placeholder="/content/drive/MyDrive/video.mp4",
            )
            source_file_in = gr.Textbox(
                label="Path Foto / File .fsz Sumber",
                value=init_cfg.get("source_file", ""),
                placeholder="/content/drive/MyDrive/wajah.jpg atau faceset_multi_sudut.fsz",
            )

        load_vid_btn = gr.Button("Muat Info Video & Slider Frame")

        with gr.Row():
            frame_slider = gr.Slider(label="Pilih Frame Preview", minimum=0, maximum=100, step=1, value=0)
            preview_btn = gr.Button("Cek Preview Frame", variant="secondary")

        preview_img = gr.Image(label="Hasil Preview Frame", type="pil")

        with gr.Row():
            det_size_in = gr.Dropdown(label="Detection Size", choices=[512, 640], value=int(init_cfg.get("det_size", 512)))
            enhancer_in = gr.Dropdown(
                label="Face Enhancer",
                choices=["none", "gpen-512", "restoreformer++", "gfpgan-1.4"],
                value=init_cfg.get("enhancer_name", "none"),
            )
            face_mode_in = gr.Dropdown(
                label="Mode Target Wajah",
                choices=["largest", "all"],
                value=init_cfg.get("face_mode", "largest"),
            )

        with gr.Row():
            blend_in = gr.Slider(label="Blend Strength", minimum=0.1, maximum=1.0, step=0.05, value=float(init_cfg.get("blend_strength", 0.85)))
            color_in = gr.Slider(label="Color Match Strength", minimum=0.0, maximum=1.0, step=0.05, value=float(init_cfg.get("color_match_strength", 0.75)))
            enh_strength_in = gr.Slider(label="Enhancer Strength", minimum=0.0, maximum=1.0, step=0.05, value=float(init_cfg.get("enhance_strength", 0.7)))

        with gr.Row():
            seg_size_in = gr.Slider(label="Ukuran Segmen Sementara (Frame)", minimum=100, maximum=600, step=50, value=int(init_cfg.get("segment_size", 250)))
            final_name_in = gr.Textbox(label="Nama File Hasil Akhir", value=init_cfg.get("final_name", "hasil_swap_final.mp4"))
            restart_cb = gr.Checkbox(label="Mulai Baru dari Frame 0", value=False)

        render_btn = gr.Button("Render / Lanjutkan Otomatis", variant="primary")
        video_out = gr.Video(label="Video Hasil Akhir")

        load_vid_btn.click(fn=on_load_video, inputs=[target_video_in], outputs=[frame_slider, status_box])
        preview_btn.click(
            fn=on_preview_frame,
            inputs=[target_video_in, source_file_in, frame_slider, det_size_in, blend_in, color_in, enhancer_in, enh_strength_in, face_mode_in],
            outputs=[preview_img, status_box],
        )
        render_btn.click(
            fn=on_render_auto,
            inputs=[
                target_video_in,
                source_file_in,
                seg_size_in,
                det_size_in,
                blend_in,
                color_in,
                enhancer_in,
                enh_strength_in,
                face_mode_in,
                final_name_in,
                restart_cb,
            ],
            outputs=[video_out, status_box],
        )

        with gr.Accordion("Faceset Studio (.fsz)", open=False):
            fsz_gallery = gr.Gallery(label="Sudut Wajah", type="filepath")
            fsz_name = gr.Textbox(label="Nama Faceset (.fsz)", value="faceset_multi_sudut")
            fsz_btn = gr.Button("Buat & Pakai Faceset .fsz")
            fsz_file = gr.File(label="Download .fsz")
            fsz_status = gr.Textbox(label="Status Faceset", interactive=False)

            def on_fsz(imgs, fname):
                fpath, msg = build_faceset_file(imgs, fname, Path("/content/fast_fs/output"))
                return fpath, (fpath or ""), msg

            fsz_btn.click(fn=on_fsz, inputs=[fsz_gallery, fsz_name], outputs=[fsz_file, source_file_in, fsz_status])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-workspace", default="/content/drive/MyDrive/faceswap_workspace")
    parser.add_argument("--models-dir", default="/content/fast_fs/models")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()

    ui = build_ui(drive_workspace=args.drive_workspace, models_dir=args.models_dir)
    ui.launch(share=args.share, server_port=args.port)
