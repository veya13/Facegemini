import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image

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


def compute_job_id(target_video: str, source_file: str) -> str:
    v_path = Path(target_video)
    s_path = Path(source_file) if source_file else None
    v_size = v_path.stat().st_size if v_path.exists() else 0
    s_size = s_path.stat().st_size if (s_path and s_path.exists()) else 0
    raw = f"{v_path.name}|{v_size}|{s_path.name if s_path else ''}|{s_size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class AutoResumeSegmentManager:
    """
    Menyimpan segmen sementara & state progres di Google Drive saat render berjalan,
    melanjutkan otomatis setelah reconnect tanpa ketik manual, dan menghapus
    seluruh file segmen sementara setelah hasil akhir berhasil dibuat.
    """

    def __init__(self, drive_root: str = "/content/drive/MyDrive/faceswap_workspace"):
        self.drive_root = Path(drive_root)
        self.final_dir = self.drive_root / "final_output"
        self.temp_root = self.drive_root / "temp_segments"
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def get_job_dir(self, job_id: str) -> Path:
        job_dir = self.temp_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_state_file(self, job_id: str) -> Path:
        return self.get_job_dir(job_id) / "render_state.json"

    def load_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        state_path = self.get_state_file(job_id)
        if not state_path.exists():
            return None
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def find_latest_incomplete_state(self) -> Optional[Dict[str, Any]]:
        if not self.temp_root.exists():
            return None
        candidates: List[Dict[str, Any]] = []
        for sub in self.temp_root.iterdir():
            if sub.is_dir():
                st_file = sub / "render_state.json"
                if st_file.exists():
                    try:
                        with open(st_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if not data.get("completed", False):
                            data["_job_dir"] = str(sub)
                            data["_updated_at"] = st_file.stat().st_mtime
                            candidates.append(data)
                    except Exception:
                        pass
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.get("_updated_at", 0), reverse=True)
        return candidates[0]

    def save_state(self, job_id: str, state: Dict[str, Any]):
        state_path = self.get_state_file(job_id)
        tmp_path = state_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)

    def cleanup_temp_job(self, job_id: str):
        job_dir = self.temp_root / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        free_memory(deep_cuda=True)

    def merge_segments_to_final(
        self,
        job_id: str,
        target_video: str,
        fps: float,
        final_filename: str = "hasil_swap.mp4",
    ) -> Tuple[bool, str]:
        job_dir = self.get_job_dir(job_id)
        seg_files = sorted(job_dir.glob("part_*.mp4"))
        valid_segs = [p for p in seg_files if p.exists() and p.stat().st_size > 0]
        if not valid_segs:
            return False, "Tidak ada file potongan segmen yang valid untuk digabungkan."

        concat_list = job_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in valid_segs:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        merged_video_only = job_dir / "_merged_no_audio.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(merged_video_only),
        ]
        res_concat = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res_concat.returncode != 0 or not merged_video_only.exists():
            return False, f"Gagal menggabungkan segmen: {res_concat.stderr[-300:]}"

        final_path = self.final_dir / final_filename
        cmd_mux = [
            "ffmpeg",
            "-y",
            "-i",
            str(merged_video_only),
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
        res_mux = subprocess.run(cmd_mux, capture_output=True, text=True)
        if res_mux.returncode != 0 or not final_path.exists() or final_path.stat().st_size == 0:
            # Fallback jika video target tidak punya track audio
            shutil.copy2(merged_video_only, final_path)

        if final_path.exists() and final_path.stat().st_size > 0:
            # Otomatis hapus seluruh file potongan sementara setelah hasil akhir jadi
            self.cleanup_temp_job(job_id)
            return True, str(final_path)

        return False, "File hasil akhir kosong atau gagal dibuat."


class FastFaceSwapEngine:
    def __init__(self, models_dir: str = "/content/fast_fs/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.face_app: Optional[FaceAnalysis] = None
        self.current_det_size: int = 640
        self.swapper = None

    def get_face_analyser(self, det_size: int = 640) -> FaceAnalysis:
        if self.face_app is None or self.current_det_size != det_size:
            self.face_app = None
            free_memory(deep_cuda=True)
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(self.models_dir),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.face_app = app
            self.current_det_size = det_size
        return self.face_app

    def get_swapper(self):
        if self.swapper is None:
            model_path = self.models_dir / "inswapper_128.onnx"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model {model_path} belum ada. Pastikan inswapper_128.onnx tersedia."
                )
            self.swapper = insightface.model_zoo.get_model(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        return self.swapper

    def swap_frame(
        self,
        frame_bgr: np.ndarray,
        source_img_bgr: np.ndarray,
        det_size: int = 640,
        blend_strength: float = 0.85,
    ) -> np.ndarray:
        analyser = self.get_face_analyser(det_size=det_size)
        swapper = self.get_swapper()

        src_faces = analyser.get(source_img_bgr)
        if not src_faces:
            return frame_bgr
        src_face = sorted(
            src_faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )[0]

        tgt_faces = analyser.get(frame_bgr)
        if not tgt_faces:
            return frame_bgr

        out_frame = frame_bgr.copy()
        for tgt_face in tgt_faces:
            swapped = swapper.get(out_frame, tgt_face, src_face, paste_back=True)
            if blend_strength < 0.999:
                out_frame = cv2.addWeighted(
                    swapped, blend_strength, out_frame, 1.0 - blend_strength, 0
                )
            else:
                out_frame = swapped
        return out_frame


def create_faceset_zip(images: List[Any], faceset_name: str, out_dir: str = "/content/fast_fs/output") -> Tuple[Optional[str], str]:
    if not images:
        return None, "Belum ada foto wajah yang ditambahkan ke daftar faceset."

    clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (faceset_name or "faceset_multi_sudut")).strip("_")
    if not clean_name:
        clean_name = "faceset_multi_sudut"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stage_dir = out_path / f"_tmp_{clean_name}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, item in enumerate(images):
        img_path = None
        if isinstance(item, tuple) and len(item) > 0:
            img_path = item[0]
        elif isinstance(item, str):
            img_path = item
        elif isinstance(item, dict) and "name" in item:
            img_path = item["name"]

        if img_path and Path(img_path).exists():
            ext = Path(img_path).suffix or ".jpg"
            shutil.copy2(img_path, stage_dir / f"angle_{idx+1:02d}{ext}")
            saved_count += 1

    if saved_count == 0:
        shutil.rmtree(stage_dir, ignore_errors=True)
        return None, "Gagal membaca file gambar wajah."

    fsz_path = out_path / f"{clean_name}.fsz"
    if fsz_path.exists():
        fsz_path.unlink()

    archive_base = str(out_path / clean_name)
    zip_created = shutil.make_archive(archive_base, "zip", root_dir=str(stage_dir))
    shutil.rmtree(stage_dir, ignore_errors=True)

    os.replace(zip_created, str(fsz_path))
    return str(fsz_path), f"Faceset '{fsz_path.name}' ({saved_count} sudut) berhasil dibuat di {fsz_path}."


def build_ui(
    drive_workspace: str = "/content/drive/MyDrive/faceswap_workspace",
    models_dir: str = "/content/fast_fs/models",
):
    seg_mgr = AutoResumeSegmentManager(drive_root=drive_workspace)
    engine = FastFaceSwapEngine(models_dir=models_dir)

    saved_state = seg_mgr.find_latest_incomplete_state()
    init_settings = saved_state.get("settings", {}) if saved_state else {}

    def get_init_status_text() -> str:
        if saved_state:
            last_f = int(saved_state.get("last_completed_frame", -1))
            total_f = int(saved_state.get("total_frames", 0))
            next_f = last_f + 1
            return (
                f"Terdeteksi progres tersimpan di Google Drive: selesai sampai frame {last_f + 1}/{total_f}. "
                f"Klik 'Render / Lanjutkan Otomatis' untuk menyambung dari frame {next_f + 1} tanpa setting ulang."
            )
        return "Siap. Segmen sementara disimpan di Google Drive dan otomatis dihapus setelah hasil akhir jadi."

    def run_auto_render(
        target_video: str,
        source_file: str,
        segment_size: int,
        det_size: int,
        blend_strength: float,
        final_name: str,
        force_restart: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not target_video or not Path(target_video).exists():
            yield None, "Pilih file video target terlebih dahulu."
            return
        if not source_file or not Path(source_file).exists():
            yield None, "Pilih foto sumber wajah terlebih dahulu."
            return

        src_img = cv2.imread(source_file)
        if src_img is None:
            yield None, "Gagal membaca gambar sumber wajah."
            return

        total_frames, fps, width, height = get_video_info(target_video)
        if total_frames <= 0:
            yield None, "Tidak dapat membaca jumlah frame video target."
            return

        job_id = compute_job_id(target_video, source_file)
        job_dir = seg_mgr.get_job_dir(job_id)

        current_settings = {
            "target_video": target_video,
            "source_file": source_file,
            "segment_size": int(segment_size),
            "det_size": int(det_size),
            "blend_strength": float(blend_strength),
            "final_name": final_name or "hasil_swap.mp4",
        }

        if force_restart:
            seg_mgr.cleanup_temp_job(job_id)
            job_dir = seg_mgr.get_job_dir(job_id)
            state = None
        else:
            state = seg_mgr.load_state(job_id)

        if state and not state.get("completed", False):
            start_frame = int(state.get("last_completed_frame", -1)) + 1
            seg_index = int(state.get("next_seg_index", 0))
        else:
            start_frame = 0
            seg_index = 0

        if start_frame < 0 or start_frame >= total_frames:
            start_frame = 0
            seg_index = 0

        seg_size = max(50, int(segment_size))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while start_frame < total_frames:
            end_frame = min(total_frames, start_frame + seg_size)
            seg_path = job_dir / f"part_{seg_index:04d}.mp4"
            if seg_path.exists():
                seg_path.unlink()

            writer = cv2.VideoWriter(str(seg_path), fourcc, fps, (width, height))
            cap = cv2.VideoCapture(target_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            processed_in_seg = 0
            for f_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                out_frame = engine.swap_frame(
                    frame_bgr=frame,
                    source_img_bgr=src_img,
                    det_size=int(det_size),
                    blend_strength=float(blend_strength),
                )
                writer.write(out_frame)
                processed_in_seg += 1

                if processed_in_seg % 15 == 0 or f_idx == end_frame - 1:
                    seg_mgr.save_state(
                        job_id,
                        {
                            "job_id": job_id,
                            "last_completed_frame": f_idx,
                            "total_frames": total_frames,
                            "next_seg_index": seg_index + (1 if f_idx == end_frame - 1 else 0),
                            "completed": False,
                            "settings": current_settings,
                        },
                    )
                    free_memory(deep_cuda=(processed_in_seg % 45 == 0))
                    pct = (f_idx + 1) / max(1, total_frames)
                    progress(
                        pct,
                        desc=f"Segmen {seg_index + 1} | Frame {f_idx + 1}/{total_frames}",
                    )
                    yield None, (
                        f"Memproses frame {f_idx + 1}/{total_frames} "
                        f"(Segmen #{seg_index + 1} tersimpan sementara di Drive)."
                    )

            cap.release()
            writer.release()
            free_memory(deep_cuda=True)

            start_frame = end_frame
            seg_index += 1

        clean_final_name = (final_name or "hasil_swap.mp4").strip()
        if not clean_final_name.lower().endswith(".mp4"):
            clean_final_name += ".mp4"

        ok, final_or_err = seg_mgr.merge_segments_to_final(
            job_id=job_id,
            target_video=target_video,
            fps=fps,
            final_filename=clean_final_name,
        )
        if ok:
            yield final_or_err, (
                f"Selesai! Hasil akhir tersimpan di: {final_or_err}. "
                f"Seluruh file potongan sementara sudah otomatis dihapus."
            )
        else:
            yield None, f"Proses selesai merender segmen, tetapi penggabungan gagal: {final_or_err}"

    with gr.Blocks(title="Auto-Resume Segment FaceSwap") as demo:
        gr.Markdown("## Render Video Otomatis Per Segmen (Auto-Resume & Auto-Cleanup)")
        status_box = gr.Textbox(
            label="Status Progres Otomatis",
            value=get_init_status_text(),
            interactive=False,
        )

        with gr.Row():
            target_video_in = gr.Textbox(
                label="Path Video Target (1080p)",
                value=init_settings.get("target_video", ""),
                placeholder="/content/drive/MyDrive/video_target.mp4",
            )
            source_file_in = gr.Textbox(
                label="Path Foto / Faceset Sumber",
                value=init_settings.get("source_file", ""),
                placeholder="/content/drive/MyDrive/wajah.jpg",
            )

        with gr.Row():
            segment_size_in = gr.Slider(
                label="Ukuran Segmen (Frame per Potongan Sementara)",
                minimum=100,
                maximum=600,
                step=50,
                value=int(init_settings.get("segment_size", 250)),
            )
            det_size_in = gr.Dropdown(
                label="Detection Size",
                choices=[512, 640],
                value=int(init_settings.get("det_size", 512)),
            )
            blend_in = gr.Slider(
                label="Blend Strength",
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=float(init_settings.get("blend_strength", 0.85)),
            )

        with gr.Row():
            final_name_in = gr.Textbox(
                label="Nama File Hasil Akhir di Google Drive",
                value=init_settings.get("final_name", "hasil_swap_final.mp4"),
            )
            force_restart_cb = gr.Checkbox(
                label="Mulai dari Frame 0 (Hapus Progres Lama)",
                value=False,
            )

        run_btn = gr.Button("Render / Lanjutkan Otomatis", variant="primary")
        video_out = gr.Video(label="Hasil Video Akhir")

        run_btn.click(
            fn=run_auto_render,
            inputs=[
                target_video_in,
                source_file_in,
                segment_size_in,
                det_size_in,
                blend_in,
                final_name_in,
                force_restart_cb,
            ],
            outputs=[video_out, status_box],
        )

        with gr.Accordion("Faceset Studio (.fsz)", open=False):
            fsz_gallery = gr.Gallery(label="Potongan Sudut Wajah", type="filepath")
            fsz_name = gr.Textbox(label="Nama Faceset", value="faceset_multi_sudut")
            fsz_btn = gr.Button("Buat File .fsz")
            fsz_out_file = gr.File(label="File .fsz Hasil")
            fsz_msg = gr.Textbox(label="Status Faceset", interactive=False)

            def on_make_fsz(imgs, fname):
                fpath, msg = create_faceset_zip(imgs, fname)
                return fpath, (fpath or ""), msg

            fsz_btn.click(
                fn=on_make_fsz,
                inputs=[fsz_gallery, fsz_name],
                outputs=[fsz_out_file, source_file_in, fsz_msg],
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-workspace", default="/content/drive/MyDrive/faceswap_workspace")
    parser.add_argument("--models-dir", default="/content/fast_fs/models")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()

    app = build_ui(
        drive_workspace=args.drive_workspace,
        models_dir=args.models_dir,
    )
    app.launch(share=args.share, server_port=args.port)
