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

MODEL_URLS = {
    "inswapper_128.onnx": [
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
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
    "swapper": None,
    "enhancer_name": None,
    "enhancer_obj": None,
    "source_key": None,
    "source_face": None,
}


def run_cmd(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def ensure_packages(use_gradio: bool) -> None:
    pkgs = ["insightface==0.7.3", "onnxruntime-gpu", "opencv-python"]
    if use_gradio:
        pkgs.extend(["gradio==5.13.0", "pydantic==2.10.6"])
    print("[SETUP] Memastikan paket inti & CUDA ONNX terpasang...")
    run_cmd(f"{sys.executable} -m pip install -q " + " ".join(pkgs))


def download_model(filename: str, min_bytes: int = 10_000_000) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    urls = MODEL_URLS.get(filename, [])
    for url in urls:
        print(f"[UNDUH] {url}")
        rc = run_cmd(f'wget -q --show-progress "{url}" -O "{dest}"')
        if rc == 0 and dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"[SUKSES] Tersimpan: {dest}")
            return dest
        if dest.exists():
            dest.unlink()

    raise RuntimeError(f"Gagal mengunduh model {filename}.")


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


def get_swapper():
    if _CACHE["swapper"] is not None:
        return _CACHE["swapper"]

    import insightface

    swapper_path = download_model("inswapper_128.onnx", min_bytes=50_000_000)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    swapper = insightface.model_zoo.get_model(str(swapper_path), providers=providers)
    _CACHE["swapper"] = swapper
    return swapper


def get_enhancer(enhancer_choice: str) -> Optional[OnnxFaceEnhancer]:
    if enhancer_choice == "none":
        return None
    if _CACHE["enhancer_name"] == enhancer_choice and _CACHE["enhancer_obj"] is not None:
        return _CACHE["enhancer_obj"]

    mapping = {
        "restoreformer++": ("restoreformer_plus_plus.onnx", 100_000_000),
        "gpen-512": ("GPEN-BFR-512.onnx", 100_000_000),
        "codeformer": ("CodeFormerv0.1.onnx", 100_000_000),
    }
    fname, min_b = mapping[enhancer_choice]
    mpath = download_model(fname, min_bytes=min_b)
    obj = OnnxFaceEnhancer(mpath, enhancer_choice)
    _CACHE["enhancer_name"] = enhancer_choice
    _CACHE["enhancer_obj"] = obj
    return obj


def _make_merged_face_obj(base_face, mean_emb: np.ndarray):
    """Membungkus embedding rata-rata tanpa menabrak read-only property normed_embedding pada insightface.Face."""
    norm = float(np.linalg.norm(mean_emb))
    normed = (mean_emb / norm) if norm > 0 else mean_emb
    return SimpleNamespace(
        bbox=getattr(base_face, "bbox", None),
        kps=getattr(base_face, "kps", None),
        det_score=getattr(base_face, "det_score", 1.0),
        embedding=mean_emb.astype(np.float32),
        normed_embedding=normed.astype(np.float32),
    )


def load_source_face(source_file: str, app):
    """Mendukung satu foto (.jpg/.png) maupun arsip faceset (.fsz / .zip) dengan rata-rata embedding multi-sudut."""
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
            # Ambil maksimal 30 sampel sudut merata agar ringan di RAM Colab
            if len(all_imgs) > 30:
                step = len(all_imgs) / 30.0
                all_imgs = [all_imgs[int(i * step)] for i in range(30)]

            for fpath in all_imgs:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    detected_faces.append(best)

        gc.collect()
        if not detected_faces:
            raise RuntimeError("Tidak ada wajah yang terdeteksi di dalam file faceset (.fsz).")

        base_face = detected_faces[0]
        raw_embs = [
            f.normed_embedding
            for f in detected_faces
            if getattr(f, "normed_embedding", None) is not None
        ]
        if raw_embs:
            mean_emb = np.mean(np.stack(raw_embs, axis=0), axis=0)
            merged_face = _make_merged_face_obj(base_face, mean_emb)
        else:
            merged_face = base_face

        print(f"[FACESET] Berhasil memuat {len(detected_faces)} sudut wajah dari {src_path.name}", flush=True)
        _CACHE["source_key"] = cache_key
        _CACHE["source_face"] = merged_face
        return merged_face

    src_img = cv2.imread(str(src_path))
    if src_img is None:
        raise FileNotFoundError(f"Gagal membaca file sumber: {src_path}")
    faces = app.get(src_img)
    if not faces:
        raise RuntimeError("Wajah tidak terdeteksi pada gambar sumber.")
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    _CACHE["source_key"] = cache_key
    _CACHE["source_face"] = best_face
    return best_face


def build_landmark_mask(frame_shape, face, feather_blur: int = 21) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    pts = None
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        pts = np.array(face.landmark_2d_106, dtype=np.float32)
    elif hasattr(face, "kps") and face.kps is not None:
        kps = np.array(face.kps, dtype=np.float32)
        center = kps.mean(axis=0)
        expanded = center + (kps - center) * 1.65
        pts = np.vstack([kps, expanded])

    if pts is not None and len(pts) >= 5:
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    else:
        x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
        cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), 1.0, -1)

    k = max(11, feather_blur | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)[..., None]


def process_face_on_frame(
    frame: np.ndarray,
    face,
    src_face,
    swapper,
    enhancer_obj: Optional[OnnxFaceEnhancer],
    enhancer_blend: float,
    mask_blur: int,
    blend_alpha: float,
) -> np.ndarray:
    swapped_full = swapper.get(frame.copy(), face, src_face, paste_back=True)

    if enhancer_obj is not None and enhancer_blend > 0.0:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
        pad_w = int((x2 - x1) * 0.25)
        pad_h = int((y2 - y1) * 0.25)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)

        if cx2 > cx1 and cy2 > cy1:
            crop = swapped_full[cy1:cy2, cx1:cx2]
            enhanced_crop = enhancer_obj.enhance_crop(crop)
            alpha = float(np.clip(enhancer_blend, 0.0, 1.0))
            blended_crop = (
                enhanced_crop.astype(np.float32) * alpha
                + crop.astype(np.float32) * (1.0 - alpha)
            )
            swapped_full[cy1:cy2, cx1:cx2] = np.clip(blended_crop, 0, 255).astype(np.uint8)

    alpha_mask = build_landmark_mask(frame.shape, face, feather_blur=mask_blur) * float(
        np.clip(blend_alpha, 0.0, 1.0)
    )
    out = swapped_full.astype(np.float32) * alpha_mask + frame.astype(np.float32) * (1.0 - alpha_mask)
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
    enhancer: str,
    enhancer_blend: float,
    face_mode: str,
    det_size: int,
    mask_blur: int,
    blend_alpha: float,
) -> Tuple[Optional[np.ndarray], str]:
    try:
        if not source_file or not target_file:
            return None, "Unggah Sumber Wajah (.jpg/.png/.fsz) dan Target, lalu geser slider / klik Cek Preview."

        app = get_face_analyser(det_size=int(det_size))
        swapper = get_swapper()
        enhancer_obj = get_enhancer(enhancer)
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
            status = f"Live Preview Frame {safe_idx + 1} / {total}"
        else:
            frame = cv2.imread(str(target_path))
            if frame is None:
                return None, "Gagal membaca gambar target."
            status = "Live Preview Gambar Target"

        out_frame = frame.copy()
        faces = select_target_faces(app.get(frame), mode=face_mode)
        for face in faces:
            out_frame = process_face_on_frame(
                out_frame,
                face,
                src_face,
                swapper,
                enhancer_obj,
                enhancer_blend=float(enhancer_blend),
                mask_blur=int(mask_blur),
                blend_alpha=float(blend_alpha),
            )

        rgb_preview = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        return rgb_preview, status
    except Exception as exc:
        return None, f"Error Preview: {exc}"


def run_swap_task(
    source_file: str,
    target_file: str,
    enhancer: str = "restoreformer++",
    enhancer_blend: float = 0.15,
    face_mode: str = "largest",
    det_size: int = 640,
    mask_blur: int = 21,
    blend_alpha: float = 1.0,
    output_file: str = "output/gradio_result.mp4",
    progress=None,
) -> str:
    if not source_file or not target_file:
        raise ValueError("Sumber wajah dan file target wajib diisi.")

    app = get_face_analyser(det_size=int(det_size))
    swapper = get_swapper()
    enhancer_obj = get_enhancer(enhancer)
    src_face = load_source_face(source_file, app)

    target_path = Path(target_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if target_path.suffix.lower() in video_exts:
        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Gagal membuka video target: {target_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        temp_video = out_path.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            out_frame = frame.copy()
            faces = select_target_faces(app.get(frame), mode=face_mode)
            for face in faces:
                out_frame = process_face_on_frame(
                    out_frame,
                    face,
                    src_face,
                    swapper,
                    enhancer_obj,
                    enhancer_blend=float(enhancer_blend),
                    mask_blur=int(mask_blur),
                    blend_alpha=float(blend_alpha),
                )
            writer.write(out_frame)

            if idx % 20 == 0 or idx == total:
                gc.collect()
                if progress is not None:
                    progress((idx, total), desc=f"Merender frame {idx}/{total}")
                print(f"[CUDA PROSES] Frame {idx}/{total}", flush=True)

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
        return str(out_path)

    tgt_img = cv2.imread(str(target_path))
    if tgt_img is None:
        raise FileNotFoundError(f"Gagal membaca foto target: {target_path}")
    out = tgt_img.copy()
    faces = select_target_faces(app.get(tgt_img), mode=face_mode)
    for face in faces:
        out = process_face_on_frame(
            out,
            face,
            src_face,
            swapper,
            enhancer_obj,
            enhancer_blend=float(enhancer_blend),
            mask_blur=int(mask_blur),
            blend_alpha=float(blend_alpha),
        )

    img_out = out_path if out_path.suffix.lower() in {".jpg", ".jpeg", ".png"} else out_path.with_suffix(".png")
    cv2.imwrite(str(img_out), out)
    return str(img_out)


def launch_gradio_ui(share: bool = True) -> None:
    import gradio as gr
    import onnxruntime as ort

    providers = ort.get_available_providers()
    gpu_status = "CUDA GPU AKTIF" if "CUDAExecutionProvider" in providers else "CPU MODE (Aktifkan GPU T4 di Colab!)"

    with gr.Blocks(title="FaceSwap ONNX CUDA Live Preview") as demo:
        gr.Markdown(f"## FaceSwap ONNX CUDA UI (Live Preview & .FSZ) — **{gpu_status}**")

        with gr.Row():
            with gr.Column(scale=1):
                source_in = gr.File(
                    label="Sumber Wajah (.jpg / .png / faceset .fsz)",
                    type="filepath",
                )
                target_in = gr.File(
                    label="Video atau Foto Target",
                    type="filepath",
                )
                enhancer_in = gr.Dropdown(
                    choices=["restoreformer++", "gpen-512", "codeformer", "none"],
                    value="restoreformer++",
                    label="Enhancer ONNX",
                )
                enhancer_blend_in = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.15,
                    label="Enhancer Blend (Mentok Kiri = Natural)",
                )
                face_mode_in = gr.Dropdown(
                    choices=["largest", "all"],
                    value="largest",
                    label="Target Wajah",
                )
                det_size_in = gr.Slider(minimum=320, maximum=1024, step=64, value=640, label="Det Size")
                mask_blur_in = gr.Slider(minimum=11, maximum=51, step=2, value=21, label="Mask Blur")
                blend_alpha_in = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Face Blend Alpha")

            with gr.Column(scale=1):
                preview_img = gr.Image(label="Live Preview Hasil Per Frame", interactive=False)
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Geser Frame untuk Cek Hasil (Live Preview)",
                )
                preview_btn = gr.Button("Cek / Refresh Preview Frame", variant="secondary")
                preview_status = gr.Textbox(label="Info Preview", interactive=False)
                run_btn = gr.Button("Render Video Lengkap", variant="primary")
                output_file_box = gr.File(label="Hasil Video/Foto Lengkap")
                status_box = gr.Textbox(label="Status Render", interactive=False)

        preview_inputs = [
            source_in,
            target_in,
            frame_slider,
            enhancer_in,
            enhancer_blend_in,
            face_mode_in,
            det_size_in,
            mask_blur_in,
            blend_alpha_in,
        ]

        # Saat video target diunggah, cukup update jumlah maksimum frame slider tanpa memicu crash paralel
        def on_target_uploaded(tgt):
            total = get_video_frame_count(tgt)
            max_idx = max(0, total - 1)
            return gr.update(minimum=0, maximum=max_idx, value=0), f"Target siap ({total} frame). Geser slider atau klik Cek Preview."

        target_in.change(
            fn=on_target_uploaded,
            inputs=[target_in],
            outputs=[frame_slider, preview_status],
        )

        # Trigger preview saat slider dilepas atau tombol Cek Preview ditekan
        frame_slider.release(
            fn=preview_single_frame,
            inputs=preview_inputs,
            outputs=[preview_img, preview_status],
        )
        preview_btn.click(
            fn=preview_single_frame,
            inputs=preview_inputs,
            outputs=[preview_img, preview_status],
        )

        def on_render_full(src, tgt, enh, eblend, fmode, dsz, mblur, balpha, progress=gr.Progress()):
            res_path = run_swap_task(
                source_file=src,
                target_file=tgt,
                enhancer=enh,
                enhancer_blend=float(eblend),
                face_mode=fmode,
                det_size=int(dsz),
                mask_blur=int(mblur),
                blend_alpha=float(balpha),
                progress=progress,
            )
            return res_path, f"Selesai: {res_path}"

        run_btn.click(
            fn=on_render_full,
            inputs=[
                source_in,
                target_in,
                enhancer_in,
                enhancer_blend_in,
                face_mode_in,
                det_size_in,
                mask_blur_in,
                blend_alpha_in,
            ],
            outputs=[output_file_box, status_box],
        )

    demo.queue(max_size=5).launch(share=share, debug=False)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="FaceSwap ONNX CUDA Live Preview Runner")
    parser.add_argument("--no-gradio", action="store_true")
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--output", default="output/result.mp4")
    parser.add_argument("--enhancer", choices=["restoreformer++", "gpen-512", "codeformer", "none"], default="restoreformer++")
    parser.add_argument("--enhancer-blend", type=float, default=0.15)
    parser.add_argument("--face-mode", choices=["largest", "all"], default="largest")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--mask-blur", type=int, default=21)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    use_gradio = not args.no_gradio
    if not args.skip_install:
        ensure_packages(use_gradio=use_gradio)

    if use_gradio:
        launch_gradio_ui(share=not args.no_share)
        return

    out = run_swap_task(
        source_file=args.source,
        target_file=args.target,
        enhancer=args.enhancer,
        enhancer_blend=args.enhancer_blend,
        face_mode=args.face_mode,
        det_size=args.det_size,
        mask_blur=args.mask_blur,
        blend_alpha=args.blend_alpha,
        output_file=args.output,
    )
    print(f"[SELESAI] {out}")


if __name__ == "__main__":
    main()
