import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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


def run_cmd(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def ensure_packages(use_gradio: bool) -> None:
    # Tanpa basicsr/gfpgan sehingga jauh lebih cepat & anti bentrok di Python 3.13
    pkgs = ["insightface==0.7.3", "onnxruntime-gpu", "opencv-python"]
    if use_gradio:
        pkgs.extend(["gradio==5.13.0", "pydantic==2.10.6"])
    print("[SETUP] Memastikan paket inti & CUDA ONNX terpasang...")
    run_cmd(f"{sys.executable} -m pip install -q " + " ".join(pkgs))


def download_model(filename: str, min_bytes: int = 10_000_000) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[OK] Model siap: {dest} ({dest.stat().st_size:,} bytes)")
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
    """Menjalankan model ONNX enhancer (RestoreFormer++, GPEN-512, CodeFormer) langsung via CUDA."""

    def __init__(self, model_path: Path, model_type: str):
        import onnxruntime as ort

        self.model_type = model_type
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
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
            # Untuk CodeFormerv0.1.onnx yang membutuhkan input weight (w)
            feed[self.input_names[1]] = np.array([0.5], dtype=np.float64)

        out = self.session.run(None, feed)[0][0]
        out = np.transpose(out, (1, 2, 0))
        out = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def build_face_analyser(det_size: int = 640):
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def build_swapper(swapper_path: Path):
    import insightface

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return insightface.model_zoo.get_model(str(swapper_path), providers=providers)


def build_enhancer(enhancer_choice: str) -> Optional[OnnxFaceEnhancer]:
    if enhancer_choice == "none":
        return None
    mapping = {
        "restoreformer++": ("restoreformer_plus_plus.onnx", 100_000_000),
        "gpen-512": ("GPEN-BFR-512.onnx", 100_000_000),
        "codeformer": ("CodeFormerv0.1.onnx", 100_000_000),
    }
    fname, min_b = mapping[enhancer_choice]
    mpath = download_model(fname, min_bytes=min_b)
    return OnnxFaceEnhancer(mpath, enhancer_choice)


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

    # Terapkan enhancer hanya pada area crop wajah (jauh lebih cepat & mata lebih akurat)
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
) -> str:
    if not source_file or not target_file:
        raise ValueError("Foto sumber dan file target wajib diisi.")

    swapper_path = download_model("inswapper_128.onnx", min_bytes=50_000_000)
    app = build_face_analyser(det_size=int(det_size))
    swapper = build_swapper(swapper_path)
    enhancer_obj = build_enhancer(enhancer)

    src_img = cv2.imread(source_file)
    if src_img is None:
        raise FileNotFoundError(f"Gagal membaca foto sumber: {source_file}")

    src_faces = app.get(src_img)
    if not src_faces:
        raise RuntimeError("Wajah tidak terdeteksi pada foto sumber.")
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

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
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            if idx % 30 == 0 or idx == total:
                print(f"[CUDA PROSES] Frame {idx}/{total}")

        cap.release()
        writer.release()

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

    with gr.Blocks(title="FaceSwap ONNX CUDA UI") as demo:
        gr.Markdown(f"## FaceSwap ONNX CUDA UI — Status: **{gpu_status}**")
        with gr.Row():
            source_in = gr.File(label="Foto Wajah Sumber (.png/.jpg)", type="filepath")
            target_in = gr.File(label="Video atau Foto Target", type="filepath")

        with gr.Row():
            enhancer_in = gr.Dropdown(
                choices=["restoreformer++", "gpen-512", "codeformer", "none"],
                value="restoreformer++",
                label="Enhancer ONNX (countfloyd)",
            )
            enhancer_blend_in = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.15,
                label="Enhancer Blend (Mentok Kiri = Paling Natural)",
            )
            face_mode_in = gr.Dropdown(
                choices=["largest", "all"],
                value="largest",
                label="Target Wajah",
            )

        with gr.Row():
            det_size_in = gr.Slider(minimum=320, maximum=1024, step=64, value=640, label="Det Size")
            mask_blur_in = gr.Slider(minimum=11, maximum=51, step=2, value=21, label="Mask Blur")
            blend_alpha_in = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Face Blend Alpha")

        run_btn = gr.Button("Jalankan Proses", variant="primary")
        output_file_box = gr.File(label="Hasil Keluaran")
        status_box = gr.Textbox(label="Status / Path File", interactive=False)

        def on_click(src, tgt, enh, eblend, fmode, dsz, mblur, balpha):
            res_path = run_swap_task(
                source_file=src,
                target_file=tgt,
                enhancer=enh,
                enhancer_blend=float(eblend),
                face_mode=fmode,
                det_size=int(dsz),
                mask_blur=int(mblur),
                blend_alpha=float(balpha),
            )
            return res_path, f"Selesai: {res_path}"

        run_btn.click(
            fn=on_click,
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

    demo.launch(share=share, debug=False)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="FaceSwap ONNX CUDA Gradio Runner")
    parser.add_argument("--no-gradio", action="store_true")
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--output", default="output/result.mp4")
    parser.add_argument("--enhancer", choices=["restoreformer++", "gfpgan", "gpen-512", "codeformer", "none"], default="restoreformer++")
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
