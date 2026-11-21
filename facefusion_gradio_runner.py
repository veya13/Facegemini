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
DRIVE_MODELS_DIR = Path("/content/drive/MyDrive/FaceSwap-Models")

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"


def run_cmd(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def ensure_packages(enhancer: str, use_gradio: bool) -> None:
    pkgs = ["insightface==0.7.3", "onnxruntime-gpu", "opencv-python"]
    if enhancer in ("gfpgan", "codeformer"):
        pkgs.extend(["gfpgan", "basicsr", "facexlib"])
    if use_gradio:
        pkgs.extend(["gradio==5.13.0", "pydantic==2.10.6"])
    print("[SETUP] Memastikan paket dependensi terpasang...")
    run_cmd(f"{sys.executable} -m pip install -q " + " ".join(pkgs))


def sync_from_drive(filename: str, dest: Path) -> bool:
    drive_file = DRIVE_MODELS_DIR / filename
    if drive_file.exists() and drive_file.stat().st_size > 1_000_000:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_file, dest)
        print(f"[DRIVE -> LOKAL] Disalin dari Drive: {filename}")
        return True
    return False


def save_to_drive(src: Path) -> None:
    if DRIVE_MODELS_DIR.parent.exists() and src.exists():
        DRIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, DRIVE_MODELS_DIR / src.name)
        print(f"[LOKAL -> DRIVE] Backup tersimpan: {src.name}")


def download_file(url: str, dest: Path, min_bytes: int = 10_000_000) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    print(f"[UNDUH] {url}")
    rc = run_cmd(f'wget -q --show-progress "{url}" -O "{dest}"')
    if rc == 0 and dest.exists() and dest.stat().st_size >= min_bytes:
        save_to_drive(dest)
        return True
    if dest.exists():
        dest.unlink()
    return False


def prepare_models(swapper_path: Path, enhancer: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not (swapper_path.exists() and swapper_path.stat().st_size > 50_000_000):
        sync_from_drive(swapper_path.name, swapper_path)

    if not (swapper_path.exists() and swapper_path.stat().st_size > 50_000_000):
        raise FileNotFoundError(
            f"Bobot swapper tidak ditemukan di '{swapper_path}' maupun '{DRIVE_MODELS_DIR / swapper_path.name}'."
        )

    if enhancer == "gfpgan":
        gfp_path = MODELS_DIR / "GFPGANv1.4.pth"
        if not sync_from_drive(gfp_path.name, gfp_path):
            if not download_file(GFPGAN_URL, gfp_path, min_bytes=50_000_000):
                raise RuntimeError("Gagal menyiapkan bobot GFPGANv1.4.pth.")
    elif enhancer == "codeformer":
        cf_path = MODELS_DIR / "codeformer.pth"
        if not sync_from_drive(cf_path.name, cf_path):
            if not download_file(CODEFORMER_URL, cf_path, min_bytes=100_000_000):
                raise RuntimeError("Gagal menyiapkan bobot codeformer.pth.")


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


def build_enhancer(enhancer: str):
    if enhancer == "none":
        return None
    if enhancer in ("gfpgan", "codeformer"):
        from gfpgan import GFPGANer

        weight_file = MODELS_DIR / ("GFPGANv1.4.pth" if enhancer == "gfpgan" else "codeformer.pth")
        if not weight_file.exists():
            weight_file = MODELS_DIR / "GFPGANv1.4.pth"
            if not weight_file.exists():
                download_file(GFPGAN_URL, weight_file, min_bytes=50_000_000)
        return GFPGANer(
            model_path=str(weight_file),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    raise ValueError(f"Enhancer tidak dikenal: {enhancer}")


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


def swap_single_face(
    original_frame: np.ndarray,
    swapped_frame: np.ndarray,
    face,
    mask_blur: int = 21,
    blend_alpha: float = 1.0,
) -> np.ndarray:
    alpha_mask = build_landmark_mask(original_frame.shape, face, feather_blur=mask_blur) * float(
        np.clip(blend_alpha, 0.0, 1.0)
    )
    orig_f = original_frame.astype(np.float32)
    swap_f = swapped_frame.astype(np.float32)
    blended = swap_f * alpha_mask + orig_f * (1.0 - alpha_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


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
    swapper_model: str = "models/inswapper_128.onnx",
    enhancer: str = "gfpgan",
    face_mode: str = "largest",
    det_size: int = 640,
    mask_blur: int = 21,
    blend_alpha: float = 1.0,
    enhance_every: int = 1,
    output_file: str = "output/gradio_result.mp4",
) -> str:
    if not source_file or not target_file:
        raise ValueError("Foto sumber dan file target wajib diisi.")

    ensure_packages(enhancer, use_gradio=False)
    swapper_path = Path(swapper_model)
    prepare_models(swapper_path, enhancer)

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
            if faces:
                raw_swapped = frame.copy()
                for face in faces:
                    raw_swapped = swapper.get(raw_swapped, face, src_face, paste_back=True)
                    out_frame = swap_single_face(
                        out_frame,
                        raw_swapped,
                        face,
                        mask_blur=int(mask_blur),
                        blend_alpha=float(blend_alpha),
                    )
                if enhancer_obj is not None and int(enhance_every) > 0 and (idx % int(enhance_every) == 0):
                    _, _, restored = enhancer_obj.enhance(
                        out_frame,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                    )
                    if restored is not None:
                        out_frame = restored
            writer.write(out_frame)

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
        raise FileNotFoundError(f"Gagal membaca gambar target: {target_path}")
    out = tgt_img.copy()
    faces = select_target_faces(app.get(tgt_img), mode=face_mode)
    if faces:
        raw_swapped = tgt_img.copy()
        for face in faces:
            raw_swapped = swapper.get(raw_swapped, face, src_face, paste_back=True)
            out = swap_single_face(
                out,
                raw_swapped,
                face,
                mask_blur=int(mask_blur),
                blend_alpha=float(blend_alpha),
            )
        if enhancer_obj is not None:
            _, _, restored = enhancer_obj.enhance(out, has_aligned=False, only_center_face=False, paste_back=True)
            if restored is not None:
                out = restored

    img_out = out_path if out_path.suffix.lower() in {".jpg", ".jpeg", ".png"} else out_path.with_suffix(".png")
    cv2.imwrite(str(img_out), out)
    return str(img_out)


def launch_gradio_ui(share: bool = True) -> None:
    import gradio as gr

    with gr.Blocks(title="FaceSwap Colab UI") as demo:
        gr.Markdown("## FaceSwap Colab UI (Gradio)")
        with gr.Row():
            source_in = gr.File(label="Foto Wajah Sumber (.png/.jpg)", type="filepath")
            target_in = gr.File(label="Video atau Foto Target", type="filepath")

        with gr.Row():
            swapper_in = gr.Textbox(value="models/inswapper_128.onnx", label="Path Model Swapper (.onnx)")
            enhancer_in = gr.Dropdown(
                choices=["none", "gfpgan", "codeformer"],
                value="gfpgan",
                label="Enhancer",
            )
            face_mode_in = gr.Dropdown(
                choices=["largest", "all"],
                value="largest",
                label="Target Wajah",
            )

        with gr.Row():
            det_size_in = gr.Slider(minimum=320, maximum=1024, step=64, value=640, label="Det Size")
            mask_blur_in = gr.Slider(minimum=11, maximum=51, step=2, value=21, label="Mask Blur")
            blend_alpha_in = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Blend Alpha")
            enhance_every_in = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Enhance Every N Frame")

        run_btn = gr.Button("Jalankan Proses", variant="primary")
        output_file_box = gr.File(label="Hasil Keluaran")
        status_box = gr.Textbox(label="Status / Path File", interactive=False)

        def on_click(src, tgt, swp, enh, fmode, dsz, mblur, balpha, eevery):
            res_path = run_swap_task(
                source_file=src,
                target_file=tgt,
                swapper_model=swp,
                enhancer=enh,
                face_mode=fmode,
                det_size=int(dsz),
                mask_blur=int(mblur),
                blend_alpha=float(balpha),
                enhance_every=int(eevery),
            )
            return res_path, f"Selesai: {res_path}"

        run_btn.click(
            fn=on_click,
            inputs=[
                source_in,
                target_in,
                swapper_in,
                enhancer_in,
                face_mode_in,
                det_size_in,
                mask_blur_in,
                blend_alpha_in,
                enhance_every_in,
            ],
            outputs=[output_file_box, status_box],
        )

    demo.launch(share=share, debug=False)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="FaceFusion-Style Gradio Runner")
    parser.add_argument("--no-gradio", action="store_true", help="Jalankan mode CLI tanpa UI Gradio")
    parser.add_argument("--no-share", action="store_true", help="Matikan share link publik Gradio")
    parser.add_argument("--source", default="", help="Path foto wajah sumber (mode CLI)")
    parser.add_argument("--target", default="", help="Path video/foto target (mode CLI)")
    parser.add_argument("--output", default="output/result.mp4", help="Path output (mode CLI)")
    parser.add_argument("--swapper-model", default="models/inswapper_128.onnx")
    parser.add_argument("--enhancer", choices=["none", "gfpgan", "codeformer"], default="gfpgan")
    parser.add_argument("--face-mode", choices=["largest", "all"], default="largest")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--mask-blur", type=int, default=21)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--enhance-every", type=int, default=1)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    use_gradio = not args.no_gradio
    if not args.skip_install:
        ensure_packages(args.enhancer, use_gradio=use_gradio)

    if use_gradio:
        launch_gradio_ui(share=not args.no_share)
        return

    out = run_swap_task(
        source_file=args.source,
        target_file=args.target,
        swapper_model=args.swapper_model,
        enhancer=args.enhancer,
        face_mode=args.face_mode,
        det_size=args.det_size,
        mask_blur=args.mask_blur,
        blend_alpha=args.blend_alpha,
        enhance_every=args.enhance_every,
        output_file=args.output,
    )
    print(f"[SELESAI] {out}")


if __name__ == "__main__":
    main()
