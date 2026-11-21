import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK_DIR = Path("/content/facefusion")
FSZ_EXTRACT_DIR = Path("/content/extracted_faceset")


def run_cmd(cmd: str) -> int:
    print(f"[CMD] {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def extract_fsz_archives() -> None:
    """Mengekstrak otomatis semua file .fsz yang ada di /content agar fotonya bisa langsung dipilih di UI."""
    FSZ_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    fsz_files = list(Path("/content").glob("*.fsz"))
    if not fsz_files:
        return

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for fsz in fsz_files:
        target_sub = FSZ_EXTRACT_DIR / fsz.stem
        if target_sub.exists():
            shutil.rmtree(target_sub)
        target_sub.mkdir(parents=True, exist_ok=True)

        count = 0
        with zipfile.ZipFile(fsz, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                ext = Path(member.filename).suffix.lower()
                if ext in img_exts:
                    count += 1
                    out_file = target_sub / f"angle_{count:03d}{ext}"
                    with zf.open(member) as src, open(out_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        print(f"[FACESET] {fsz.name} diekstrak ({count} foto sudut wajah) -> {target_sub}", flush=True)


def patch_requirements_for_py313() -> None:
    """Melepas pin versi ketat pada paket C/C++ agar memakai prebuilt wheel Python 3.13 di Colab."""
    req_file = WORK_DIR / "requirements.txt"
    if not req_file.exists():
        return

    lines = req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    patched = []
    relax_pkgs = ("onnx", "onnxruntime", "numpy", "scipy", "opencv-python", "tqdm", "psutil", "gradio")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            patched.append(line)
            continue

        pkg_name = re.split(r"[=<>!~;\[]", stripped)[0].strip().lower()
        if pkg_name in ("onnxruntime", "onnxruntime-gpu"):
            continue
        if pkg_name in relax_pkgs:
            patched.append(pkg_name)
        else:
            patched.append(line)

    req_file.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print("[PATCH] requirements.txt disesuaikan untuk Python 3.13.", flush=True)


def setup_facefusion() -> None:
    os.chdir("/content")
    if not WORK_DIR.exists():
        print("[SETUP] Mengunduh FaceFusion 3.9.2...", flush=True)
        rc = run_cmd(f"git clone https://github.com/facefusion/facefusion.git --branch 3.9.2 --single-branch {WORK_DIR}")
        if rc != 0:
            raise RuntimeError("Gagal melakukan git clone FaceFusion 3.9.2.")

    os.chdir(WORK_DIR)
    patch_requirements_for_py313()

    print("[SETUP] Menginstal onnx, onnxruntime-gpu, dan dependensi FaceFusion 3.9.2...", flush=True)
    run_cmd(f"{sys.executable} -m pip install -q --upgrade pip wheel setuptools")
    run_cmd(f"{sys.executable} -m pip install -q onnx onnxruntime-gpu")
    run_cmd(f"{sys.executable} -m pip install -q -r requirements.txt")


def create_share_wrapper() -> Path:
    """Membuat runner yang memaksa gradio.Blocks.launch selalu memakai share=True."""
    wrapper_path = WORK_DIR / "run_colab_share.py"
    wrapper_code = """import sys
import gradio as gr

_orig_launch = gr.Blocks.launch

def _patched_launch(self, *args, **kwargs):
    kwargs["share"] = True
    return _orig_launch(self, *args, **kwargs)

gr.Blocks.launch = _patched_launch

from facefusion import core
if __name__ == "__main__":
    sys.argv = ["facefusion.py", "run", "--execution-providers", "cuda"]
    core.cli()
"""
    wrapper_path.write_text(wrapper_code, encoding="utf-8")
    print("[PATCH] Wrapper share=True dibuat di run_colab_share.py", flush=True)
    return wrapper_path


def main() -> None:
    extract_fsz_archives()
    setup_facefusion()
    wrapper = create_share_wrapper()
    os.chdir(WORK_DIR)
    print("\n[START] Menjalankan FaceFusion 3.9.2 UI dengan public link gradio.live...", flush=True)
    run_cmd(f"{sys.executable} {wrapper.name}")


if __name__ == "__main__":
    main()
