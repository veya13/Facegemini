import os
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


def setup_facefusion() -> None:
    os.chdir("/content")
    if not WORK_DIR.exists():
        print("[SETUP] Mengunduh repositori FaceFusion...", flush=True)
        run_cmd(f"git clone https://github.com/facefusion/facefusion.git --branch 3.1.1 --single-branch {WORK_DIR}")

    os.chdir(WORK_DIR)

    # Pasang dependensi GPU CUDA (onnxruntime-gpu) & FaceFusion
    print("[SETUP] Menginstal dependensi FaceFusion & CUDA GPU...", flush=True)
    run_cmd(f"{sys.executable} install.py --onnxruntime cuda --skip-conda")


def patch_gradio_share() -> None:
    """Memastikan UI FaceFusion membuka tautan publik gradio.live agar bisa diakses dari browser HP/Colab."""
    ui_core = WORK_DIR / "facefusion" / "uis" / "core.py"
    if not ui_core.exists():
        return
    content = ui_core.read_text(encoding="utf-8", errors="ignore")
    if "share = False" in content:
        content = content.replace("share = False", "share = True")
        ui_core.write_text(content, encoding="utf-8")
        print("[PATCH] Mengaktifkan share=True pada UI FaceFusion.", flush=True)


def main() -> None:
    extract_fsz_archives()
    setup_facefusion()
    patch_gradio_share()
    os.chdir(WORK_DIR)
    print("\n[START] Menjalankan FaceFusion UI dengan akselerasi CUDA...", flush=True)
    run_cmd(f"{sys.executable} facefusion.py run --execution-providers cuda")


if __name__ == "__main__":
    main()
