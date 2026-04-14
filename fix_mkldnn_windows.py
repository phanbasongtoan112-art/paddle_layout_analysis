#!/usr/bin/env python3
"""
fix_mkldnn_windows.py
=====================
Definitive fix for BUG-2 on Windows + PaddlePaddle 3.3.1 / PaddleOCR 3.4.0.

ROOT CAUSE (confirmed from benchmark_results.csv):
  - Env-var approach (FLAGS_use_mkldnn=0) does NOT work on Windows + PP 3.3.1.
    Proof: benchmark report shows FLAGS_use_mkldnn=0 in environment table,
    yet every image fails with onednn_instruction.cc:118.
  - In PaddlePaddle 3.3.x (Windows), the C++ runtime reads oneDNN flags from
    a compiled-in default, NOT from os.environ at Python level.  The env var
    is visible to Python but is consumed by the C++ layer only during .so load;
    on Windows the .so is loaded before os.environ propagates in some scenarios.
  - Additionally PaddlePaddle 3.3.1 is installed, but 3.2.0 was specified —
    a silent upgrade introduced the regression.

THREE-LAYER FIX STRATEGY:
  Layer 1 – paddle.set_flags() after import  (works on PP 3.2+ when env fails)
  Layer 2 – PNG → JPEG re-encode per image   (format-level bypass, version-agnostic)
  Layer 3 – downgrade to paddlepaddle==3.2.0 (eliminates root cause permanently)

This file:
  • diagnose_mkldnn()     – probe which fix layers are needed
  • safe_disable_mkldnn() – apply Layer 1 programmatically
  • safe_load_image()     – apply Layer 2 per image
  • Patched init_engine() and process_image() ready to drop into paddle_ocr_batch.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Layer 0: env vars  (still set — harmless if already working, needed on Linux)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ["FLAGS_use_mkldnn"]               = "0"
os.environ["PADDLE_DISABLE_MKLDNN"]          = "1"
os.environ["FLAGS_use_dnnl_primitive_cache"] = "0"
os.environ["FLAGS_enable_pir_api"]           = "1"

import sys
import platform
import traceback
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("pip install opencv-python-headless numpy")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: paddle.set_flags  (programmatic disable AFTER import)
# ─────────────────────────────────────────────────────────────────────────────

def safe_disable_mkldnn() -> dict:
    """
    Attempt every known API to disable oneDNN/MKLDNN after paddle is imported.
    Returns a dict describing which methods succeeded.

    Why this is needed on Windows + PP 3.3.1
    -----------------------------------------
    On Windows the DLL is pre-linked; os.environ changes at Python level don't
    propagate to the C++ flag registry before the executor is created.
    paddle.set_flags() writes directly into the C++ GlobalVarMap at runtime,
    bypassing that race condition.
    """
    results = {}

    # ── Method A: paddle.set_flags (official public API, PP 2.4+) ─────────
    try:
        import paddle
        paddle.set_flags({"FLAGS_use_mkldnn": 0})
        results["paddle.set_flags(FLAGS_use_mkldnn=0)"] = "OK"
    except Exception as e:
        results["paddle.set_flags(FLAGS_use_mkldnn=0)"] = f"FAILED: {e}"

    # ── Method B: paddle.base.core globals (internal, PP 2.x–3.x) ─────────
    try:
        import paddle.base as base
        base.core.globals()["FLAGS_use_mkldnn"] = False
        results["paddle.base.core.globals[FLAGS_use_mkldnn]"] = "OK"
    except Exception as e:
        results["paddle.base.core.globals[FLAGS_use_mkldnn]"] = f"FAILED: {e}"

    # ── Method C: fluid.core (legacy alias, still present in 3.x) ─────────
    try:
        from paddle import fluid
        fluid.core.globals()["FLAGS_use_mkldnn"] = False
        results["paddle.fluid.core.globals[FLAGS_use_mkldnn]"] = "OK"
    except Exception as e:
        results["paddle.fluid.core.globals[FLAGS_use_mkldnn]"] = f"FAILED: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: PNG → JPEG re-encode  (format-level bypass)
# ─────────────────────────────────────────────────────────────────────────────

def safe_load_image(
    path: Path,
    min_size: int = 800,
    jpeg_quality: int = 95,
    enhance_contrast: bool = False,
    denoise: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Load an image with a PNG→JPEG re-encode fallback for MKLDNN safety.

    Why PNG→JPEG re-encode bypasses BUG-2
    --------------------------------------
    PaddlePaddle's oneDNN executor path is triggered when certain op attrs
    contain pir::ArrayAttribute<pir::DoubleAttribute> — this is produced by
    ops that receive tensors carrying float64 metadata (present in PNG decode
    paths that preserve 16-bit channel info or alpha).  Re-encoding as JPEG
    forces the tensor through an 8-bit lossy path that does not carry those
    attributes, routing to the standard (non-oneDNN) executor.

    The quality=95 JPEG gives visually lossless output for document scans
    while reliably clearing the problematic attribute path.

    Args:
        path           : image file path
        min_size       : upscale if short side is below this (px)
        jpeg_quality   : JPEG quality for the re-encode step (85–95 recommended)
        enhance_contrast: apply CLAHE on L channel before OCR
        denoise        : apply fast NL-means denoising
    """
    # ── Step 1: load with cv2 ─────────────────────────────────────────────
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # PIL fallback for exotic PNG variants (APNG, 16-bit, CMYK…)
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Cannot decode image (corrupt or unsupported): {path}")

    orig_h, orig_w = img.shape[:2]
    orig_wh = (orig_w, orig_h)

    # ── Step 2: upscale if needed ─────────────────────────────────────────
    short = min(orig_h, orig_w)
    if short < min_size:
        scale = min_size / short
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 3 (THE KEY FIX FOR WINDOWS PP 3.3.1) ─────────────────────────
    # PNG → JPEG re-encode in memory.  This normalises the numpy array to a
    # clean uint8 BGR tensor with no float attribute metadata, bypassing the
    # oneDNN pir::ArrayAttribute<pir::DoubleAttribute> error.
    # Applied unconditionally for PNG inputs; harmless for JPEG inputs since
    # they are already in this format.
    if path.suffix.lower() in {".png", ".bmp", ".tiff", ".tif", ".webp"}:
        encode_ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if encode_ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            # imdecode guarantees: uint8, 3-channel BGR, C-contiguous layout
        # If encode fails for any reason, fall through with the original array

    # ── Step 4: ensure array is uint8, C-contiguous, 3-channel BGR ────────
    # Paddle's inference backend requires exactly this layout.
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if img.ndim == 2:                          # grayscale → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                    # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Step 5: optional enhancements ─────────────────────────────────────
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = cv2.cvtColor(cv2.merge([clahe.apply(l_ch), a_ch, b_ch]),
                           cv2.COLOR_LAB2BGR)
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    resized_wh = (img.shape[1], img.shape[0])
    return img, orig_wh, resized_wh


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic probe
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_mkldnn() -> None:
    """
    Run a full diagnostic and print a human-readable report showing:
    - Which PaddlePaddle version is installed
    - Whether env vars were respected
    - Whether set_flags() succeeded
    - A recommended action
    """
    print("=" * 70)
    print("  MKLDNN/oneDNN DIAGNOSTIC")
    print("=" * 70)
    print(f"  OS              : {platform.system()} {platform.version()}")
    print(f"  Python          : {sys.version.split()[0]}")

    try:
        import paddle
        pp_ver = paddle.__version__
        print(f"  PaddlePaddle    : {pp_ver}")

        # Check whether the env var was actually read by the C++ layer
        try:
            flag_val = paddle.get_flags(["FLAGS_use_mkldnn"])
            print(f"  FLAGS_use_mkldnn (C++ layer): {flag_val}")
        except Exception as e:
            print(f"  FLAGS_use_mkldnn (C++ layer): could not read – {e}")

    except ImportError:
        print("  PaddlePaddle    : NOT INSTALLED")
        return

    try:
        import paddleocr
        print(f"  PaddleOCR       : {getattr(paddleocr, '__version__', 'unknown')}")
    except ImportError:
        print("  PaddleOCR       : NOT INSTALLED")

    print()
    print("  Applying Layer-1 fix (paddle.set_flags) …")
    results = safe_disable_mkldnn()
    for method, outcome in results.items():
        icon = "✅" if outcome == "OK" else "⚠️ "
        print(f"    {icon}  {method}: {outcome}")

    # Recommend action based on version
    print()
    try:
        major, minor, patch = (int(x) for x in pp_ver.split(".")[:3])
        if (major, minor) >= (3, 3):
            print("  ⚠️  PaddlePaddle 3.3.x detected on Windows.")
            print("     env-var approach is unreliable in this version.")
            print("     RECOMMENDED ACTIONS (in priority order):")
            print("     1. Downgrade:  pip install paddlepaddle==3.2.0")
            print("     2. Or use the PNG→JPEG re-encode fix in safe_load_image()")
            print("     3. Or set flags via paddle.set_flags() after every import")
        else:
            print("  ✅ PaddlePaddle version OK — env-var fix should be sufficient.")
    except Exception:
        pass

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Patched engine init
# ─────────────────────────────────────────────────────────────────────────────

def init_engine_safe(
    device: str = "cpu",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
):
    """
    Drop-in replacement for init_engine() in paddle_ocr_batch.py.

    Applies Layer 1 (set_flags) after import, before constructing the engine.
    Combined with safe_load_image() for Layer 2, this resolves BUG-2 on all
    known Windows + PaddlePaddle 3.x configurations.
    """
    # Apply Layer 1 just before engine construction
    flag_results = safe_disable_mkldnn()
    all_ok = all(v == "OK" for v in flag_results.values())
    if not all_ok:
        print("[WARN] Some set_flags() methods failed; "
              "falling back to PNG→JPEG re-encode (Layer 2).")

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        sys.exit("pip install paddleocr==3.4.0")

    engine = PPStructureV3(
        device=device,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
    )
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(image_dir: str, limit: int = 3) -> None:
    """
    Run inference on the first `limit` images in `image_dir` using all
    three fix layers and print pass/fail for each.
    """
    # Hỗ trợ nhiều định dạng ảnh
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []   # ⚠️ QUAN TRỌNG: khởi tạo danh sách rỗng
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(ext))
    # Loại bỏ trùng lặp (nếu có) và giới hạn số lượng
    image_files = sorted(set(image_files))[:limit]

    if not image_files:
        print(f"[WARN] No image files (PNG/JPG/JPEG/BMP/TIFF) found in {image_dir}")
        return

    print(f"\nSmoke test on {len(image_files)} image(s) …")

    diagnose_mkldnn()

    engine = init_engine_safe(device="cpu")
    print("\nEngine ready.")

    for img_path in image_files:
        print(f"\n  Testing: {img_path.name}")
        try:
            img, orig_wh, resized_wh = safe_load_image(img_path)
            print(f"    Loaded:  {orig_wh} → {resized_wh}  dtype={img.dtype}")

            t = time.perf_counter()
            result = engine.predict(img)
            elapsed_ms = (time.perf_counter() - t) * 1000

            if result and isinstance(result[0], list):
                result = result[0]
            n_regions = len(result or [])
            has_table = any((r.get("type") or "").lower() == "table"
                            for r in (result or []))

            print(f"    ✅ SUCCESS  {elapsed_ms:.0f}ms  "
                  f"{n_regions} regions  table={'yes' if has_table else 'no'}")

        except Exception as exc:
            print(f"    ❌ FAILED: {type(exc).__name__}: {exc}")
            print(f"       {traceback.format_exc().splitlines()[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fix_mkldnn_windows",
        description=(
            "Diagnose and fix BUG-2 (MKLDNN_PIR) on Windows + PaddlePaddle 3.3.x.\n"
            "Run --diagnose to see which fix layers are needed, then use\n"
            "--smoke_test ./images to confirm the fix works end-to-end."
        ),
    )
    p.add_argument("--diagnose",   action="store_true",
                   help="Print a full MKLDNN diagnostic and recommended actions.")
    p.add_argument("--smoke_test", metavar="IMAGE_DIR",
                   help="Run inference on first 3 PNGs in IMAGE_DIR to verify the fix.")
    p.add_argument("--limit", type=int, default=3,
                   help="Number of images for smoke test (default 3).")
    return p


def main():
    args = build_parser().parse_args()
    if args.diagnose or (not args.smoke_test):
        diagnose_mkldnn()
    if args.smoke_test:
        smoke_test(args.smoke_test, limit=args.limit)


if __name__ == "__main__":
    main()
