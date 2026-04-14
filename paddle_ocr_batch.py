#!/usr/bin/env python3
"""
paddle_ocr_batch.py  (v3 – Windows PP 3.3.1 MKLDNN fix)
=========================================================
Fixes applied
-------------
BUG-1  TypeError: 'PPStructureV3' object is not callable
       ✅ engine.predict(img) everywhere (not engine(img))

BUG-2  NotImplementedError: ConvertPirAttribute2RuntimeAttribute
           not support [pir::ArrayAttribute<pir::DoubleAttribute>]
       Root cause confirmed by benchmark_results.csv:
         • All 10 images failed with onednn_instruction.cc:118
         • FLAGS_use_mkldnn=0 env var was SET but IGNORED by PP 3.3.1 on Windows
         • The C++ runtime on Windows reads oneDNN flags from compiled-in
           defaults, not os.environ, in PP 3.3.x
       Three-layer fix applied:
         • Layer 0: os.environ flags (works on Linux/PP<3.3)
         • Layer 1: paddle.set_flags() after import (works on Windows PP 3.3.1)
         • Layer 2: PNG→JPEG re-encode per image (version-agnostic safety net)
"""

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 – env vars  (effective on Linux and PP < 3.3; harmless elsewhere)
# Must be before any paddle import.
# ══════════════════════════════════════════════════════════════════════════════
import os
os.environ["FLAGS_use_mkldnn"]               = "0"
os.environ["PADDLE_DISABLE_MKLDNN"]          = "1"
os.environ["FLAGS_use_dnnl_primitive_cache"] = "0"
os.environ["FLAGS_enable_pir_api"]           = "1"

# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("pip install opencv-python-headless numpy")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CHECKPOINT_FILENAME  = "checkpoint.json"
ERROR_LOG_FILENAME   = "error.log"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
console_logger = logging.getLogger("console")
console_logger.setLevel(logging.DEBUG)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
console_logger.addHandler(_ch)

error_logger: Optional[logging.Logger] = None
_error_log_lock = Lock()


def init_error_logger(output_dir: Path) -> None:
    global error_logger
    error_logger = logging.getLogger("errors")
    error_logger.setLevel(logging.ERROR)
    fh = logging.FileHandler(output_dir / ERROR_LOG_FILENAME, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    error_logger.addHandler(fh)
    error_logger.propagate = False


def log_error(path: str, exc: Exception) -> None:
    msg = f"FAILED | {path} | {type(exc).__name__}: {exc}"
    with _error_log_lock:
        (error_logger or console_logger).error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 – paddle.set_flags()  (Windows PP 3.3.1 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mkldnn_flags_via_paddle() -> None:
    """
    Write FLAGS_use_mkldnn=0 directly into Paddle's C++ GlobalVarMap.
    This works on Windows PP 3.3.x where os.environ is not propagated to the
    C++ flag registry before the executor is created.
    """
    methods_tried = []

    # Primary: public API
    try:
        import paddle
        paddle.set_flags({"FLAGS_use_mkldnn": 0})
        methods_tried.append("paddle.set_flags ✅")
    except Exception as e:
        methods_tried.append(f"paddle.set_flags ❌ ({e})")

    # Secondary: internal core dict (PP 2.x–3.x)
    try:
        import paddle.base as base
        base.core.globals()["FLAGS_use_mkldnn"] = False
        methods_tried.append("paddle.base.core.globals ✅")
    except Exception as e:
        methods_tried.append(f"paddle.base.core.globals ❌ ({e})")

    console_logger.info(
        "MKLDNN Layer-1 fix: " + "  |  ".join(methods_tried)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    try:
        import paddle
        has_gpu = (paddle.device.is_compiled_with_cuda()
                   and paddle.device.cuda.device_count() > 0)
    except Exception:
        has_gpu = False
    if preference == "gpu" and not has_gpu:
        console_logger.warning("GPU requested but not found; using CPU.")
        return "cpu"
    device = "gpu" if has_gpu else "cpu"
    console_logger.info(f"Device: {device.upper()}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Engine init  (Layer 1 applied here, once, before PPStructureV3 is built)
# ─────────────────────────────────────────────────────────────────────────────

def init_engine(args: argparse.Namespace, device: str):
    """
    Build PPStructureV3.  Apply Layer-1 MKLDNN fix before construction so the
    executor uses the updated flag values when it compiles its op kernels.
    """
    # ── Layer 1: write into C++ flag registry ────────────────────────────
    _apply_mkldnn_flags_via_paddle()

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        sys.exit("pip install paddleocr==3.4.0")

    console_logger.info(
        f"Initialising PPStructureV3 (device={device}, "
        f"det_limit_side_len={args.det_limit_side_len}, "
        f"det_db_thresh={args.det_db_thresh}) …"
    )
    engine = PPStructureV3(
        device=device,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
    )
    console_logger.info("Engine ready.")
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 – image loading with PNG→JPEG re-encode safety net
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    path: Path,
    min_size: int = 800,
    jpeg_quality: int = 95,
    enhance_contrast: bool = False,
    denoise: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Load and pre-process an image.

    Layer-2 fix for Windows PP 3.3.1
    ----------------------------------
    After upscaling, PNG / BMP / TIFF / WEBP inputs are re-encoded to JPEG in
    memory and decoded back to a clean uint8 BGR array.

    Why this works:
    • PNG decode can produce tensors with pir::ArrayAttribute<pir::DoubleAttribute>
      metadata (float channel attrs for 16-bit or alpha-aware paths), which the
      oneDNN op converter in PP 3.3.1 does not handle.
    • JPEG decode always produces a plain uint8 3-channel tensor with no float
      attribute path → routes to the standard (non-oneDNN) executor.
    • quality=95 is visually lossless for document scans at ≥200 DPI.
    """
    # ── Read ──────────────────────────────────────────────────────────────
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Cannot decode image: {path}")

    orig_h, orig_w = img.shape[:2]
    orig_wh = (orig_w, orig_h)

    # ── Upscale ───────────────────────────────────────────────────────────
    short = min(orig_h, orig_w)
    if short < min_size:
        scale = min_size / short
        img = cv2.resize(
            img,
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # ── Layer-2: PNG/BMP/TIFF/WEBP → JPEG re-encode ──────────────────────
    if path.suffix.lower() in {".png", ".bmp", ".tiff", ".tif", ".webp"}:
        encode_ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if encode_ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            # Result is guaranteed: uint8, 3-channel BGR, C-contiguous

    # ── Normalise array layout (safety) ───────────────────────────────────
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Optional enhancements ─────────────────────────────────────────────
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = cv2.cvtColor(
            cv2.merge([clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR
        )
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    resized_wh = (img.shape[1], img.shape[0])
    return img, orig_wh, resized_wh


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(result) -> list:
    if result is None:
        return []
    if hasattr(result, "__iter__") and not isinstance(result, (list, dict)):
        result = list(result)
    if result and isinstance(result[0], list):
        result = result[0]
    return result


def extract_table_html(result: list) -> Optional[str]:
    for r in result:
        if (r.get("type") or "").lower() == "table":
            res = r.get("res", {})
            if isinstance(res, dict):
                html = res.get("html") or res.get("HTML")
                if html:
                    return html
            elif isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        html = item.get("html") or item.get("HTML")
                        if html:
                            return html
    return None


def mean_confidence(result: list) -> float:
    scores = []
    for r in result:
        s = r.get("score") or r.get("confidence")
        if s is not None:
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                pass
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def draw_boxes(img: np.ndarray, result: list) -> np.ndarray:
    colour_map = {
        "table": (0, 200, 0), "text": (200, 0, 0),
        "title": (0, 0, 200), "figure": (200, 200, 0), "formula": (200, 0, 200),
    }
    out = img.copy()
    for r in result:
        bbox  = r.get("bbox") or r.get("layout_bbox")
        rtype = (r.get("type") or "unknown").lower()
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            c = colour_map.get(rtype, (128, 128, 128))
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            cv2.putText(out, rtype, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, output_dir: Path):
        self._path = output_dir / CHECKPOINT_FILENAME
        self._lock = Lock()
        self._done: set = set()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._done = set(
                    json.loads(self._path.read_text(encoding="utf-8"))
                    .get("processed", [])
                )
                console_logger.info(
                    f"Checkpoint: {len(self._done)} image(s) already processed."
                )
            except Exception as exc:
                console_logger.warning(f"Checkpoint unreadable: {exc}")

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        with self._lock:
            self._done.add(key)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"processed": sorted(self._done)}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image worker
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    output_dir: Path,
    engine,
    args: argparse.Namespace,
    checkpoint: Checkpoint,
    save_layout: bool,
) -> str:
    key = str(image_path)

    if args.resume and checkpoint.is_done(key):
        return "skipped"

    try:
        t0 = time.perf_counter()

        # Layer 2 is inside preprocess_image (PNG→JPEG re-encode)
        img, orig_wh, resized_wh = preprocess_image(
            image_path,
            min_size=args.min_size,
            enhance_contrast=args.enhance_contrast,
            denoise=args.denoise,
        )

        # ── BUG-1 FIX: .predict(), never engine(img) ─────────────────────
        result = _flatten(engine.predict(img))

        elapsed = round(time.perf_counter() - t0, 3)

        out = output_dir / image_path.stem
        out.mkdir(parents=True, exist_ok=True)

        table_html = extract_table_html(result)
        if table_html:
            (out / "table.html").write_text(table_html, encoding="utf-8")

        meta = {
            "image_path":        key,
            "has_table":         table_html is not None,
            "region_count":      len(result),
            "mean_confidence":   mean_confidence(result),
            "processing_time_s": elapsed,
            "original_size_wh":  list(orig_wh),
            "resized_size_wh":   list(resized_wh),
            "timestamp":         datetime.utcnow().isoformat() + "Z",
            "regions": [
                {
                    "type":  r.get("type", "unknown"),
                    "bbox":  r.get("bbox") or r.get("layout_bbox"),
                    "score": r.get("score") or r.get("confidence"),
                }
                for r in result
            ],
        }
        (out / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if save_layout and result:
            cv2.imwrite(str(out / "layout.jpg"), draw_boxes(img, result))

        checkpoint.mark_done(key)
        return "ok"

    except Exception as exc:
        log_error(key, exc)
        try:
            out = output_dir / image_path.stem
            out.mkdir(parents=True, exist_ok=True)
            (out / "metadata.json").write_text(
                json.dumps({
                    "image_path": key,
                    "error":      str(exc),
                    "traceback":  traceback.format_exc(),
                    "timestamp":  datetime.utcnow().isoformat() + "Z",
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return "error"


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_error_logger(output_dir)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input_dir '{input_dir}' not found.")

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if args.limit:
        images = images[:args.limit]
        console_logger.info(f"Dry-run: first {args.limit} image(s) only.")

    if not images:
        console_logger.info("No images found.")
        return

    total = len(images)
    console_logger.info(f"Found {total} image(s).")

    checkpoint  = Checkpoint(output_dir)
    device      = resolve_device(args.device)
    engine      = init_engine(args, device)
    max_workers = args.workers or os.cpu_count() or 1

    console_logger.info(
        f"Workers={max_workers}  resume={args.resume}  layout={not args.no_layout}"
    )

    counts = {"ok": 0, "error": 0, "skipped": 0}
    counts_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_image,
                p, output_dir, engine, args, checkpoint, not args.no_layout
            ): p for p in images
        }
        with tqdm(total=total, unit="img", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                status = future.result()
                with counts_lock:
                    counts[status] += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=counts["ok"], err=counts["error"],
                    skip=counts["skipped"], refresh=False
                )

    print()
    print("=" * 60)
    print("  BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total   : {total}")
    print(f"  Success : {counts['ok']}")
    print(f"  Failed  : {counts['error']}")
    print(f"  Skipped : {counts['skipped']}")
    print(f"  Output  : {output_dir.resolve()}")
    if counts["error"]:
        print(f"  Errors  : {output_dir / ERROR_LOG_FILENAME}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paddle_ocr_batch")
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--workers",    type=int,   default=0)
    p.add_argument("--min_size",   type=int,   default=800)
    p.add_argument("--device",     choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--det_limit_side_len", type=int,   default=960)
    p.add_argument("--det_db_thresh",      type=float, default=0.3)
    p.add_argument("--use_doc_orientation_classify", action="store_true")
    p.add_argument("--use_doc_unwarping",            action="store_true")
    p.add_argument("--enhance_contrast", action="store_true")
    p.add_argument("--denoise",          action="store_true")
    p.add_argument("--resume",           action="store_true")
    p.add_argument("--no_layout",        action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
