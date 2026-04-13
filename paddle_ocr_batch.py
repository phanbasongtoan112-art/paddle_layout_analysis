#!/usr/bin/env python3
"""
paddle_ocr_batch.py
===================
Production-ready batch image processing with PaddleOCR PPStructureV3.
Performs layout detection and HTML table extraction across thousands of images
using parallel workers, checkpointing, GPU auto-detection, and structured output.

Requirements:
    pip install paddlepaddle==3.2.0   (or paddlepaddle-gpu for CUDA)
    pip install paddleocr==3.4.0
    pip install tqdm opencv-python-headless

Usage:
    python paddle_ocr_batch.py \
        --input_dir  ./images \
        --output_dir ./results \
        --workers    4 \
        --min_size   800 \
        --device     auto \
        --resume
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("ERROR: opencv-python-headless is not installed. Run: pip install opencv-python-headless")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed. Run: pip install numpy")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: tqdm is not installed. Run: pip install tqdm")

# PaddleOCR imports are deferred to init_engine() so we can report a clean error.

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CHECKPOINT_FILENAME   = "checkpoint.json"
ERROR_LOG_FILENAME    = "error.log"

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup  (file logger for errors; console logger for info/summary)
# ──────────────────────────────────────────────────────────────────────────────
console_logger = logging.getLogger("console")
console_logger.setLevel(logging.DEBUG)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
console_logger.addHandler(_ch)

# The file-based error logger is initialised after we know output_dir.
error_logger: Optional[logging.Logger] = None
_error_log_lock = Lock()  # serialise writes from multiple threads


def init_error_logger(output_dir: Path) -> None:
    """Create (or append to) error.log inside output_dir."""
    global error_logger
    error_log_path = output_dir / ERROR_LOG_FILENAME
    error_logger = logging.getLogger("errors")
    error_logger.setLevel(logging.ERROR)
    fh = logging.FileHandler(error_log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    error_logger.addHandler(fh)
    error_logger.propagate = False


def log_error(image_path: str, exc: Exception) -> None:
    """Thread-safe error logging."""
    msg = f"FAILED | {image_path} | {type(exc).__name__}: {exc}"
    with _error_log_lock:
        if error_logger:
            error_logger.error(msg)
        else:
            console_logger.error(msg)


# ──────────────────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(preference: str) -> str:
    """
    Return 'gpu' or 'cpu' based on the user's --device argument and
    what is actually available at runtime.

    preference: 'auto' | 'gpu' | 'cpu'
    """
    if preference == "cpu":
        return "cpu"

    # Try to detect CUDA through paddle
    try:
        import paddle
        has_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        has_gpu = False

    if preference == "gpu":
        if not has_gpu:
            console_logger.warning(
                "--device=gpu was requested but no CUDA GPU detected; falling back to CPU."
            )
            return "cpu"
        return "gpu"

    # 'auto': use GPU if available, else CPU
    device = "gpu" if has_gpu else "cpu"
    console_logger.info(f"Auto-detected device: {device.upper()}")
    return device


# ──────────────────────────────────────────────────────────────────────────────
# PPStructureV3 engine initialisation (called ONCE in the main process)
# ──────────────────────────────────────────────────────────────────────────────

def init_engine(args: argparse.Namespace, device: str):
    """
    Initialise the PPStructureV3 pipeline.

    The engine is heavy; it must be created once and shared across threads.
    PaddleOCR's inference is thread-safe for read operations, so sharing a
    single instance with a ThreadPoolExecutor is safe as long as you do not
    mutate engine state between calls.
    """
    try:
        from paddleocr import PPStructureV3
    except ImportError:
        sys.exit(
            "ERROR: paddleocr is not installed or the version does not support PPStructureV3.\n"
            "       Run: pip install paddleocr==3.4.0"
        )

    use_gpu = device == "gpu"

    console_logger.info(
        f"Initialising PPStructureV3  (device={device}, "
        f"det_limit_side_len={args.det_limit_side_len}, "
        f"det_db_thresh={args.det_db_thresh}) …"
    )

    engine = PPStructureV3(
    device=device,  # thay vì use_gpu=True/False
    use_doc_orientation_classify=args.use_doc_orientation_classify,
    use_doc_unwarping=args.use_doc_unwarping,
    )

    console_logger.info("Engine ready.")
    return engine


# ──────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    image_path: Path,
    min_size: int = 800,
    enhance_contrast: bool = False,
    denoise: bool = False,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Load and optionally upscale/enhance an image.

    Returns
    -------
    img          : BGR numpy array ready for PPStructureV3
    orig_size    : (width, height) of the original image
    resized_size : (width, height) after pre-processing (same as orig if no resize)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cv2.imread returned None – file may be corrupt or unsupported: {image_path}")

    orig_h, orig_w = img.shape[:2]
    orig_size = (orig_w, orig_h)

    # ── Upscale if short side is below min_size ────────────────────────────
    short_side = min(orig_h, orig_w)
    if short_side < min_size:
        scale = min_size / short_side
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    resized_size = (img.shape[1], img.shape[0])  # (w, h)

    # ── Optional contrast enhancement (CLAHE on luminance channel) ─────────
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── Optional fast non-local means denoising ────────────────────────────
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return img, orig_size, resized_size


# ──────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_table_html(structure_result: list) -> Optional[str]:
    """
    Walk the PPStructureV3 result list and return the HTML string of the
    first detected table, or None if no table was found.

    PPStructureV3 returns a list of region dicts.  Each dict has a 'type'
    key ('table', 'text', 'title', 'figure', …) and a 'res' payload.
    For tables, res['html'] contains the HTML string.
    """
    if not structure_result:
        return None

    for region in structure_result:
        region_type = region.get("type", "").lower()
        if region_type == "table":
            res = region.get("res", {})
            # PPStructureV3 may return res as a dict or a list of dicts
            if isinstance(res, dict):
                html = res.get("html") or res.get("HTML")
                if html:
                    return html
            elif isinstance(res, list):
                for item in res:
                    html = item.get("html") or item.get("HTML") if isinstance(item, dict) else None
                    if html:
                        return html
    return None


def get_region_confidence(structure_result: list) -> float:
    """
    Return the mean confidence across all detected regions, or 0.0 if
    confidence information is not available.
    """
    scores = []
    for region in (structure_result or []):
        score = region.get("score") or region.get("confidence")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def draw_layout_boxes(img: np.ndarray, structure_result: list) -> np.ndarray:
    """
    Draw colour-coded bounding boxes for each detected region.
    Returns a copy of the image with boxes overlaid.
    """
    colour_map = {
        "table":   (0,   200, 0),    # green
        "text":    (200, 0,   0),    # blue
        "title":   (0,   0,   200),  # red
        "figure":  (200, 200, 0),    # cyan
        "formula": (200, 0,   200),  # magenta
    }
    default_colour = (128, 128, 128)

    annotated = img.copy()
    for region in (structure_result or []):
        bbox = region.get("bbox") or region.get("layout_bbox")
        rtype = region.get("type", "unknown").lower()
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            colour = colour_map.get(rtype, default_colour)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                annotated, rtype, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA
            )
    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ──────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    """
    Thread-safe JSON checkpoint.
    Tracks which image paths have already been successfully processed.
    """

    def __init__(self, output_dir: Path):
        self._path = output_dir / CHECKPOINT_FILENAME
        self._lock = Lock()
        self._processed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._processed = set(data.get("processed", []))
                console_logger.info(
                    f"Checkpoint loaded: {len(self._processed)} previously processed image(s)."
                )
            except Exception as exc:
                console_logger.warning(f"Could not read checkpoint ({exc}); starting fresh.")

    def is_done(self, image_path: str) -> bool:
        return image_path in self._processed

    def mark_done(self, image_path: str) -> None:
        with self._lock:
            self._processed.add(image_path)
            # Flush to disk every time for crash-safety
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"processed": sorted(self._processed)}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)  # atomic rename on POSIX


# ──────────────────────────────────────────────────────────────────────────────
# Core per-image worker
# ──────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    output_dir: Path,
    engine,
    args: argparse.Namespace,
    checkpoint: Checkpoint,
    save_layout_image: bool = True,
) -> str:
    """
    Process a single image through PPStructureV3 and write outputs.

    Returns
    -------
    'skipped'  – already in checkpoint
    'ok'       – processed successfully
    'error'    – an exception was caught (details in error.log)
    """
    img_key = str(image_path)

    # ── Skip if already processed (resume mode) ────────────────────────────
    if args.resume and checkpoint.is_done(img_key):
        return "skipped"

    try:
        t_start = time.perf_counter()

        # ── Pre-process ───────────────────────────────────────────────────
        img, orig_size, resized_size = preprocess_image(
            image_path,
            min_size=args.min_size,
            enhance_contrast=args.enhance_contrast,
            denoise=args.denoise,
        )

        # ── Run PPStructureV3 ─────────────────────────────────────────────
        result = engine.predict(img)  # returns list of region dicts

        elapsed = round(time.perf_counter() - t_start, 3)

        # ── Prepare output directory ──────────────────────────────────────
        stem = image_path.stem
        out_folder = output_dir / stem
        out_folder.mkdir(parents=True, exist_ok=True)

        # ── Extract table HTML ────────────────────────────────────────────
        table_html = extract_table_html(result)
        has_table  = table_html is not None

        if has_table:
            (out_folder / "table.html").write_text(table_html, encoding="utf-8")

        # ── Metadata ──────────────────────────────────────────────────────
        metadata = {
            "image_path":        str(image_path),
            "has_table":         has_table,
            "region_count":      len(result) if result else 0,
            "mean_confidence":   get_region_confidence(result),
            "processing_time_s": elapsed,
            "original_size_wh":  list(orig_size),
            "resized_size_wh":   list(resized_size),
            "timestamp":         datetime.utcnow().isoformat() + "Z",
            "regions": [
                {
                    "type":  r.get("type", "unknown"),
                    "bbox":  r.get("bbox") or r.get("layout_bbox"),
                    "score": r.get("score") or r.get("confidence"),
                }
                for r in (result or [])
            ],
        }
        (out_folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Optional layout visualisation ─────────────────────────────────
        if save_layout_image and result:
            annotated = draw_layout_boxes(img, result)
            cv2.imwrite(str(out_folder / "layout.jpg"), annotated)

        # ── Mark checkpoint ───────────────────────────────────────────────
        checkpoint.mark_done(img_key)
        return "ok"

    except Exception as exc:
        log_error(img_key, exc)
        # Write a minimal metadata file so the run is traceable
        try:
            out_folder = output_dir / image_path.stem
            out_folder.mkdir(parents=True, exist_ok=True)
            error_meta = {
                "image_path": str(image_path),
                "error":      str(exc),
                "traceback":  traceback.format_exc(),
                "timestamp":  datetime.utcnow().isoformat() + "Z",
            }
            (out_folder / "metadata.json").write_text(
                json.dumps(error_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass  # if we can't even write the error metadata, just move on
        return "error"


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def collect_images(input_dir: Path, limit: Optional[int] = None) -> list[Path]:
    """Return a sorted list of image paths from input_dir (non-recursive)."""
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None and limit > 0:
        images = images[:limit]
        console_logger.info(f"Dry-run: limited to first {limit} image(s).")
    return images


def run_batch(args: argparse.Namespace) -> None:
    # ── Setup output directory & logging ──────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_error_logger(output_dir)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input_dir '{input_dir}' does not exist or is not a directory.")

    # ── Collect images ────────────────────────────────────────────────────
    images = collect_images(input_dir, limit=args.limit)
    if not images:
        console_logger.info("No supported images found in input directory. Exiting.")
        return

    total = len(images)
    console_logger.info(f"Found {total} image(s) in '{input_dir}'.")

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint = Checkpoint(output_dir)

    # ── Determine device & initialise engine (ONCE) ───────────────────────
    device = resolve_device(args.device)
    engine = init_engine(args, device)

    # ── Determine worker count ────────────────────────────────────────────
    max_workers = args.workers if args.workers > 0 else os.cpu_count() or 1
    console_logger.info(
        f"Starting batch with {max_workers} worker thread(s) "
        f"(resume={args.resume}, save_layout={not args.no_layout}) …"
    )

    # ── Counters ──────────────────────────────────────────────────────────
    counts = {"ok": 0, "error": 0, "skipped": 0}
    counts_lock = Lock()

    # ── Submit all tasks ──────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                process_image,
                img_path,
                output_dir,
                engine,
                args,
                checkpoint,
                not args.no_layout,
            ): img_path
            for img_path in images
        }

        with tqdm(total=total, unit="img", desc="Processing", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_path):
                result_status = future.result()   # 'ok' | 'error' | 'skipped'
                with counts_lock:
                    counts[result_status] += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=counts["ok"],
                    err=counts["error"],
                    skip=counts["skipped"],
                    refresh=False,
                )

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Total images   : {total}")
    print(f"  Successful     : {counts['ok']}")
    print(f"  Failed         : {counts['error']}")
    print(f"  Skipped (done) : {counts['skipped']}")
    print(f"  Output dir     : {output_dir.resolve()}")
    if counts["error"] > 0:
        print(f"  Error log      : {output_dir / ERROR_LOG_FILENAME}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle_ocr_batch",
        description=(
            "Batch layout detection & table HTML extraction using PaddleOCR PPStructureV3.\n"
            "Processes thousands of images with parallel workers, checkpointing, and GPU support."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Basic run with 8 workers
  python paddle_ocr_batch.py --input_dir ./images --output_dir ./results --workers 8

  # Dry-run: test on first 10 images only
  python paddle_ocr_batch.py --input_dir ./images --output_dir ./results --limit 10

  # Resume a previously interrupted run
  python paddle_ocr_batch.py --input_dir ./images --output_dir ./results --resume

  # Force CPU, enable contrast enhancement, skip layout images
  python paddle_ocr_batch.py --input_dir ./images --output_dir ./results \\
      --device cpu --enhance_contrast --no_layout
""",
    )

    # ── I/O ───────────────────────────────────────────────────────────────
    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument(
        "--input_dir", required=True,
        help="Directory containing input images (.jpg, .png, .bmp, …).",
    )
    io_grp.add_argument(
        "--output_dir", required=True,
        help=(
            "Root directory where per-image output folders will be created.  "
            "Each image gets a sub-folder <stem>/ containing table.html, "
            "metadata.json, and optionally layout.jpg."
        ),
    )

    # ── Parallelism ───────────────────────────────────────────────────────
    parallel_grp = parser.add_argument_group("Parallelism")
    parallel_grp.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Number of parallel ThreadPoolExecutor workers.  "
            "0 (default) = number of CPU cores.  "
            "For GPU-heavy workloads try 1–2; for CPU try cpu_count/2."
        ),
    )

    # ── Image pre-processing ──────────────────────────────────────────────
    pre_grp = parser.add_argument_group("Image pre-processing")
    pre_grp.add_argument(
        "--min_size", type=int, default=800,
        help=(
            "Minimum short-side length in pixels.  "
            "Images smaller than this are upscaled (LANCZOS4).  Default: 800."
        ),
    )
    pre_grp.add_argument(
        "--enhance_contrast", action="store_true",
        help="Apply CLAHE contrast enhancement before OCR (useful for faded scans).",
    )
    pre_grp.add_argument(
        "--denoise", action="store_true",
        help="Apply non-local means denoising before OCR (slow; use only if needed).",
    )

    # ── Device ────────────────────────────────────────────────────────────
    dev_grp = parser.add_argument_group("Device")
    dev_grp.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto",
        help=(
            "'auto' detects CUDA; 'gpu' forces GPU (fails with warning if absent); "
            "'cpu' always uses CPU.  Default: auto."
        ),
    )

    # ── Engine tuning ─────────────────────────────────────────────────────
    engine_grp = parser.add_argument_group("PPStructureV3 engine tuning")
    engine_grp.add_argument(
        "--det_limit_side_len", type=int, default=960,
        help=(
            "Max side length fed to the text detector.  "
            "Larger → better recall on small text, slower.  Default: 960."
        ),
    )
    engine_grp.add_argument(
        "--det_db_thresh", type=float, default=0.3,
        help=(
            "DB binarisation threshold [0–1].  "
            "Lower → more (potentially noisy) boxes.  Default: 0.3."
        ),
    )
    engine_grp.add_argument(
        "--use_doc_orientation_classify", action="store_true",
        help="Predict and auto-correct page orientation (0/90/180/270°).",
    )
    engine_grp.add_argument(
        "--use_doc_unwarping", action="store_true",
        help="Correct perspective/warping before OCR.",
    )

    # ── Behaviour ─────────────────────────────────────────────────────────
    beh_grp = parser.add_argument_group("Behaviour")
    beh_grp.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip images already recorded in checkpoint.json inside output_dir.  "
            "Safe to use after an interrupted run."
        ),
    )
    beh_grp.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N images (dry-run / testing).  Default: no limit.",
    )
    beh_grp.add_argument(
        "--no_layout", action="store_true",
        help="Do NOT save layout.jpg (annotated bounding-box image) to save disk space.",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    console_logger.info("paddle_ocr_batch.py starting …")
    console_logger.info(f"  input_dir  : {args.input_dir}")
    console_logger.info(f"  output_dir : {args.output_dir}")
    console_logger.info(f"  min_size   : {args.min_size}px")
    console_logger.info(f"  workers    : {args.workers or 'auto (cpu_count)'}")
    console_logger.info(f"  device     : {args.device}")
    console_logger.info(f"  resume     : {args.resume}")
    console_logger.info(f"  limit      : {args.limit or 'none'}")

    run_batch(args)


if __name__ == "__main__":
    main()
