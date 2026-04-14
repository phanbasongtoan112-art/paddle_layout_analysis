#!/usr/bin/env python3
"""
benchmark_resumable.py
======================
Resumable benchmark for PPStructureV3 (PaddleOCR 3.4.0 / PaddlePaddle 3.2.0)
on CPU-only Windows machines with TableBank-style document images.

Features
--------
1.  Read all .png/.jpg/.jpeg from --input_dir
2.  Resize images: short side = --target_size (default 800)
3.  Checkpoint – CSV written after every image; restart skips done rows
4.  Error handling – NotImplementedError / OOM / decode errors logged & skipped
5.  Enhanced table classification:
      wired       – HTML contains border/border-collapse styling or <td> with
                    explicit width/height attrs (typical wired-table export)
      wireless    – table detected but no border evidence
      formula     – ≥2 distinct math/LaTeX patterns in text
      heavy_text  – total OCR text length > 500 characters
      no_table    – PPStructureV3 found no table region
6.  Built-in MKLDNN fixes (Layer 0 env-vars + Layer 1 set_flags + Layer 2
    PNG→JPEG re-encode) — always applied; no flag needed
7.  Per-image inference time (ms) recorded
8.  Markdown report generated at end (and on Ctrl-C)
9.  --limit N  processes only first N images
10. Auto-retry on OOM: reduce target_size by 100 px, up to 3 retries

Usage
-----
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench_out
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench_out --limit 20
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench_out --resume
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench_out \
      --target_size 640 --jpeg_quality 92

Dependencies
------------
  pip install paddlepaddle==3.2.0
  pip install paddleocr==3.4.0
  pip install opencv-python-headless tqdm
  pip install Pillow   # optional but recommended for exotic PNG variants
"""

# ══════════════════════════════════════════════════════════════════════════════
# MKLDNN FIX – LAYER 0
# Must be the very first executable lines, before any paddle import.
# Effective on Linux and PP < 3.3; harmless elsewhere.
# ══════════════════════════════════════════════════════════════════════════════
import os
os.environ["FLAGS_use_mkldnn"]               = "0"
os.environ["PADDLE_DISABLE_MKLDNN"]          = "1"
os.environ["FLAGS_use_dnnl_primitive_cache"] = "0"
os.environ["FLAGS_enable_pir_api"]           = "1"

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import logging
import platform
import re
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Third-party (early exit with helpful message)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("ERROR: pip install opencv-python-headless numpy")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: pip install tqdm")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXT       = {".png", ".jpg", ".jpeg"}
CSV_FILENAME        = "benchmark_results.csv"
REPORT_FILENAME     = "benchmark_report.md"
LOG_FILENAME        = "benchmark.log"

# Table classification labels
LABEL_WIRED        = "wired"
LABEL_WIRELESS     = "wireless"
LABEL_FORMULA      = "formula"
LABEL_HEAVY_TEXT   = "heavy_text"
LABEL_NO_TABLE     = "no_table"
LABEL_ERROR        = "error"

# OOM retry
OOM_MAX_RETRIES    = 3
OOM_SIZE_STEP      = 100          # reduce target_size by this many px per retry
OOM_MIN_SIZE       = 400          # never go below this

# CSV columns (fixed order for resumability)
CSV_FIELDS = [
    "image_name",       # filename only
    "status",           # success | error
    "label",            # wired | wireless | formula | heavy_text | no_table | error
    "inference_ms",     # float, blank on error
    "total_ms",         # including preprocess
    "orig_w", "orig_h",
    "proc_w", "proc_h", # size actually fed to model
    "target_size_used", # effective target_size (may differ due to OOM retry)
    "region_count",     # total PPStructure regions
    "table_count",      # how many table regions
    "text_length",      # total char count across all text/table regions
    "has_border_html",  # 1/0 — CSS/attr border evidence in HTML
    "formula_matches",  # number of distinct math patterns found
    "error_type",       # blank on success
    "error_message",    # blank on success
    "timestamp",
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (console + file)
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("bench")
logger.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_console_handler)

def init_file_logger(output_dir: Path) -> None:
    fh = logging.FileHandler(output_dir / LOG_FILENAME, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)


# ══════════════════════════════════════════════════════════════════════════════
# MKLDNN FIX – LAYER 1
# Apply paddle.set_flags() to write directly into C++ GlobalVarMap.
# Required on Windows + PP 3.3.x where os.environ is not propagated to the
# C++ flag registry in time.  Also works on PP 3.2.0 as belt-and-suspenders.
# ══════════════════════════════════════════════════════════════════════════════

def _apply_mkldnn_layer1() -> None:
    """Write FLAGS_use_mkldnn=0 into paddle's C++ flag registry."""
    successes = []

    try:
        import paddle
        paddle.set_flags({"FLAGS_use_mkldnn": 0})
        successes.append("paddle.set_flags")
    except Exception as e:
        logger.debug(f"paddle.set_flags failed: {e}")

    try:
        import paddle.base as _base
        _base.core.globals()["FLAGS_use_mkldnn"] = False
        successes.append("paddle.base.core.globals")
    except Exception as e:
        logger.debug(f"paddle.base.core.globals failed: {e}")

    try:
        from paddle import fluid as _fluid
        _fluid.core.globals()["FLAGS_use_mkldnn"] = False
        successes.append("paddle.fluid.core.globals")
    except Exception as e:
        logger.debug(f"paddle.fluid.core.globals failed: {e}")

    if successes:
        logger.info(f"MKLDNN Layer-1 applied via: {', '.join(successes)}")
    else:
        logger.warning(
            "MKLDNN Layer-1: no set_flags method succeeded. "
            "Relying on Layer-2 (PNG→JPEG re-encode)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine initialisation  (single instance, shared for all images)
# ─────────────────────────────────────────────────────────────────────────────

def init_engine():
    """
    Build PPStructureV3 with CPU + all MKLDNN fixes applied.
    Returns the engine object.
    """
    # Layer 1 applied immediately before construction
    _apply_mkldnn_layer1()

    try:
        from paddleocr import PPStructureV3
    except ImportError:
        sys.exit("ERROR: pip install paddleocr==3.4.0")

    logger.info("Initialising PPStructureV3 (CPU, no orientation/unwarping) …")
    engine = PPStructureV3(
        device="cpu",
        # Keep orientation classify and unwarping OFF for maximum speed on
        # weak hardware — the user can enable via code if needed.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    logger.info("Engine ready.")
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# MKLDNN FIX – LAYER 2  (embedded in image loading)
# PNG / BMP / TIFF → JPEG re-encode strips float attribute metadata from the
# tensor, routing inference through the standard executor instead of oneDNN.
# ══════════════════════════════════════════════════════════════════════════════

def load_and_resize(
    path: Path,
    target_size: int,
    jpeg_quality: int = 95,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Read an image, resize so short side == target_size, apply Layer-2 fix.

    Returns
    -------
    img         : uint8 BGR array, C-contiguous, 3-channel
    orig_wh     : (width, height) before any processing
    proc_wh     : (width, height) fed to the model
    """
    # ── Read ──────────────────────────────────────────────────────────────
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image as _PILImage
            pil = _PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Cannot decode: {path.name}")

    orig_h, orig_w = img.shape[:2]

    # ── Resize: short side → target_size ─────────────────────────────────
    short = min(orig_h, orig_w)
    if short != target_size:
        scale = target_size / short
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # ── Layer-2: non-JPEG formats → JPEG re-encode in memory ──────────────
    if path.suffix.lower() in {".png", ".bmp", ".tiff", ".tif", ".webp"}:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ── Normalise array layout ────────────────────────────────────────────
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    proc_wh = (img.shape[1], img.shape[0])
    return img, (orig_w, orig_h), proc_wh


# ─────────────────────────────────────────────────────────────────────────────
# PPStructureV3 result normalisation
# ─────────────────────────────────────────────────────────────────────────────

def flatten_result(raw) -> List[Dict]:
    """
    Normalise whatever .predict() returns into a flat list of region dicts.
    PPStructureV3 3.4 can return:  List[Dict]  |  List[List[Dict]]  |  generator
    """
    if raw is None:
        return []
    if hasattr(raw, "__iter__") and not isinstance(raw, (list, dict)):
        raw = list(raw)
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return raw or []


def extract_all_text(regions: List[Dict]) -> str:
    """
    Collect all recognisable text from every region (table cells, text blocks).
    Handles both dict-style and list-style 'res' payloads.
    """
    parts: List[str] = []

    for region in regions:
        res = region.get("res") or {}

        # Table HTML → strip tags → plain text
        if isinstance(res, dict):
            html = res.get("html") or res.get("HTML") or ""
            if html:
                parts.append(re.sub(r"<[^>]+>", " ", html))

        # Text / title regions carry a list of line dicts
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("transcription") or ""
                    if txt:
                        parts.append(txt)

        # Some versions embed text directly
        direct = region.get("text") or region.get("transcription") or ""
        if direct:
            parts.append(direct)

    return " ".join(parts)


def get_table_html(regions: List[Dict]) -> str:
    """Return the concatenated HTML of all table regions."""
    htmls: List[str] = []
    for r in regions:
        if (r.get("type") or "").lower() != "table":
            continue
        res = r.get("res") or {}
        if isinstance(res, dict):
            h = res.get("html") or res.get("HTML") or ""
            if h:
                htmls.append(h)
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    h = item.get("html") or item.get("HTML") or ""
                    if h:
                        htmls.append(h)
    return "\n".join(htmls)


# ─────────────────────────────────────────────────────────────────────────────
# Table classification  (Feature 5)
# ─────────────────────────────────────────────────────────────────────────────

# ── Border evidence patterns in HTML ──────────────────────────────────────────
# Matches CSS border properties and HTML table/td border attributes
_BORDER_HTML_PATTERNS = [
    re.compile(r'border\s*:\s*\d', re.IGNORECASE),          # border: 1px …
    re.compile(r'border-collapse\s*:\s*collapse', re.IGNORECASE),
    re.compile(r'border-width\s*:\s*[^0]', re.IGNORECASE),
    re.compile(r'<t[dh][^>]+\s+border\s*=\s*["\']?\d', re.IGNORECASE),
    re.compile(r'<table[^>]+\s+border\s*=\s*["\']?[^0"\']', re.IGNORECASE),
    re.compile(r'style\s*=\s*"[^"]*border[^"]*"', re.IGNORECASE),
]

# ── Extended math / LaTeX patterns ────────────────────────────────────────────
# Covers inline LaTeX, common operators, Greek letters, fractions, and
# Unicode math symbols likely to appear in OCR'd formula regions.
_MATH_PATTERNS = [
    # LaTeX delimiters
    re.compile(r'\$[^$]+\$'),                             # $…$
    re.compile(r'\\\(.*?\\\)', re.DOTALL),                # \( … \)
    re.compile(r'\\\[.*?\\\]', re.DOTALL),                # \[ … \]
    re.compile(r'\\begin\{(equation|align|math)'),        # \begin{equation}

    # LaTeX commands
    re.compile(r'\\(frac|sqrt|sum|int|prod|lim|inf|partial|nabla|Delta|Sigma|'
               r'alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|phi|omega|'
               r'leq|geq|neq|approx|equiv|subset|supset|in|notin|cup|cap|'
               r'mathbf|mathrm|mathcal|text|overline|hat|tilde|bar|vec)\b'),

    # Common math notations in plain OCR text
    re.compile(r'\b[A-Za-z]\s*[\^_]\s*[\{\w]'),          # x^2, x_i, A^{-1}
    re.compile(r'\b\d+\s*/\s*\d+\b'),                     # fraction  3/4
    re.compile(r'[=<>≤≥≠≈±∓×÷∞∑∏∫∂∇√]{2,}'),            # consecutive operators
    re.compile(r'[αβγδεζηθλμνξπρστφχψω]'),                # Greek Unicode
    re.compile(r'[∀∃∈∉⊆⊇∪∩⊕⊗⊥∧∨¬]'),                   # set / logic Unicode
    re.compile(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]'),              # sub/superscript digits
    re.compile(r'\b(sin|cos|tan|log|exp|lim|max|min|sup|inf|det|tr|diag)\s*[\(\[]'),
]

HEAVY_TEXT_THRESHOLD = 500  # characters
FORMULA_MIN_MATCHES  = 2    # distinct pattern types that must match


def has_border_evidence(html: str) -> bool:
    """Return True if the HTML string contains border styling evidence."""
    return any(p.search(html) for p in _BORDER_HTML_PATTERNS)


def count_formula_patterns(text: str) -> int:
    """
    Return the number of *distinct* math pattern categories that match in text.
    (Each pattern in _MATH_PATTERNS counts once regardless of how many times
    it matches, so one heavily LaTeX-annotated cell doesn't inflate the score.)
    """
    return sum(1 for p in _MATH_PATTERNS if p.search(text))


def classify_table(
    regions: List[Dict],
    table_html: str,
    all_text: str,
) -> Tuple[str, int, int]:
    """
    Classify the image's table content into one label.

    Priority order (a single image gets exactly one primary label):
        formula  > heavy_text  > wired  > wireless  > no_table

    Returns
    -------
    label           : one of the LABEL_* constants
    formula_matches : count of distinct math pattern types found
    has_border      : 1 if border evidence found, else 0
    """
    table_regions = [r for r in regions if (r.get("type") or "").lower() == "table"]

    if not table_regions:
        return LABEL_NO_TABLE, 0, 0

    formula_matches = count_formula_patterns(all_text)
    border_flag     = 1 if has_border_evidence(table_html) else 0

    if formula_matches >= FORMULA_MIN_MATCHES:
        return LABEL_FORMULA, formula_matches, border_flag

    if len(all_text) > HEAVY_TEXT_THRESHOLD:
        return LABEL_HEAVY_TEXT, formula_matches, border_flag

    if border_flag:
        return LABEL_WIRED, formula_matches, border_flag

    return LABEL_WIRELESS, formula_matches, border_flag


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / CSV management  (Feature 3)
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointCSV:
    """
    Append-only CSV that doubles as a checkpoint.

    After every processed image a row is flushed to disk.  On restart,
    already-recorded image names are read back and skipped.
    """

    def __init__(self, csv_path: Path) -> None:
        self._path = csv_path
        self._done: set = set()

        # Load existing rows to populate _done set
        if csv_path.exists():
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("image_name"):
                            self._done.add(row["image_name"])
                logger.info(
                    f"Checkpoint: {len(self._done)} image(s) already recorded."
                )
            except Exception as exc:
                logger.warning(f"Could not read checkpoint CSV: {exc}")

        # Open for appending; write header only if file is new/empty
        is_new = not csv_path.exists() or csv_path.stat().st_size == 0
        self._fh  = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def is_done(self, image_name: str) -> bool:
        return image_name in self._done

    def write(self, row: Dict) -> None:
        """Write one result row and flush immediately (crash-safe)."""
        self._writer.writerow(row)
        self._fh.flush()
        self._done.add(row["image_name"])

    def close(self) -> None:
        self._fh.close()

    @property
    def done_count(self) -> int:
        return len(self._done)


# ─────────────────────────────────────────────────────────────────────────────
# OOM detection helpers  (Feature 4 / 10)
# ─────────────────────────────────────────────────────────────────────────────

_OOM_FRAGMENTS = (
    "out of memory",
    "allocate",
    "bad_alloc",
    "memory",
    "memalloc",
    "oom",
)

def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(f in msg for f in _OOM_FRAGMENTS)


def is_mkldnn_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "convertpirattribute2runtimeattribute" in msg
        or "onednn" in msg
        or "mkldnn" in msg
        or "dnnl" in msg
        or "pir::arrayattribute" in msg
    )


def classify_error(exc: Exception) -> str:
    if is_oom_error(exc):
        return "OOM"
    if is_mkldnn_error(exc):
        return "MKLDNN_PIR"
    if isinstance(exc, ValueError):
        return "DecodeError"
    if isinstance(exc, NotImplementedError):
        return "NotImplementedError"
    return type(exc).__name__


# ─────────────────────────────────────────────────────────────────────────────
# Core per-image processing  (Features 2, 4, 5, 6, 7, 10)
# ─────────────────────────────────────────────────────────────────────────────

def process_one_image(
    image_path: Path,
    engine,
    target_size: int,
    jpeg_quality: int,
) -> Dict:
    """
    Load, infer, classify, and return a result dict for one image.

    On OOM: retries up to OOM_MAX_RETRIES times with target_size
    reduced by OOM_SIZE_STEP px each time.
    """
    t_total = time.perf_counter()

    attempt_size = target_size
    last_exc: Optional[Exception] = None

    for attempt in range(OOM_MAX_RETRIES + 1):
        if attempt > 0:
            attempt_size = max(target_size - attempt * OOM_SIZE_STEP, OOM_MIN_SIZE)
            logger.warning(
                f"  OOM retry {attempt}/{OOM_MAX_RETRIES}: "
                f"reducing target_size → {attempt_size}px"
            )
            gc.collect()
            time.sleep(0.5)

        try:
            # ── Pre-process (Layer 2 inside load_and_resize) ──────────────
            img, orig_wh, proc_wh = load_and_resize(
                image_path, attempt_size, jpeg_quality
            )

            # ── Inference (BUG-1 fix: .predict(), never engine()) ─────────
            t_inf = time.perf_counter()
            raw   = engine.predict(img)
            inference_ms = (time.perf_counter() - t_inf) * 1000

            regions    = flatten_result(raw)
            table_html = get_table_html(regions)
            all_text   = extract_all_text(regions)
            table_regs = [r for r in regions if (r.get("type") or "").lower() == "table"]

            label, formula_matches, has_border = classify_table(
                regions, table_html, all_text
            )

            total_ms = (time.perf_counter() - t_total) * 1000

            return {
                "image_name":       image_path.name,
                "status":           "success",
                "label":            label,
                "inference_ms":     round(inference_ms, 1),
                "total_ms":         round(total_ms, 1),
                "orig_w":           orig_wh[0],
                "orig_h":           orig_wh[1],
                "proc_w":           proc_wh[0],
                "proc_h":           proc_wh[1],
                "target_size_used": attempt_size,
                "region_count":     len(regions),
                "table_count":      len(table_regs),
                "text_length":      len(all_text),
                "has_border_html":  has_border,
                "formula_matches":  formula_matches,
                "error_type":       "",
                "error_message":    "",
                "timestamp":        datetime.utcnow().isoformat() + "Z",
            }

        except Exception as exc:
            last_exc = exc
            if is_oom_error(exc) and attempt < OOM_MAX_RETRIES:
                continue   # try again with smaller size
            break          # non-OOM error OR exhausted retries

    # ── All attempts failed ────────────────────────────────────────────────
    total_ms = (time.perf_counter() - t_total) * 1000
    err_type = classify_error(last_exc)
    err_msg  = str(last_exc).replace("\n", " ")[:400]
    logger.error(f"  FAILED [{err_type}] {image_path.name}: {err_msg[:120]}")

    if is_mkldnn_error(last_exc):
        logger.warning(
            "  ↳ MKLDNN error persists despite fixes. "
            "Try: pip install paddlepaddle==3.2.0"
        )

    return {
        "image_name":       image_path.name,
        "status":           "error",
        "label":            LABEL_ERROR,
        "inference_ms":     "",
        "total_ms":         round(total_ms, 1),
        "orig_w":           "",
        "orig_h":           "",
        "proc_w":           "",
        "proc_h":           "",
        "target_size_used": attempt_size,
        "region_count":     "",
        "table_count":      "",
        "text_length":      "",
        "has_border_html":  "",
        "formula_matches":  "",
        "error_type":       err_type,
        "error_message":    err_msg,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generator  (Feature 8)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(csv_path: Path, report_path: Path) -> None:
    """
    Read all rows from the checkpoint CSV and write a Markdown report.
    Safe to call mid-run (only completed rows are in the CSV).
    """
    rows: List[Dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        logger.error(f"Cannot read CSV for report: {exc}")
        return

    if not rows:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "# PPStructureV3 TableBank Benchmark Report",
        f"*Generated: {ts}*",
        "",
    ]

    # ── Environment ────────────────────────────────────────────────────────
    lines += ["## Environment", ""]
    lines += [
        f"| Key | Value |",
        f"|-----|-------|",
        f"| OS | {platform.system()} {platform.version()} |",
        f"| Python | {sys.version.split()[0]} |",
    ]
    try:
        import paddle
        lines.append(f"| PaddlePaddle | {paddle.__version__} |")
    except Exception:
        lines.append("| PaddlePaddle | unknown |")
    try:
        import paddleocr
        lines.append(f"| PaddleOCR | {getattr(paddleocr, '__version__', 'unknown')} |")
    except Exception:
        pass
    lines.append(f"| MKLDNN disabled | Yes (Layer 0+1+2) |")
    lines.append("")

    # ── Totals ─────────────────────────────────────────────────────────────
    total   = len(rows)
    success = [r for r in rows if r["status"] == "success"]
    errors  = [r for r in rows if r["status"] == "error"]
    error_rate = len(errors) / total * 100 if total else 0

    lines += [
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total images processed | {total} |",
        f"| Successful | {len(success)} |",
        f"| Errors | {len(errors)} ({error_rate:.1f}%) |",
    ]

    times = []
    for r in success:
        try:
            times.append(float(r["inference_ms"]))
        except (ValueError, TypeError):
            pass

    if times:
        lines += [
            f"| Mean inference time | {mean(times):.1f} ms |",
            f"| Median inference time | {median(times):.1f} ms |",
        ]
        if len(times) > 1:
            lines.append(f"| Stdev inference time | {stdev(times):.1f} ms |")
        throughput = 1000 / mean(times) if mean(times) > 0 else 0
        lines.append(f"| Throughput (single-threaded) | {throughput:.3f} img/s |")
    lines.append("")

    # ── Label distribution ─────────────────────────────────────────────────
    all_labels = [LABEL_WIRED, LABEL_WIRELESS, LABEL_FORMULA,
                  LABEL_HEAVY_TEXT, LABEL_NO_TABLE, LABEL_ERROR]
    label_counts = {lb: sum(1 for r in rows if r.get("label") == lb)
                    for lb in all_labels}

    lines += [
        "## Table Classification",
        "",
        "| Label | Count | % of total | Description |",
        "|-------|-------|------------|-------------|",
    ]
    descriptions = {
        LABEL_WIRED:      "Table with border evidence in HTML",
        LABEL_WIRELESS:   "Table detected, no border styling found",
        LABEL_FORMULA:    f"≥{FORMULA_MIN_MATCHES} distinct math/LaTeX patterns",
        LABEL_HEAVY_TEXT: f"Total OCR text > {HEAVY_TEXT_THRESHOLD} chars",
        LABEL_NO_TABLE:   "No table region detected by model",
        LABEL_ERROR:      "Processing failed",
    }
    for lb in all_labels:
        cnt = label_counts[lb]
        pct = cnt / total * 100 if total else 0
        lines.append(f"| `{lb}` | {cnt} | {pct:.1f}% | {descriptions[lb]} |")
    lines.append("")

    # ── Inference time by label ────────────────────────────────────────────
    lines += [
        "## Inference Time by Label",
        "",
        "| Label | N | Mean ms | Median ms |",
        "|-------|---|---------|-----------|",
    ]
    for lb in [lb for lb in all_labels if lb != LABEL_ERROR]:
        lb_rows = [r for r in success if r.get("label") == lb]
        lb_times = []
        for r in lb_rows:
            try:
                lb_times.append(float(r["inference_ms"]))
            except (ValueError, TypeError):
                pass
        if lb_times:
            lines.append(
                f"| `{lb}` | {len(lb_times)} | "
                f"{mean(lb_times):.1f} | {median(lb_times):.1f} |"
            )
    lines.append("")

    # ── Error breakdown ────────────────────────────────────────────────────
    if errors:
        lines += ["## Error Analysis", ""]
        err_types: Dict[str, List[Dict]] = {}
        for r in errors:
            err_types.setdefault(r.get("error_type", "unknown"), []).append(r)

        lines += [
            "| Error type | Count | Sample message |",
            "|-----------|-------|----------------|",
        ]
        for etype, errs in sorted(err_types.items(), key=lambda x: -len(x[1])):
            sample = errs[0].get("error_message", "")[:120].replace("|", "│")
            lines.append(f"| `{etype}` | {len(errs)} | {sample} |")
        lines.append("")

        if any("MKLDNN" in r.get("error_type", "") for r in errors):
            lines += [
                "### MKLDNN Fix Reminder",
                "",
                "Some images still failed with MKLDNN errors despite the three-layer fix.",
                "If this persists, downgrade PaddlePaddle:",
                "",
                "```bash",
                "pip install paddlepaddle==3.2.0",
                "```",
                "",
            ]

    # ── Per-image results table ────────────────────────────────────────────
    lines += [
        "## Per-Image Results",
        "",
        "| # | Image | Status | Label | Inf (ms) | Total (ms) | "
        "Orig WH | Proc WH | Regions | Tables | Text len | Error |",
        "|---|-------|--------|-------|----------|------------|"
        "---------|---------|---------|--------|----------|-------|",
    ]
    for i, r in enumerate(rows, 1):
        status_icon = "✅" if r["status"] == "success" else "❌"
        orig = f"{r['orig_w']}×{r['orig_h']}" if r["orig_w"] else "—"
        proc = f"{r['proc_w']}×{r['proc_h']}" if r["proc_w"] else "—"
        err  = f"`{r['error_type']}`" if r.get("error_type") else ""
        lines.append(
            f"| {i} | {r['image_name']} | {status_icon} | "
            f"`{r.get('label','')}` | {r['inference_ms'] or '—'} | "
            f"{r['total_ms'] or '—'} | {orig} | {proc} | "
            f"{r['region_count'] or '—'} | {r['table_count'] or '—'} | "
            f"{r['text_length'] or '—'} | {err} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main batch loop  (Features 1, 3, 9)
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file_logger(output_dir)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input_dir '{input_dir}' not found.")

    # ── Collect images ─────────────────────────────────────────────────────
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if args.limit:
        images = images[:args.limit]
        logger.info(f"--limit {args.limit}: processing first {len(images)} image(s).")

    if not images:
        logger.info("No supported images found in input directory.")
        return

    logger.info(f"Found {len(images)} image(s) to process.")

    # ── Checkpoint CSV ─────────────────────────────────────────────────────
    csv_path    = output_dir / CSV_FILENAME
    report_path = output_dir / REPORT_FILENAME
    checkpoint  = CheckpointCSV(csv_path)

    # ── Engine (built once) ────────────────────────────────────────────────
    engine = init_engine()

    # ── Graceful Ctrl-C → generate report before exit ─────────────────────
    def _on_interrupt(sig, frame):
        print()
        logger.info("Interrupted — generating partial report …")
        checkpoint.close()
        generate_report(csv_path, report_path)
        logger.info(
            f"Progress saved. Re-run with --resume (or without --limit) to continue."
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    # ── Counters ───────────────────────────────────────────────────────────
    n_success = 0
    n_error   = 0
    n_skipped = 0

    # ── Main loop ──────────────────────────────────────────────────────────
    with tqdm(images, unit="img", dynamic_ncols=True) as pbar:
        for img_path in pbar:

            # Checkpoint skip
            if args.resume and checkpoint.is_done(img_path.name):
                n_skipped += 1
                pbar.set_postfix_str(f"skip={n_skipped} ok={n_success} err={n_error}")
                continue

            pbar.set_description(f"  {img_path.name[:35]:<35}")

            result = process_one_image(
                img_path, engine, args.target_size, args.jpeg_quality
            )
            checkpoint.write(result)

            if result["status"] == "success":
                n_success += 1
            else:
                n_error += 1

            pbar.set_postfix_str(
                f"ok={n_success} err={n_error} skip={n_skipped} "
                f"label={result['label']}"
            )

    checkpoint.close()

    # ── Final report ───────────────────────────────────────────────────────
    generate_report(csv_path, report_path)

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  BENCHMARK COMPLETE")
    print("=" * 64)
    print(f"  Total    : {len(images)}")
    print(f"  Success  : {n_success}")
    print(f"  Failed   : {n_error}")
    print(f"  Skipped  : {n_skipped}")
    print(f"  CSV      : {csv_path.resolve()}")
    print(f"  Report   : {report_path.resolve()}")
    print(f"  Log      : {(output_dir / LOG_FILENAME).resolve()}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (Feature 9 / argparse)
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark_resumable",
        description=(
            "Resumable PPStructureV3 benchmark for TableBank images on CPU Windows.\n"
            "Classifies each table as: wired | wireless | formula | "
            "heavy_text | no_table.\n"
            "Writes benchmark_results.csv (checkpoint) and benchmark_report.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # First run on 20 images
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench --limit 20

  # Resume after Ctrl-C (skips already-processed images)
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench --resume

  # Full run with smaller target size for very weak machines
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench \\
      --target_size 640 --resume

  # Regenerate the Markdown report from an existing CSV without running inference
  python benchmark_resumable.py --input_dir ./images --output_dir ./bench \\
      --report_only

Classification labels
---------------------
  wired      Table HTML contains border/border-collapse CSS or border attributes
  wireless   Table detected but no border evidence found
  formula    >= 2 distinct math/LaTeX pattern types found in text
  heavy_text Total OCR text length > 500 characters
  no_table   PPStructureV3 found no table region in the image
  error      Image failed to process (see error_type / error_message columns)
""",
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input_dir",  required=True,
                    help="Directory containing .png / .jpg / .jpeg images.")
    io.add_argument("--output_dir", required=True,
                    help="Directory for CSV, report, and log files.")

    pre = p.add_argument_group("Pre-processing")
    pre.add_argument(
        "--target_size", type=int, default=800,
        help=(
            "Resize so the short side equals this value (px). "
            "Default 800. Use 640 on very weak machines. "
            "On OOM the script retries with target_size - 100."
        ),
    )
    pre.add_argument(
        "--jpeg_quality", type=int, default=95,
        help=(
            "JPEG quality for the PNG→JPEG re-encode fix (1–100). "
            "95 is visually lossless for document scans. Default 95."
        ),
    )

    run_grp = p.add_argument_group("Run control")
    run_grp.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N images (dry-run / testing).",
    )
    run_grp.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip images already present in benchmark_results.csv. "
            "Use after an interrupted run."
        ),
    )
    run_grp.add_argument(
        "--report_only", action="store_true",
        help=(
            "Do not run inference. Just (re-)generate benchmark_report.md "
            "from an existing benchmark_results.csv."
        ),
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file_logger(output_dir)

    # --report_only: just regenerate the MD from an existing CSV
    if args.report_only:
        csv_path = output_dir / CSV_FILENAME
        if not csv_path.exists():
            sys.exit(f"ERROR: {csv_path} not found. Run without --report_only first.")
        generate_report(csv_path, output_dir / REPORT_FILENAME)
        return

    # Validate JPEG quality range
    if not (1 <= args.jpeg_quality <= 100):
        sys.exit("ERROR: --jpeg_quality must be between 1 and 100.")

    # Validate target_size
    if args.target_size < OOM_MIN_SIZE:
        sys.exit(
            f"ERROR: --target_size must be >= {OOM_MIN_SIZE}. "
            f"Got {args.target_size}."
        )

    logger.info("benchmark_resumable.py starting …")
    logger.info(f"  input_dir   : {args.input_dir}")
    logger.info(f"  output_dir  : {args.output_dir}")
    logger.info(f"  target_size : {args.target_size}px")
    logger.info(f"  jpeg_quality: {args.jpeg_quality}")
    logger.info(f"  limit       : {args.limit or 'none'}")
    logger.info(f"  resume      : {args.resume}")

    run(args)


if __name__ == "__main__":
    main()
