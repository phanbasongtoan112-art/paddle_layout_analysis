#!/usr/bin/env python3
"""
benchmark_ppstructure.py
========================
Benchmark + diagnostic script for PaddleOCR PPStructureV3.

Diagnoses and works around two known errors in PaddlePaddle 3.x / PaddleOCR 3.4:

  BUG-1  TypeError: 'PPStructureV3' object is not callable
         Cause:  PPStructureV3 dropped __call__; use .predict() instead.
         Fix:    Always call engine.predict(img).

  BUG-2  NotImplementedError: ConvertPirAttribute2RuntimeAttribute
             not support [pir::ArrayAttribute<pir::DoubleAttribute>]
         Cause:  oneDNN (MKLDNN) doesn't support the new PIR operator
                 attributes used by PPStructureV3 3.4, especially on PNG
                 images whose tensors have float64 attributes.
         Fix:    Set FLAGS_use_mkldnn=0 (and related env vars) BEFORE
                 importing paddle, OR initialise the engine with
                 enable_mkldnn=False.

Outputs:
  benchmark_results.csv  – per-image row: name, status, time_ms, error_type, …
  benchmark_report.md    – human-readable Markdown summary table + statistics

Usage:
    python benchmark_ppstructure.py --input_dir ./images --output_dir ./bench_out
    python benchmark_ppstructure.py --input_dir ./images --output_dir ./bench_out --limit 20
    python benchmark_ppstructure.py --input_dir ./images --output_dir ./bench_out --strategy all
"""

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Disable oneDNN/MKLDNN *before* any paddle import.
# These env vars must be set before the paddle .so is loaded.
# This is the primary fix for BUG-2.
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ["FLAGS_use_mkldnn"]          = "0"   # disable MKLDNN globally
os.environ["FLAGS_enable_pir_api"]      = "1"   # keep PIR enabled (required by 3.x)
os.environ["PADDLE_DISABLE_MKLDNN"]     = "1"   # belt-and-suspenders alias
os.environ["FLAGS_use_dnnl_primitive_cache"] = "0"

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import io
import platform
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")   # suppress paddle verbose warnings

# ─────────────────────────────────────────────────────────────────────────────
# Third-party (fail fast with helpful messages)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("pip install opencv-python-headless")

try:
    import numpy as np
except ImportError:
    sys.exit("pip install numpy")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed – memory tracking disabled.  pip install psutil")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Error taxonomy — maps exception substrings to short labels shown in CSV
ERROR_TAXONOMY: List[Tuple[str, str]] = [
    ("object is not callable",                          "BUG-1:NotCallable"),
    ("ConvertPirAttribute2RuntimeAttribute",            "BUG-2:MKLDNN_PIR"),
    ("not support [pir::ArrayAttribute",                "BUG-2:MKLDNN_PIR"),
    ("mkldnn",                                          "BUG-2:MKLDNN_PIR"),
    ("dnnl",                                            "BUG-2:MKLDNN_PIR"),
    ("imread returned None",                            "IMG:CorruptOrMissing"),
    ("out of memory",                                   "OOM"),
    ("cuda",                                            "GPU:CUDAError"),
    ("NotImplementedError",                             "NotImplemented"),
    ("AttributeError",                                  "AttributeError"),
    ("ValueError",                                      "ValueError"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def rss_mb() -> float:
    """Return current process RSS in MiB, or 0 if psutil is unavailable."""
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


@contextmanager
def track_memory():
    """Context manager that yields a dict filled with before/after/delta RSS."""
    info: Dict[str, float] = {"before": 0.0, "after": 0.0, "delta": 0.0}
    info["before"] = rss_mb()
    try:
        yield info
    finally:
        info["after"] = rss_mb()
        info["delta"] = info["after"] - info["before"]


# ─────────────────────────────────────────────────────────────────────────────
# Classify error strings into short labels
# ─────────────────────────────────────────────────────────────────────────────

def classify_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    for fragment, label in ERROR_TAXONOMY:
        if fragment.lower() in msg:
            return label
    return f"{type(exc).__name__}"


# ─────────────────────────────────────────────────────────────────────────────
# Image loading with pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, min_size: int = 800) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    """
    Read an image, upscale if short side < min_size (LANCZOS4), return
    (bgr_array, original_wh, resized_wh).

    PNG images are loaded via cv2 (which always returns uint8 BGR arrays), so
    the BUG-2 trigger is actually inside Paddle's op attrs, not in numpy dtype.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # Try PIL as fallback (handles some exotic PNG sub-formats)
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"cv2.imread returned None – file may be corrupt: {path}")

    orig_h, orig_w = img.shape[:2]
    orig_wh = (orig_w, orig_h)

    short = min(orig_h, orig_w)
    if short < min_size:
        scale = min_size / short
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    resized_wh = (img.shape[1], img.shape[0])
    return img, orig_wh, resized_wh


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory – multiple strategies to try
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(strategy: str, device: str = "cpu"):
    """
    Build a PPStructureV3 engine using the specified strategy.

    Strategies
    ----------
    safe       – disable mkldnn, no orientation/unwarping (fastest, most compat)
    standard   – like safe but with orientation classify
    full       – all features on
    mkldnn_on  – deliberately enables mkldnn (to reproduce BUG-2)
    callable   – uses safe config but tests __call__ instead of .predict()
                 (to reproduce BUG-1)
    """
    from paddleocr import PPStructureV3  # import here; env vars already set above

    base_kwargs = dict(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    if strategy == "safe":
        # Primary recommended config: mkldnn off (env already set), predict() API
        return PPStructureV3(**base_kwargs), "predict"

    elif strategy == "standard":
        return PPStructureV3(
            **{**base_kwargs, "use_doc_orientation_classify": True}
        ), "predict"

    elif strategy == "full":
        return PPStructureV3(
            **{**base_kwargs,
               "use_doc_orientation_classify": True,
               "use_doc_unwarping": True}
        ), "predict"

    elif strategy == "mkldnn_on":
        # Reproduce BUG-2: temporarily re-enable mkldnn for this engine
        # (will fail on many PNG images – documented in output)
        os.environ["FLAGS_use_mkldnn"] = "1"
        try:
            engine = PPStructureV3(**base_kwargs)
        finally:
            os.environ["FLAGS_use_mkldnn"] = "0"
        return engine, "predict"

    elif strategy == "callable":
        # Reproduce BUG-1: try engine(img) instead of engine.predict(img)
        return PPStructureV3(**base_kwargs), "call"

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def run_inference(engine, call_style: str, img: np.ndarray) -> List[Dict]:
    """
    Run inference using either .predict() [correct] or __call__ [buggy].

    call_style: 'predict' | 'call'
    """
    if call_style == "predict":
        # ── CORRECT API for PaddleOCR 3.x ────────────────────────────────
        return engine.predict(img)
    elif call_style == "call":
        # ── BUG-1 reproduction: will raise TypeError in PaddleOCR 3.x ────
        return engine(img)   # type: ignore[operator]
    else:
        raise ValueError(f"Unknown call_style: {call_style!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-image benchmark run
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    image_path: Path,
    engine,
    call_style: str,
    min_size: int,
    warmup: bool = False,
) -> Dict[str, Any]:
    """
    Run one image through the engine and return a result dict.

    Returns dict with keys:
      image_name, status, inference_time_ms, preprocess_time_ms,
      total_time_ms, memory_delta_mb, region_count, has_table,
      error_type, error_message, image_format, orig_w, orig_h,
      resized_w, resized_h, warmup
    """
    rec: Dict[str, Any] = {
        "image_name":         image_path.name,
        "image_format":       image_path.suffix.lower().lstrip("."),
        "status":             "pending",
        "inference_time_ms":  None,
        "preprocess_time_ms": None,
        "total_time_ms":      None,
        "memory_delta_mb":    None,
        "region_count":       None,
        "has_table":          None,
        "error_type":         "",
        "error_message":      "",
        "orig_w":             None,
        "orig_h":             None,
        "resized_w":          None,
        "resized_h":          None,
        "warmup":             warmup,
    }

    t_total_start = time.perf_counter()

    try:
        # ── Pre-processing ────────────────────────────────────────────────
        t_pre = time.perf_counter()
        img, orig_wh, resized_wh = load_image(image_path, min_size)
        rec["preprocess_time_ms"] = round((time.perf_counter() - t_pre) * 1000, 2)
        rec["orig_w"], rec["orig_h"]       = orig_wh
        rec["resized_w"], rec["resized_h"] = resized_wh

        # ── Inference (with memory tracking) ─────────────────────────────
        with track_memory() as mem:
            t_inf = time.perf_counter()
            result = run_inference(engine, call_style, img)
            rec["inference_time_ms"] = round((time.perf_counter() - t_inf) * 1000, 2)
        rec["memory_delta_mb"] = round(mem["delta"], 2)

        # ── Parse result ──────────────────────────────────────────────────
        if result is None:
            result = []

        # PPStructureV3.predict() may return a generator or a list
        if hasattr(result, "__iter__") and not isinstance(result, (list, dict)):
            result = list(result)

        # Flatten: predict() can return [[region, …]] or [region, …]
        if result and isinstance(result[0], list):
            result = result[0]

        rec["region_count"] = len(result)
        rec["has_table"]    = any(
            (r.get("type") or "").lower() == "table" for r in result
        )
        rec["status"] = "success"

    except Exception as exc:
        rec["status"]        = "fail"
        rec["error_type"]    = classify_error(exc)
        rec["error_message"] = str(exc).replace("\n", " ")[:300]

    rec["total_time_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Multi-strategy benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = ["safe", "standard", "full", "mkldnn_on", "callable"]


def run_benchmark(args: argparse.Namespace) -> List[Dict]:
    """
    Run the benchmark for all requested strategies and return all result rows.
    """
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images ────────────────────────────────────────────────────
    images = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if args.limit:
        images = images[:args.limit]

    if not images:
        print(f"[ERROR] No supported images in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s).")

    # ── Resolve strategies ────────────────────────────────────────────────
    if args.strategy == "all":
        strategies = STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategy.split(",")]

    # ── Detect device ─────────────────────────────────────────────────────
    device = _detect_device(args.device)
    print(f"Device: {device.upper()}")

    all_rows: List[Dict] = []

    for strategy in strategies:
        print(f"\n{'─'*60}")
        print(f"  Strategy: {strategy}")
        print(f"{'─'*60}")

        # ── Build engine for this strategy ────────────────────────────────
        try:
            engine, call_style = build_engine(strategy, device)
            print(f"  Engine ready (call_style={call_style})")
        except Exception as exc:
            print(f"  [SKIP] Could not build engine for '{strategy}': {exc}")
            continue

        # ── Optional warmup pass (first image, result discarded) ──────────
        if args.warmup and images:
            print(f"  Warming up on {images[0].name} …")
            benchmark_one(images[0], engine, call_style,
                          args.min_size, warmup=True)

        # ── Main benchmark loop ───────────────────────────────────────────
        rows: List[Dict] = []
        with tqdm(images, desc=f"  [{strategy}]", unit="img") as pbar:
            for img_path in pbar:
                row = benchmark_one(img_path, engine, call_style, args.min_size)
                row["strategy"] = strategy
                rows.append(row)
                all_rows.append(row)

                status_char = "✓" if row["status"] == "success" else "✗"
                pbar.set_postfix_str(
                    f"{status_char} {row['image_name']} "
                    f"({row['inference_time_ms'] or '—'}ms)"
                )

        # ── Per-strategy mini-summary ─────────────────────────────────────
        success = [r for r in rows if r["status"] == "success"]
        fail    = [r for r in rows if r["status"] == "fail"]
        print(f"\n  Results: {len(success)} success / {len(fail)} fail")
        if fail:
            by_type: Dict[str, int] = {}
            for r in fail:
                by_type[r["error_type"]] = by_type.get(r["error_type"], 0) + 1
            for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {etype}: {count} image(s)")
        if success:
            times = [r["inference_time_ms"] for r in success]
            print(f"  Inference time:  mean={mean(times):.1f}ms  "
                  f"median={median(times):.1f}ms  "
                  f"{'stdev='+str(round(stdev(times),1))+'ms' if len(times)>1 else ''}")

        # Free engine to recover memory between strategies
        del engine
        gc.collect()

    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "strategy", "image_name", "image_format", "status",
    "inference_time_ms", "preprocess_time_ms", "total_time_ms",
    "memory_delta_mb", "region_count", "has_table",
    "orig_w", "orig_h", "resized_w", "resized_h",
    "error_type", "error_message", "warmup",
]


def write_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written → {path}")


def write_markdown(rows: List[Dict], path: Path, env_info: Dict) -> None:
    """
    Write a Markdown report with:
    - Environment summary
    - Per-strategy statistics table
    - Full per-image results table
    - Error analysis section
    - Fix recommendations
    """
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "# PPStructureV3 Benchmark Report",
        f"*Generated: {ts}*",
        "",
        "## Environment",
        "| Key | Value |",
        "|-----|-------|",
    ]
    for k, v in env_info.items():
        lines.append(f"| {k} | {v} |")

    # ── Per-strategy stats table ───────────────────────────────────────────
    lines += ["", "## Strategy Summary", ""]
    lines.append(
        "| Strategy | Total | Success | Fail | Mean ms | Median ms | "
        "Stdev ms | Tables found | Error types |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    strategies_seen = list(dict.fromkeys(r["strategy"] for r in rows))
    for strat in strategies_seen:
        strat_rows = [r for r in rows if r["strategy"] == strat and not r["warmup"]]
        total   = len(strat_rows)
        success = [r for r in strat_rows if r["status"] == "success"]
        fail    = [r for r in strat_rows if r["status"] == "fail"]
        n_tables = sum(1 for r in success if r.get("has_table"))

        times = [r["inference_time_ms"] for r in success if r["inference_time_ms"]]
        mean_t   = f"{mean(times):.1f}"   if times else "—"
        median_t = f"{median(times):.1f}" if times else "—"
        stdev_t  = f"{stdev(times):.1f}"  if len(times) > 1 else "—"

        err_types = list(dict.fromkeys(r["error_type"] for r in fail if r["error_type"]))
        err_str   = ", ".join(f"`{e}`" for e in err_types) if err_types else "—"

        lines.append(
            f"| `{strat}` | {total} | {len(success)} | {len(fail)} | "
            f"{mean_t} | {median_t} | {stdev_t} | {n_tables} | {err_str} |"
        )

    # ── Per-image results table (non-warmup rows) ──────────────────────────
    lines += ["", "## Per-Image Results", ""]
    lines.append(
        "| # | Strategy | Image | Format | Status | Inf (ms) | Total (ms) | "
        "Mem ΔMiB | Regions | Table | Error |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    display_rows = [r for r in rows if not r["warmup"]]
    for i, r in enumerate(display_rows, 1):
        status_icon = "✅" if r["status"] == "success" else "❌"
        inf_ms  = r["inference_time_ms"]  or "—"
        tot_ms  = r["total_time_ms"]      or "—"
        mem_mb  = r["memory_delta_mb"]    if r["memory_delta_mb"] is not None else "—"
        regions = r["region_count"]       if r["region_count"] is not None else "—"
        table   = ("✓" if r["has_table"] else "✗") if r["has_table"] is not None else "—"
        err     = f"`{r['error_type']}`" if r["error_type"] else ""
        lines.append(
            f"| {i} | `{r['strategy']}` | {r['image_name']} | "
            f"{r['image_format']} | {status_icon} | {inf_ms} | {tot_ms} | "
            f"{mem_mb} | {regions} | {table} | {err} |"
        )

    # ── Error analysis ─────────────────────────────────────────────────────
    fail_rows = [r for r in display_rows if r["status"] == "fail"]
    if fail_rows:
        lines += ["", "## Error Analysis", ""]
        err_groups: Dict[str, List[Dict]] = {}
        for r in fail_rows:
            err_groups.setdefault(r["error_type"], []).append(r)

        for etype, errs in sorted(err_groups.items(), key=lambda x: -len(x[1])):
            lines += [
                f"### `{etype}` ({len(errs)} image(s))",
                "",
                f"**Affected images:** {', '.join(r['image_name'] for r in errs[:10])}"
                + (" …" if len(errs) > 10 else ""),
                "",
                f"**Sample message:** `{errs[0]['error_message'][:200]}`",
                "",
            ]

    # ── Fix recommendations ────────────────────────────────────────────────
    all_error_types = {r["error_type"] for r in fail_rows}

    lines += ["", "## Fix Recommendations", ""]

    if "BUG-1:NotCallable" in all_error_types:
        lines += [
            "### BUG-1 – `TypeError: 'PPStructureV3' object is not callable`",
            "",
            "**Cause:** PaddleOCR 3.x removed `__call__`; the API changed from "
            "`engine(img)` to `engine.predict(img)`.",
            "",
            "**Fix:** Replace every `engine(img)` call with `engine.predict(img)` "
            "in your batch script.",
            "",
            "```python",
            "# ❌ Old (PaddleOCR 2.x style – breaks in 3.x)",
            "result = engine(img)",
            "",
            "# ✅ New (PaddleOCR 3.x correct API)",
            "result = engine.predict(img)",
            "```",
            "",
        ]

    if "BUG-2:MKLDNN_PIR" in all_error_types:
        lines += [
            "### BUG-2 – `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`",
            "",
            "**Cause:** oneDNN/MKLDNN on CPU does not yet support "
            "`pir::ArrayAttribute<pir::DoubleAttribute>` introduced in PPStructureV3 3.4. "
            "Triggered especially on PNG images (likely float attribute paths in ops).",
            "",
            "**Fix A (recommended) – disable oneDNN before importing paddle:**",
            "",
            "```python",
            "import os",
            "os.environ['FLAGS_use_mkldnn']      = '0'  # ← must be BEFORE import paddle",
            "os.environ['PADDLE_DISABLE_MKLDNN'] = '1'",
            "os.environ['FLAGS_enable_pir_api']  = '1'  # keep PIR on",
            "",
            "import paddle",
            "from paddleocr import PPStructureV3",
            "engine = PPStructureV3(device='cpu', ...)",
            "result = engine.predict(img)",
            "```",
            "",
            "**Fix B – set env vars on the command line (shell):**",
            "",
            "```bash",
            "FLAGS_use_mkldnn=0 python paddle_ocr_batch.py --input_dir ./images ...",
            "```",
            "",
            "**Fix C – use GPU instead of CPU (avoids MKLDNN entirely):**",
            "",
            "```python",
            "engine = PPStructureV3(device='gpu', ...)",
            "```",
            "",
            "**Fix D – downgrade PaddlePaddle to 2.6.x (last stable MKLDNN+PIR):**",
            "",
            "```bash",
            "pip install paddlepaddle==2.6.2",
            "```",
            "",
        ]

    lines += [
        "## Corrected `paddle_ocr_batch.py` Snippets",
        "",
        "The two critical fixes for the production script:",
        "",
        "```python",
        "# ── Fix BUG-2: place at TOP of file, before any other import ──────",
        "import os",
        "os.environ['FLAGS_use_mkldnn']      = '0'",
        "os.environ['PADDLE_DISABLE_MKLDNN'] = '1'",
        "os.environ['FLAGS_enable_pir_api']  = '1'",
        "",
        "# ── Fix BUG-1: in init_engine(), return the engine object ──────────",
        "def init_engine(args, device):",
        "    from paddleocr import PPStructureV3",
        "    engine = PPStructureV3(",
        "        device=device,",
        "        use_doc_orientation_classify=args.use_doc_orientation_classify,",
        "        use_doc_unwarping=args.use_doc_unwarping,",
        "    )",
        "    return engine   # store and pass to workers",
        "",
        "# ── Fix BUG-1: in process_image(), call .predict() not engine() ────",
        "result = engine.predict(img)   # ✅ correct for PaddleOCR 3.x",
        "# result = engine(img)         # ❌ raises TypeError in 3.x",
        "```",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Environment probe
# ─────────────────────────────────────────────────────────────────────────────

def probe_environment() -> Dict[str, str]:
    info: Dict[str, str] = {
        "Python":   sys.version.split()[0],
        "Platform": platform.platform(),
        "CPU":      platform.processor() or "unknown",
    }

    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        info["RAM"] = f"{vm.total / 1024**3:.1f} GiB total, {vm.available / 1024**3:.1f} GiB free"

    try:
        import paddle
        info["PaddlePaddle"] = paddle.__version__
        has_cuda = (paddle.device.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0)
        info["CUDA available"] = str(has_cuda)
        if has_cuda:
            info["GPU count"] = str(paddle.device.cuda.device_count())
    except Exception as e:
        info["PaddlePaddle"] = f"import error: {e}"

    try:
        import paddleocr
        info["PaddleOCR"] = getattr(paddleocr, "__version__", "unknown")
    except Exception as e:
        info["PaddleOCR"] = f"import error: {e}"

    info["FLAGS_use_mkldnn"] = os.environ.get("FLAGS_use_mkldnn", "not set")
    info["FLAGS_enable_pir_api"] = os.environ.get("FLAGS_enable_pir_api", "not set")

    return info


def _detect_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    try:
        import paddle
        has_gpu = (paddle.device.is_compiled_with_cuda()
                   and paddle.device.cuda.device_count() > 0)
    except Exception:
        has_gpu = False
    if pref == "gpu" and not has_gpu:
        print("[WARN] GPU requested but not available; falling back to CPU.")
        return "cpu"
    return "gpu" if has_gpu else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark_ppstructure",
        description=(
            "Benchmark PaddleOCR PPStructureV3 across multiple inference strategies.\n"
            "Diagnoses BUG-1 (NotCallable) and BUG-2 (MKLDNN/PIR) automatically.\n"
            "Outputs benchmark_results.csv and benchmark_report.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies
----------
  safe       Disable mkldnn, use .predict() API. Recommended fix.
  standard   safe + orientation classify.
  full       safe + orientation + unwarping.
  mkldnn_on  Re-enable mkldnn to reproduce BUG-2.
  callable   Use engine(img) to reproduce BUG-1.
  all        Run all of the above.

Examples
--------
  # Quick diagnostic on 10 images with safe strategy only
  python benchmark_ppstructure.py \\
      --input_dir ./images --output_dir ./bench --limit 10 --strategy safe

  # Full multi-strategy benchmark
  python benchmark_ppstructure.py \\
      --input_dir ./images --output_dir ./bench --strategy all

  # Reproduce and document both bugs
  python benchmark_ppstructure.py \\
      --input_dir ./images --output_dir ./bench --strategy mkldnn_on,callable --limit 20
""",
    )
    p.add_argument("--input_dir",  required=True, help="Folder with input images.")
    p.add_argument("--output_dir", required=True, help="Folder for CSV/Markdown output.")
    p.add_argument(
        "--strategy", default="safe",
        help=(
            "Comma-separated list of strategies to run, or 'all'.  "
            "Choices: safe, standard, full, mkldnn_on, callable, all.  "
            "Default: safe."
        ),
    )
    p.add_argument("--limit",    type=int, default=None, metavar="N",
                   help="Process only first N images (testing).  Default: all.")
    p.add_argument("--min_size", type=int, default=800,
                   help="Upscale images shorter than this (px).  Default: 800.")
    p.add_argument("--device",   choices=["auto", "cpu", "gpu"], default="auto",
                   help="Device: auto|cpu|gpu.  Default: auto.")
    p.add_argument("--warmup",   action="store_true",
                   help="Run one warmup inference per strategy (discarded from stats).")
    p.add_argument("--no_bug_repro", action="store_true",
                   help="Exclude mkldnn_on and callable from 'all' (skip bug reproduction).")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # If --no_bug_repro, remove reproduction strategies from 'all'
    if args.no_bug_repro and args.strategy == "all":
        args.strategy = "safe,standard,full"

    print("=" * 60)
    print("  PPStructureV3 Benchmark")
    print("=" * 60)

    env_info = probe_environment()
    print("\nEnvironment:")
    for k, v in env_info.items():
        print(f"  {k:<22}: {v}")
    print()

    # ── Run benchmarks ────────────────────────────────────────────────────
    all_rows = run_benchmark(args)

    if not all_rows:
        print("[ERROR] No results collected.")
        sys.exit(1)

    # ── Write outputs ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "benchmark_results.csv"
    md_path  = output_dir / "benchmark_report.md"

    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path, env_info)

    # ── Final console summary ─────────────────────────────────────────────
    non_warmup = [r for r in all_rows if not r["warmup"]]
    total   = len(non_warmup)
    success = sum(1 for r in non_warmup if r["status"] == "success")
    fail    = total - success

    print()
    print("=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Total runs  : {total}")
    print(f"  Successful  : {success}")
    print(f"  Failed      : {fail}")

    ok_times = [r["inference_time_ms"] for r in non_warmup
                if r["status"] == "success" and r["inference_time_ms"]]
    if ok_times:
        throughput = 1000 / mean(ok_times)  # images/second
        print(f"  Mean inf.   : {mean(ok_times):.1f} ms/img")
        print(f"  Throughput  : {throughput:.2f} img/s  (single-threaded inference)")

    print(f"  CSV         : {csv_path.resolve()}")
    print(f"  Report      : {md_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
