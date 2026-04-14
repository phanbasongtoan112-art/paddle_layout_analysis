# PPStructureV3 Benchmark Report
*Generated: 2026-04-14 10:32:17*

## Environment
| Key | Value |
|-----|-------|
| Python | 3.13.3 |
| Platform | Windows-11-10.0.26200-SP0 |
| CPU | Intel64 Family 6 Model 126 Stepping 5, GenuineIntel |
| RAM | 15.8 GiB total, 7.0 GiB free |
| PaddlePaddle | 3.2.0 |
| CUDA available | False |
| PaddleOCR | 3.4.0 |
| FLAGS_use_mkldnn | 0 |
| FLAGS_enable_pir_api | 1 |

## Strategy Summary

| Strategy | Total | Success | Fail | Mean ms | Median ms | Stdev ms | Tables found | Error types |
|---|---|---|---|---|---|---|---|---|
| `safe` | 10 | 10 | 0 | 280633.2 | 203589.7 | 145249.6 | 0 | — |

## Per-Image Results

| # | Strategy | Image | Format | Status | Inf (ms) | Total (ms) | Mem ΔMiB | Regions | Table | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (1).jpg | jpg | ✅ | 432233.26 | 432275.04 | 783.88 | 1 | ✗ |  |
| 2 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (10).jpg | jpg | ✅ | 210953.26 | 210976.85 | 735.5 | 1 | ✗ |  |
| 3 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (100).jpg | jpg | ✅ | 580720.07 | 580805.87 | -764.29 | 1 | ✗ |  |
| 4 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (11).jpg | jpg | ✅ | 342992.93 | 343055.64 | 49.24 | 1 | ✗ |  |
| 5 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (12).jpg | jpg | ✅ | 171528.73 | 171595.37 | 358.55 | 1 | ✗ |  |
| 6 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (13).jpg | jpg | ✅ | 372395.38 | 372447.25 | -384.57 | 1 | ✗ |  |
| 7 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (14).jpg | jpg | ✅ | 173827.54 | 173862.75 | 64.18 | 1 | ✗ |  |
| 8 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (15).jpg | jpg | ✅ | 135905.99 | 135948.93 | 342.66 | 1 | ✗ |  |
| 9 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (16).jpg | jpg | ✅ | 196226.04 | 196314.91 | -861.04 | 1 | ✗ |  |
| 10 | `safe` | 138.tar_1706.07989.gz_CoPS3_arXiv_2017_17_ori (17).jpg | jpg | ✅ | 189548.98 | 189597.45 | -104.79 | 1 | ✗ |  |

## Fix Recommendations

## Corrected `paddle_ocr_batch.py` Snippets

The two critical fixes for the production script:

```python
# ── Fix BUG-2: place at TOP of file, before any other import ──────
import os
os.environ['FLAGS_use_mkldnn']      = '0'
os.environ['PADDLE_DISABLE_MKLDNN'] = '1'
os.environ['FLAGS_enable_pir_api']  = '1'

# ── Fix BUG-1: in init_engine(), return the engine object ──────────
def init_engine(args, device):
    from paddleocr import PPStructureV3
    engine = PPStructureV3(
        device=device,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
    )
    return engine   # store and pass to workers

# ── Fix BUG-1: in process_image(), call .predict() not engine() ────
result = engine.predict(img)   # ✅ correct for PaddleOCR 3.x
# result = engine(img)         # ❌ raises TypeError in 3.x
```
