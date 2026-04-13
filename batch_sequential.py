import os
import json
import time
import cv2
from pathlib import Path
from paddleocr import PPStructureV3
from tqdm import tqdm

# Cấu hình
INPUT_DIR = "./images"
OUTPUT_DIR = "./results_seq"
MIN_SIZE = 800
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Khởi tạo engine (một lần)
engine = PPStructureV3(device="cpu", use_doc_orientation_classify=False, use_doc_unwarping=False)

# Lấy danh sách ảnh
image_paths = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in SUPPORTED_EXT]
print(f"Found {len(image_paths)} images")

# Tạo thư mục output
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for img_path in tqdm(image_paths, desc="Processing"):
    try:
        # Đọc và resize ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIZE:
            scale = MIN_SIZE / min(h, w)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Dự đoán
        result = engine.predict(img)
        
        # Lưu kết quả
        stem = img_path.stem
        out_dir = Path(OUTPUT_DIR) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Trích xuất HTML (nếu có bảng)
        html = None
        for region in result:
            if region.get("type") == "table":
                html = region.get("res", {}).get("html")
                if html:
                    with open(out_dir / "table.html", "w", encoding="utf-8") as f:
                        f.write(html)
                    break
        
        # Lưu metadata
        metadata = {
            "image": str(img_path),
            "has_table": html is not None,
            "processing_time": time.time(),
            "original_size": [w, h],
            "resized_size": [img.shape[1], img.shape[0]]
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        print(f"Error on {img_path.name}: {e}")
        continue