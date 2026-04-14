# PaddleOCR Layout Analysis – Phát hiện bố cục & trích xuất bảng từ ảnh

Pipeline xử lý hàng loạt ảnh (batch) sử dụng **PaddleOCR PPStructureV3** để:
- Phát hiện các vùng (region) trên ảnh tài liệu: bảng, văn bản, tiêu đề, hình ảnh, công thức…
- Trích xuất nội dung bảng ra file HTML (nếu có bảng).
- Xuất metadata (tọa độ, độ tin cậy, kích thước…) và ảnh minh họa có khung màu.

Có thể chạy trên hàng nghìn ảnh, hỗ trợ checkpoint (tiếp tục sau khi dừng), đa luồng, và đóng gói Docker để dễ dàng triển khai.

---

## Yêu cầu

### Nếu chạy bằng Python (trực tiếp)
- Python 3.9
- Các thư viện: `paddlepaddle`, `paddleocr`, `paddlex[ocr]`, `opencv-python-headless`, `tqdm`

### Nếu chạy bằng Docker
- Docker Desktop (cài từ [docker.com](https://www.docker.com/products/docker-desktop/))
- Khoảng 5-10 GB dung lượng trống

---

## Cách cài đặt và chạy

### 1. Chạy trực tiếp bằng Python (không Docker)

**Bước 1: Tạo môi trường ảo Python 3.9**
```bash
py -3.9 -m venv paddle_env
.\paddle_env\Scripts\activate   # Windows
# source paddle_env/bin/activate # Linux/Mac
Bước 2: Cài đặt thư viện

bash
pip install paddlepaddle==3.2.0
pip install 'paddlex[ocr]'
pip install paddleocr==3.4.0 opencv-python-headless tqdm
Bước 3: Chuẩn bị ảnh đầu vào

Đặt tất cả ảnh (.jpg, .png, .bmp, …) vào thư mục images.

Bước 4: Chạy script

bash
python paddle_ocr_batch.py --input_dir ./images --output_dir ./results --limit 10 --workers 4
💡 Tham số --limit 10 giới hạn xử lý 10 ảnh đầu tiên (dùng để kiểm tra). Bỏ --limit để xử lý toàn bộ.

2. Chạy bằng Docker (đóng gói sẵn môi trường)
Bước 1: Build Docker image

bash
docker build -t paddleocr-batch .
Bước 2: Chạy container

bash
docker run --rm -v "${PWD}/images:/app/images" -v "${PWD}/results:/app/results" paddleocr-batch --input_dir /app/images --output_dir /app/results --limit 10 --workers 4
📌 Lưu ý: Đường dẫn images và results là thư mục trên máy bạn, sẽ được gắn vào container.

Cấu trúc kết quả đầu ra
Sau khi chạy, thư mục results/ sẽ có cấu trúc:

text
results/
├── checkpoint.json          # danh sách ảnh đã xử lý thành công (dùng để resume)
├── error.log                # ghi lại lỗi nếu có
├── image1/
│   ├── table.html           # bảng dạng HTML (chỉ xuất hiện nếu ảnh có bảng)
│   ├── metadata.json        # thông tin chi tiết: vùng, kích thước, thời gian, độ tin cậy
│   └── layout.jpg           # ảnh gốc có vẽ khung các vùng
├── image2/
│   └── ...
└── ...
Ý nghĩa các file
metadata.json – Chứa danh sách các vùng phát hiện được (regions), mỗi vùng có type (table, text, title…), bbox (tọa độ), score (độ tin cậy).

table.html – Bảng được trích xuất, mở bằng trình duyệt để xem.

layout.jpg – Hình ảnh minh họa giúp kiểm tra trực quan.

Các tham số dòng lệnh quan trọng
Tham số	Ý nghĩa	Mặc định
--input_dir	Thư mục chứa ảnh đầu vào	bắt buộc
--output_dir	Thư mục lưu kết quả	bắt buộc
--workers	Số luồng xử lý song song	cpu_count()
--min_size	Kích thước tối thiểu cạnh ngắn (px)	800
--enhance_contrast	Tăng độ tương phản (CLAHE)	False
--denoise	Khử nhiễu ảnh	False
--device	auto, cpu hoặc gpu	auto
--limit	Chỉ xử lý N ảnh đầu tiên	None
--resume	Bỏ qua ảnh đã có trong checkpoint	False
--no_layout	Không lưu ảnh layout.jpg	False
Nguyên lý hoạt động
Tiền xử lý – Phóng to ảnh nếu quá nhỏ, tăng cường độ tương phản (nếu bật), khử nhiễu (nếu bật).

Phát hiện bố cục – Mô hình PPStructureV3 chia ảnh thành các vùng và gán nhãn (table, text, title, figure, formula…).

Trích xuất bảng – Nếu vùng table được phát hiện, pipeline nhận dạng cấu trúc bảng, OCR từng ô và xuất thành HTML.

Xuất kết quả – Lưu metadata.json, table.html (nếu có), và layout.jpg.

Xử lý batch – Dùng đa luồng, checkpoint, log lỗi để xử lý hàng nghìn ảnh ổn định.

## Xử lý lỗi MKLDNN trên Windows

Nếu gặp lỗi `NotImplementedError: ConvertPirAttribute2RuntimeAttribute...`, hãy đặt các biến môi trường sau ở đầu script:
```python
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

## Kết quả mẫu (smoke test)
Với 10 ảnh JPG, chiến lược `safe` đạt 100% thành công, thời gian trung bình 281s/ảnh. Xem báo cáo chi tiết tại `bench/benchmark_report.md`.