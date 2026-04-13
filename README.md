# PaddleOCR Batch Processing

## Yêu cầu
- Cài Docker Desktop: https://docker.com

## Cách chạy
1. Giải nén file zip
2. Mở terminal tại thư mục này
3. Build image:
   docker build -t paddleocr-batch .
4. Chạy xử lý (thay đổi tham số nếu cần):
   docker run --rm -v "%cd%/images:/app/images" -v "%cd%/results:/app/results" paddleocr-batch --input_dir /app/images --output_dir /app/results --limit 10 --workers 4

Kết quả xuất ra thư mục `results`.