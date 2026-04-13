# Sử dụng Python 3.9-slim làm base image
FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và PaddlePaddle
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói Python (chú ý thứ tự: paddlex[ocr] trước paddleocr)
RUN pip install --no-cache-dir \
    paddlepaddle==3.2.0 \
    'paddlex[ocr]' \
    paddleocr==3.4.0 \
    opencv-python-headless \
    tqdm

# Copy script chính vào container
COPY paddle_ocr_batch.py /app/paddle_ocr_batch.py

# Đảm bảo script có quyền thực thi
RUN chmod +x /app/paddle_ocr_batch.py

# Entrypoint mặc định
ENTRYPOINT ["python", "/app/paddle_ocr_batch.py"]