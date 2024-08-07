FROM ubuntu:20.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Thiết lập lệnh chạy ứng dụng
CMD ["python3", "predict.py"]