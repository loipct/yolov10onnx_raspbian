FROM ubuntu:20.04

# Đặt thư mục làm việc
WORKDIR /app

# Đặt người dùng là root (mặc định)
USER root

# Cập nhật danh sách gói và cài đặt các gói cần thiết
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv 
    #     libgl1-mesa-glx \
    #     libglib2.0-0 \
    #     libgthread-2.0-0 && \
    # rm -rf /var/lib/apt/lists/*

# Cập nhật pip và cài đặt các phụ thuộc từ requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Thiết lập lệnh chạy ứng dụng
CMD ["python3", "detect.py"]
