# Vietnamese License Plate Recognizer

Repository này chứa mã nguồn cho đồ án môn học Project II (IT3930), tập trung vào việc nhận diện biển số xe Việt Nam.

## Tổng quan

Dự án này phát triển một hệ thống nhận diện biển số xe ứng dụng công nghệ Deep Learning và Computer Vision. Hệ thống sử dụng:
- Mô hình YOLOv11 để phát hiện xe và biển số trong ảnh
- Thư viện EasyOCR để  ký tự từ vùng biển số đã được phát hiện.
- Dataset [Vietnamese License Plate](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0) từ Roboflow để huấn luyện mô hình.
- Framework Streamlit để xây dựng giao diện web cho người dùng, cho phép dễ dàng tương tác và thử nghiệm hệ thống.
## Kết quả mô hình

| Model   | Precision | Recall    | mAP@0.5   | mAP@0.5:0.95 | Fitness   | Inference time (ms) |
|---------|-----------|-----------|-----------|---------------|-----------|---------------------|
| YOLO11n | 0.9952    | 0.9854    | 0.9947    | 0.7399        | 0.7652    | **0.855**           |
| YOLO11s | 0.9953    | 0.9878    | 0.9948    | 0.7484        | **0.7730**| 1.207               |
| YOLO11m | **0.9964**| 0.9845    | 0.9948    | 0.7446        | 0.7696    | 3.016               |
| YOLO11l | **0.9964**| 0.9845    | 0.9948    | 0.7446        | 0.7696    | 2.893               |
| YOLO11x | 0.9916    | **0.9901**| **0.9948**| **0.7454**     | 0.7704    | 6.908               |

Với quy mô dự án nhỏ và mục tiêu chạy trên máy cá nhân (local), phiên bản YOLOv11n đã được lựa chọn để cân bằng giữa hiệu năng và tốc độ xử lý.


## Dataset

Dữ liệu được sử dụng để huấn luyện mô hình phát hiện biển số là dataset **Vietnamese License Plate** trên Roboflow. Dataset này bao gồm các hình ảnh biển số xe máy và ô tô tại Việt Nam, được chụp từ nhiều góc độ và trong các điều kiện ánh sáng khác nhau, giúp mô hình học được tính đa dạng và tăng độ chính xác khi nhận diện trong thực tế.

Link dataset: https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0

## Cài đặt

Để triển khai và chạy dự án này trên local, thực hiện theo các bước sau:

1. Clone repository

```bash
git clone https://github.com/NguyenHoangXuanSon/VN-license-plate-recognizer.git
cd VN-license-plate-recognizer
```
2. Cài đặt các thư viện
   
```bash
pip install -r requirement.txt
```

3.Tải mô hình YOLOv11 (có thể tự huấn luyện hoặc dùng có sẵn) và đặt vào thư mục models/

## Cách chạy

Sau khi hoàn tất cài đặt, có thể chạy chương trình sử dụng Streamlit

Trong thư mục của dự án, mở Terminal và chạy lệnh sau

```bash
streamlit run app.py
```
## Kết quả nhận diện biển số 


![](demo.png)


## Cấu trúc dự án

- `app.py`: File chính chạy chương trình, xử lý giao diện và nhận diện biển số xe.
- 
- `main.py`: Điều phối luồng xử lý và gọi các chức năng chính.
- 
- `util.py`: Chứa các hàm hỗ trợ xử lý ảnh và văn bản.
- 
- `visualize.py`: Module để vẽ biển số được nhận diện lên ảnh.
- 
- `models/`: Thư mục chứa các file mô hình YOLOv11 đã được huấn luyện.
- 
- `input/`: Chứa ảnh hoặc video đầu vào để nhận diện.
- 
- `output/`: Lưu kết quả sau khi xử lý.
- 
- `requirement.txt`: Danh sách thư viện và phiên bản cần thiết để chạy dự án.


## Ghi chú
Dự án là một phần đồ án trong học phần IT3930,
Đảm bảo các thư viện như ultralytics, easyocr, opencv-python-headless, streamlit được cài đúng phiên bản trong requirement.txt.
Dataset và mô hình có thể được tùy chỉnh theo nhu cầu.
