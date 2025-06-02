# Vietnamese License Plate Recognizer

Repository này chứa mã nguồn cho đồ án môn học Project II (IT3930), tập trung vào việc nhận diện biển số xe Việt Nam.

## Tổng 

Dự án này phát triển một hệ thống nhận diện biển số xe ứng dụng công nghệ Deep Learning và Computer Vision. Hệ thống sử dụng:
- Mô hình YOLOv11 để phát hiện xe và biển số trong ảnh
- Thư viện EasyOCR để  ký tự từ vùng biển số đã được phát hiện.
- Dataset [Vietnamese License Plate](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0) từ Roboflow để huấn luyện mô hình.
- Framework Streamlit để xây dựng giao diện web cho người dùng, cho phép dễ dàng tương tác và thử nghiệm hệ thống.
  
## Dataset

Dữ liệu được sử dụng để huấn luyện mô hình phát hiện biển số là dataset **Vietnamese License Plate** trên Roboflow. Dataset này bao gồm các hình ảnh biển số xe máy và ô tô tại Việt Nam, được chụp từ nhiều góc độ và trong các điều kiện ánh sáng khác nhau, giúp mô hình học được tính đa dạng và tăng độ chính xác khi nhận diện trong thực tế.

Link dataset: https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0

## Cài đặt

1. Clone repository:

```bash
git clone https://github.com/ilbdculuv/VNLicensePlateRecognizer.git
cd VNLicensePlateRecognizer
```
2. Cài đặt các thư viện:
```bash
pip install -r requirement.txt
```
3.Tải mô hình YOLOv11 (bạn có thể tự huấn luyện hoặc dùng file có sẵn) và đặt vào thư mục models/

## Cách chạy
Chạy ứng dụng bằng Streamlit
```bash
streamlit run app.py
```
Mở trình duyệt và truy cập địa chỉ được thông báo (mặc định http://localhost:8501) để sử dụng.

## Cấu trúc dự án

- `app.py`: File chính chạy ứng dụng, xử lý giao diện và logic nhận diện.
- 
- `main.py`: Điều phối luồng xử lý và gọi các chức năng chính.
- 
- `util.py`: Chứa các hàm tiện ích hỗ trợ xử lý ảnh và văn bản.
- 
- `visualize.py`: Module vẽ khung phát hiện và biển số lên ảnh hoặc video.
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
Vui lòng đảm bảo các thư viện như ultralytics, easyocr, opencv-python-headless, streamlit được cài đúng phiên bản trong requirement.txt.
Dataset và mô hình có thể được tùy chỉnh theo nhu cầu.
