# Vietnamese License Plate Recognizer - IT3930

Đây là repository dành cho đồ án của học phần Project II - IT3930 với đề tài nhận diện biển số xe Việt Nam

## Mô tả

Dự án sử dụng:
- Mô hình **YOLOv11** để phát hiện xe và biển số.
- Dataset [Vietnamese License Plate](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0) từ Roboflow.
- EasyOCR để nhận diện ký tự biển số.
- Streamlit để xây dựng giao diện web.

## Dataset

Sử dụng dataset `Vietnamese License Plate` từ Roboflow, chứa ảnh và nhãn biển số xe Việt Nam, được chuẩn bị kỹ càng phục vụ cho bài toán detection và OCR.

Link dataset:  
https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0

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
app.py: file chính, giao diện và xử lý chính.

visualize.py: module vẽ khung và biển số lên ảnh/video.

models/: chứa các file mô hình YOLOv11 đã huấn luyện.

requirement.txt: danh sách thư viện cần thiết.

## Ghi chú
Dự án là một bài tập thực hành trong học phần IT3930, có thể mở rộng và cải tiến thêm.
Vui lòng đảm bảo các thư viện như ultralytics, easyocr, opencv-python-headless, streamlit được cài đúng phiên bản trong requirement.txt.
Dataset và mô hình có thể được tùy chỉnh theo nhu cầu.
