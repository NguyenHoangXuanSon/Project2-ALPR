# Vietnamese License Plate Recognizer - IT3930

Đây là repository cho đồ án dành cho học phần Project II - IT3930 với đề tài nhận diện biển số xe Việt Nam.

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
-dot-dot
   git clone https://github.com/ilbdculuv/VNLicensePlateRecognizer.git
   cd VNLicensePlateRecognizer
2. Cài đặt thư viện 
pip install -r requirement.txt
