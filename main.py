from ultralytics import YOLO
import cv2
import numpy as np
import util
from util import get_car, read_license_plate, write_csv
import os

coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('./models/best_yolo11n.pt')

image_path = './input/img2.png'
frame_id = os.path.splitext(os.path.basename(image_path))[0]
image = cv2.imread(image_path)

vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck
results = {frame_id: {}}
processed_plates = set()

vehicle_detections = []
for detection in coco_model(image)[0].boxes.data.tolist():
    x1, y1, x2, y2, _, class_id = detection
    if int(class_id) in vehicles:
        vehicle_detections.append([x1, y1, x2, y2, len(vehicle_detections)]) 

license_plates = license_plate_detector(image)[0].boxes.data.tolist()

for license_plate in license_plates:
    x1, y1, x2, y2, score, class_id = license_plate
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_detections)

    license_plate_crop = image[int(y1):int(y2), int(x1):int(x2)]

    # Phóng to ảnh biển số (upscale)
    license_plate_upscaled = cv2.resize(license_plate_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Chuyển sang grayscale
    gray = cv2.cvtColor(license_plate_upscaled, cv2.COLOR_BGR2GRAY)

    # tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # Làm mờ giảm noise
    blur = cv2.medianBlur(gray_clahe, 3)

    # Làm nét ảnh
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(blur, -1, kernel)

    # Áp dụng ngưỡng adaptive
    thresh_adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 5)

    # Áp dụng ngưỡng Otsu
    _, thresh_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Lưu ảnh 
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_upscaled.png', license_plate_upscaled)
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_gray_clahe.png', gray_clahe)
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_blur.png', blur)
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_sharp.png', sharp)
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_thresh_adaptive.png', thresh_adaptive)
    cv2.imwrite(f'./debug_processed_plates/plate_{car_id}_thresh_otsu.png', thresh_otsu)

    print(f"-> Thử OCR trên ảnh gốc (car_id={car_id})")
    license_plate_text, license_plate_score = read_license_plate(license_plate_crop)

    if license_plate_text is None:
        print(f"-> Thử OCR trên ảnh upscaled (car_id={car_id})")
        license_plate_text, license_plate_score = read_license_plate(license_plate_upscaled)

    if license_plate_text is None:
        print(f"-> Thử OCR trên ảnh thresh_adaptive (car_id={car_id})")
        license_plate_text, license_plate_score = read_license_plate(thresh_adaptive)

    if license_plate_text is None:
        print(f"-> Thử OCR trên ảnh gray_clahe (car_id={car_id})")
        license_plate_text, license_plate_score = read_license_plate(gray_clahe)

    if license_plate_text is None:
        print(f"-> Thử OCR trên ảnh sharp (car_id={car_id})")
        license_plate_text, license_plate_score = read_license_plate(sharp)

    if license_plate_text:
        key_id = car_id if car_id != -1 else "unknown"

        print(f" Xe {key_id}: Biển số đọc được: {license_plate_text}, score={license_plate_score}")
        results[frame_id][key_id] = {
            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]} if car_id != -1 else None,
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text': license_plate_text,
                'bbox_score': score,
                'text_score': license_plate_score
            }
        }

    # Lưu ảnh nếu chưa xử lý biển số này
    if license_plate_text not in processed_plates:
        processed_plates.add(license_plate_text)
        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_orig.png', license_plate_crop)
        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_thresh_adaptive.png', thresh_adaptive)
        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_hist_clahe.png', gray_clahe)
        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_sharp.png', sharp)
        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_thresh_otsu.png', thresh_otsu)
    else:
        print(f"Không gán được biển số cho xe nào")

write_csv(results, './result_file.csv')