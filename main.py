from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

mot_tracker = Sort()
results = {}

# load models
coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('./models/best.pt')

# load video
cap = cv2.VideoCapture('./input_video/test2.mp4')

vehicles = [2, 3, 5, 7]  # 2-car, 3-motorbike, 5-bus, 7-truck

processed_plates = set()

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret: 
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_)) 

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                print(f"Phát hiện biển số hợp lệ, Car ID: {car_id}, Vùng biển số: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                print(f"Kích thước ảnh cắt: {license_plate_crop.shape}")

                # Tiền xử lý ảnh
                # Chuyển ảnh màu sang xám 
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # Làm mờ ảnh xám, giảm nhiễu
                license_plate_crop_gray = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
                # Chuyển văn bản thành nền đen, chữ trắng
                license_plate_crop_thresh = cv2.adaptiveThreshold(
                    license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 5
                )
                # Cân bằng histogram
                license_plate_crop_hist = cv2.equalizeHist(license_plate_crop_gray)
                license_plate_crop_hist_thresh = cv2.adaptiveThreshold(
                    license_plate_crop_hist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 5
                )

                # Làm sắc nét ảnh
                # Tăng độ tương phản
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                license_plate_crop_sharp = cv2.filter2D(license_plate_crop_gray, -1, kernel)
                license_plate_crop_sharp_thresh = cv2.adaptiveThreshold(
                    license_plate_crop_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 5
                )

                # Thử lần lượt từng phương pháp tiền xử lý ảnh
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                if license_plate_text is None:
                    print("Thử lại với ảnh đã tiền xử lý (ngưỡng thích ứng)...")
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is None:
                    print("Thử lại với ảnh cân bằng histogram...")
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_hist_thresh)
                if license_plate_text is None:
                    print("Thử lại với ảnh làm sắc nét...")
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_sharp_thresh)

                # Thử phóng to ảnh 
                if license_plate_text is None:
                    print("Thử phóng to ảnh gốc ")
                    license_plate_crop_resized = cv2.resize(license_plate_crop, None, fx=3, fy=3)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_resized)

                if license_plate_text is not None:
                    print(f"Khung hình {frame_nmr}, Car ID {car_id}: Biển số đã xử lý: {license_plate_text}, Độ tin cậy: {license_plate_text_score}")
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

                    # Lưu các ảnh tiền xử lý
                    if license_plate_text not in processed_plates:
                        processed_plates.add(license_plate_text)
                        # Lưu ảnh gốc
                        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_orig.png', license_plate_crop)
                        # Lưu ảnh ngưỡng hóa
                        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_thresh.png', license_plate_crop_thresh)
                        # Lưu ảnh cân bằng histogram
                        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_hist.png', license_plate_crop_hist_thresh)
                        # Lưu ảnh làm sắc nét
                        cv2.imwrite(f'./processed_plates/plate_{license_plate_text}_sharp.png', license_plate_crop_sharp_thresh)
                        print(f"Đã lưu các ảnh tiền xử lý cho biển số: {license_plate_text}")
                else:
                    print(f"Không đọc được văn bản biển số tại khung hình {frame_nmr}, Car ID: {car_id}")
            else:
                print(f"Biển số không được gán cho xe với Car ID: {car_id}")

# write results
write_csv(results, './result_file.csv')