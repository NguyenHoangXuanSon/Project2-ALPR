

import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import easyocr
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from visualize import visualize_on_image  # import hàm visualize đã sửa

# ========== MODEL KHỞI TẠO ==========
coco_model = YOLO('yolo11n.pt')  # Model detect xe
license_plate_model = YOLO('./models/best_yolo11n.pt')  # Model detect biển số
reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2, 3, 5, 7]  # Class IDs cho các loại xe

def get_car(license_plate, vehicle_boxes):
    x1, y1, x2, y2, score, class_id = license_plate
    for i, (vx1, vy1, vx2, vy2, car_id) in enumerate(vehicle_boxes):
        expand = 0.2
        w = vx2 - vx1
        h = vy2 - vy1
        if (x1 < vx2 + w * expand and x2 > vx1 - w * expand and
            y1 < vy2 + h * expand and y2 > vy1 - h * expand):
            return vx1, vy1, vx2, vy2, car_id
    return -1, -1, -1, -1, -1

def read_license_plate(image):
    detections = reader.readtext(image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=False)
    if not detections:
        return None, None
    text = ''.join([d[1].upper() for d in detections])
    score = max([d[2] for d in detections])
    return text, score

def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

# ========== GIAO DIỆN ==========
st.set_page_config(page_title="Vietnamese Licence Plate Recognizer", layout="wide")
st.markdown("<h1 style='color:#3d7bb6 ; font-size:60px;'>Vietnamese Licence Plate Recognizer</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image_path = tfile.name
    image = cv2.imread(image_path)

    # === Detect vehicles ===
    results = coco_model(image)[0]
    vehicle_boxes = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            vehicle_boxes.append([x1, y1, x2, y2, len(vehicle_boxes)])

    # === Detect license plates ===
    plates = license_plate_model(image)[0].boxes.data.tolist()

    result_rows = []

    # === Vẽ kết quả lên ảnh gốc
    if plates:
        result_df = pd.DataFrame()
        rows = []
        for lp in plates:
            x1, y1, x2, y2, score, class_id = lp
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, vehicle_boxes)

            plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
            if plate_crop.size == 0:
                continue
            plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            text, conf = read_license_plate(plate_crop_rgb)

            rows.append({
                'car_id': car_id,
                'license_plate_bbox': f"[{int(x1)} {int(y1)} {int(x2)} {int(y2)}]",
                'license_number': text if text else '',
                'license_number_score': conf if conf else 0.0
            })
        result_df = pd.DataFrame(rows)
        vis_img = visualize_on_image(image.copy(), result_df)
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    else:
        vis_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === Giao diện 2 cột ===
    left_col, right_col = st.columns([1,1])

    # ===== LEFT: Ảnh gốc đã vẽ kết quả, có viền =====
    with left_col:
        img_b64 = image_to_base64(vis_img_rgb)
        st.markdown(
            f"""
            <div style="border: 5px solid #b0b7be; padding: 5px; border-radius: 10px; max-width: 100%;">
                <img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; border-radius: 10px;" />
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===== RIGHT: Thông tin chi tiết + OCR =====
    with right_col:
        if not plates:
            st.warning("Không phát hiện biển số.")
        else:
            for lp in plates:
                x1, y1, x2, y2, score, class_id = lp
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, vehicle_boxes)

                plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
                if plate_crop.size == 0:
                    st.error("Không crop được biển số.")
                    continue

                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                text, conf = read_license_plate(plate_crop_rgb)

                st.image(plate_crop_rgb, caption="Biển số đã cắt", width=200)

                st.markdown(f"<h3 style='color:#C70039;'>Car ID - {car_id if car_id != -1 else 'Không xác định'}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Tọa độ biển số</span>: <span style='color:#454748;'>[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]</span></b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Độ chính xác bbox</span>: <span style='color:#454748;'>{score:.2%}</span></b></p>", unsafe_allow_html=True)

                if text:
                    st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Biển số</span>: <span style='color:#454748;'>{text}</span></b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Độ tin cậy OCR</span>: <span style='color:#454748;'>{conf:.2%}</span></b></p>", unsafe_allow_html=True)
                else:
                    st.error("Không đọc được biển số.")

                st.markdown("---")
