import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from paddleocr import PaddleOCR
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from visualize import visualize_on_image 
from util import read_license_plate, get_car

coco_model = YOLO('yolo11n.pt')  
license_plate_model = YOLO('./models/best_yolo11n.pt')  

vehicles = [2, 3, 5, 7]  

def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

st.set_page_config(page_title="Vietnamese Licence Plate Recognizer", layout="wide")
st.markdown("<h1 style='color:#3d7bb6 ; font-size:60px;'>Vietnamese Licence Plate Recognizer</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image_path = tfile.name
    image = cv2.imread(image_path)

    results = coco_model(image)[0]
    vehicle_boxes = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            vehicle_boxes.append([x1, y1, x2, y2, len(vehicle_boxes)])

    plates = license_plate_model(image)[0].boxes.data.tolist()

    # Lưu kết quả OCR từng biển
    result_df = pd.DataFrame()

    if plates:
        rows = []
        for lp in plates:
            x1, y1, x2, y2, score, class_id = lp
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, vehicle_boxes)

            plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
            if plate_crop.size == 0:
                continue
            plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

            # Gọi OCR 1 lần, lưu kết quả
            text, conf = read_license_plate(plate_crop_rgb)

            rows.append({
                'car_id': car_id,
                'license_plate_bbox': f"[{int(x1)} {int(y1)} {int(x2)} {int(y2)}]",
                'license_number': text if text else '',
                'license_number_score': conf if conf else 0.0,
                'plate_crop_rgb': plate_crop_rgb  # lưu ảnh để hiển thị
            })

        result_df = pd.DataFrame(rows)
        vis_img = visualize_on_image(image.copy(), result_df)
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    else:
        vis_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    left_col, right_col = st.columns([1,1])

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

    with right_col:
        if result_df.empty:
            st.warning("Không phát hiện biển số.")
        else:
            for idx, row in result_df.iterrows():
                st.image(row['plate_crop_rgb'], caption="Biển số đã cắt", width=200)

                st.markdown(f"<h3 style='color:#C70039;'>Car ID - {row['car_id'] if row['car_id'] != -1 else 'Không xác định'}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Tọa độ biển số</span>: <span style='color:#454748;'>{row['license_plate_bbox']}</span></b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Độ chính xác bbox</span>: <span style='color:#454748;'>{float(plates[idx][4]):.2%}</span></b></p>", unsafe_allow_html=True)

                if row['license_number']:
                    st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Biển số</span>: <span style='color:#454748;'>{row['license_number']}</span></b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:18px;'><b><span style='color:#C70039;'>Độ tin cậy OCR</span>: <span style='color:#454748;'>{row['license_number_score']:.2%}</span></b></p>", unsafe_allow_html=True)
                else:
                    st.error("Không đọc được biển số.")

                st.markdown("---")
