# visualize.py
import cv2
import numpy as np
import pandas as pd

def parse_bbox(bbox_str):
    return list(map(int, bbox_str.strip('[]').replace(',', '').split()))

def visualize_on_image(image, results_df):
    image_copy = image.copy()

    # Group biển số tốt nhất theo car_id
    license_plate = {}
    for car_id in np.unique(results_df['car_id']):
        max_score = np.amax(results_df[results_df['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = results_df[
            (results_df['car_id'] == car_id) &
            (results_df['license_number_score'] == max_score)
        ]['license_number'].iloc[0]

    for _, row in results_df.iterrows():
        x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 5)

        try:
            lp_number = license_plate[row['car_id']]
            (tw, th), _ = cv2.getTextSize(lp_number, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
            tx = int((x1 + x2 - tw) / 2)
            ty = y1 - 15
            if ty - th < 0:
                ty = th + 10

            # Background trắng
            y_start = max(0, ty - th - 10)
            y_end = min(image.shape[0], ty + 10)
            x_start = max(0, tx - 10)
            x_end = min(image.shape[1], tx + tw + 10)
            image_copy[y_start:y_end, x_start:x_end, :] = (255, 255, 255)

            # Text đen
            cv2.putText(image_copy, lp_number, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
        except Exception as e:
            print(f"Lỗi khi vẽ text: {e}")

    return image_copy
