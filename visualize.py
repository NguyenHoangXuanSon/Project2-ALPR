import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=15, line_length_x=100, line_length_y=100):
    x1, y1 = top_left
    x2, y2 = bottom_right

    y1 = max(0, y1)
    y2 = min(img.shape[0], y2)
    x1 = max(0, x1)
    x2 = min(img.shape[1], x2)

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

results = pd.read_csv('./test.csv')

# load video
video_path = './input_video/test2.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./output_video/output_video6.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_number_score'] == max_)]['license_number'].iloc[0]
    }

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10,
                        line_length_x=100, line_length_y=100)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

            try:
                # Tính kích thước văn bản
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)

                # Vị trí văn bản
                text_x = int((x2 + x1 - text_width) / 2)  # Căn giữa dựa trên x1, x2 của biển số
                text_y = int(y1) - 15
                if text_y - text_height < 0:
                    text_y = text_height + 10  

                # Vẽ nền trắng cho văn bản
                y_text_start = text_y - text_height - 10
                y_text_end = text_y + 10
                x_text_start = text_x - 10
                x_text_end = text_x + text_width + 10
                if y_text_start < 0:
                    y_text_start = 0
                if x_text_start < 0:
                    x_text_start = 0
                frame[y_text_start:y_text_end, x_text_start:x_text_end, :] = (255, 255, 255)

                # Vẽ văn bản biển số
                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
            except Exception as e:
                print(f"Lỗi khi vẽ văn bản biển số: {e}")

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()