import string
import easyocr

reader = easyocr.Reader(['en'], gpu=True)
allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'image_name',
            'car_id',
            'car_bbox',
            'license_plate_bbox',
            'license_plate_bbox_score',
            'license_number',
            'license_number_score',
        ))
        for image_name in results.keys():
            for car_id, plate_result in results[image_name].items():
                if not isinstance(plate_result, dict):
                    continue
                if 'license_plate' in plate_result and 'text' in plate_result['license_plate']:
                    bbox_lp = plate_result['license_plate']['bbox']
                    car_info = plate_result.get('car')
                    if isinstance(car_info, dict):
                        bbox_car = car_info.get('bbox', [])
                        bbox_car_str = '[{} {} {} {}]'.format(*[int(coord) for coord in bbox_car]) if bbox_car else '[]'
                    else:
                        bbox_car_str = '[]'

                    f.write('{},{},{},{},{},{},{}\n'.format(
                        image_name,
                        car_id,
                        bbox_car_str,
                        '[{} {} {} {}]'.format(*[int(coord) for coord in bbox_lp]),
                        plate_result['license_plate']['bbox_score'],
                        plate_result['license_plate']['text'],
                        plate_result['license_plate']['text_score'],
                    ))


def clean_text(text):
    cleaned = text.replace(' ', '').replace('-', '').replace('.', '')
    print(f"Văn bản sau khi làm sạch: {cleaned}")
    return cleaned

def license_complies_format(text):
    if len(text) not in [8, 9]:
        return False

    if len(text) == 8:
        if not (text[0] in '0123456789' and text[1] in '0123456789'):
            return False
        if not (text[2] in string.ascii_uppercase):
            return False
        for i in range(3, 8):
            if not (text[i] in '0123456789'):
                return False
        return True

    if len(text) == 9:
        if not (text[0] in '0123456789' and text[1] in '0123456789'):
            return False
        if not (text[2] in string.ascii_uppercase):
            return False
        if not (text[3] in '0123456789'):
            return False
        for i in range(4, 9):
            if not (text[i] in '0123456789'):
                return False
        return True

    return False

def format_license(text):

    if len(text) > 9:
        text = text[:9]

    elif len(text) > 8 and len(text) < 9:
        text = text[:8]

    if len(text) not in [8, 9]:
        return text

    license_plate_ = ''
    if len(text) == 8:
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int,
                   6: dict_char_to_int, 7: dict_char_to_int}
        for j in range(8):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
    else:
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int,
                   6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
        for j in range(9):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):

    detections = reader.readtext(license_plate_crop, allowlist=allowlist, paragraph=False, text_threshold=0.5)
    print(f"Kết quả từ EasyOCR (paragraph=False): {detections}")

    if not detections:
        detections = reader.readtext(license_plate_crop, allowlist=allowlist, paragraph=True, text_threshold=0.5)
        print(f"Kết quả từ EasyOCR (paragraph=True): {detections}")

    if not detections:
        print("EasyOCR không đọc được văn bản từ biển số.")
        return None, None

    combined_text = ''
    combined_score = 0

    for detection in detections:
        if isinstance(detection, (list, tuple)):
            if len(detection) == 2:  
                bbox, text = detection
                score = 0  
            elif len(detection) == 3: 
                bbox, text, score = detection
            else:
                print(f"Cảnh báo: Định dạng không mong đợi từ EasyOCR: {detection}")
                continue
        else:
            print(f"Cảnh báo: Định dạng không mong đợi từ EasyOCR: {detection}")
            continue

        raw_text = text.upper()
        print(f"Biển số trước khi sửa (đoạn): {raw_text}, Độ tin cậy: {score}")
        combined_text += raw_text
        combined_score = max(combined_score, score)

    # Làm sạch văn bản
    combined_text = clean_text(combined_text)
    print(f"Biển số sau khi ghép và làm sạch: {combined_text}, Độ tin cậy: {combined_score}")

    formatted_text = format_license(combined_text)
    print(f"Biển số sau khi sửa: {formatted_text}")

    if license_complies_format(formatted_text):
        return formatted_text, combined_score
    else:
        print(f"Văn bản '{formatted_text}' không thỏa mãn định dạng 8 hoặc 9 ký tự.")
        return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for j, (vx1, vy1, vx2, vy2, car_id) in enumerate(vehicle_track_ids):
        expansion_factor = 0.2
        width = vx2 - vx1
        height = vy2 - vy1
        vx1_exp = vx1 - width * expansion_factor
        vy1_exp = vy1 - height * expansion_factor
        vx2_exp = vx2 + width * expansion_factor
        vy2_exp = vy2 + height * expansion_factor

        if (x1 < vx2_exp and x2 > vx1_exp and y1 < vy2_exp and y2 > vy1_exp):
            print(f"Gán thành công: Car ID {car_id}, Vùng xe mở rộng: [x1={vx1_exp}, y1={vy1_exp}, x2={vx2_exp}, y2={vy2_exp}]")
            return vx1, vy1, vx2, vy2, car_id

    return -1, -1, -1, -1, -1
