import string
import easyocr

# Khởi tạo EasyOCR 
reader = easyocr.Reader(['en'], gpu=False)
allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Chỉ cho phép chữ cái in hoa và số

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score']))
        f.close()

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

    # SỬA: Ưu tiên paragraph=False để lấy score
    detections = reader.readtext(license_plate_crop, allowlist=allowlist, paragraph=False, text_threshold=0.5)
    print(f"Kết quả từ EasyOCR (paragraph=False): {detections}")

    if not detections:
        # Thử lại với paragraph=True
        detections = reader.readtext(license_plate_crop, allowlist=allowlist, paragraph=True, text_threshold=0.5)
        print(f"Kết quả từ EasyOCR (paragraph=True): {detections}")

    if not detections:
        print("EasyOCR không đọc được văn bản từ biển số.")
        return None, None

    # Xử lý kết quả
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

    if license_complies_format(combined_text):
        formatted_text = format_license(combined_text)
        print(f"Biển số sau khi sửa: {formatted_text}")
        return formatted_text, combined_score
    else:
        print(f"Văn bản '{combined_text}' không thỏa mãn định dạng 8 hoặc 9 ký tự.")
        return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        expansion_factor = 0.2
        width = xcar2 - xcar1
        height = ycar2 - ycar1
        xcar1_exp = xcar1 - width * expansion_factor
        ycar1_exp = ycar1 - height * expansion_factor
        xcar2_exp = xcar2 + width * expansion_factor
        ycar2_exp = ycar2 + height * expansion_factor

        if (x1 < xcar2_exp and x2 > xcar1_exp and y1 < ycar2_exp and y2 > ycar1_exp):
            car_indx = j
            foundIt = True
            break

    if foundIt:
        print(f"Gán thành công: Car ID {vehicle_track_ids[car_indx][-1]}, Vùng xe mở rộng: [x1={xcar1_exp}, y1={ycar1_exp}, x2={xcar2_exp}, y2={ycar2_exp}]")
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1