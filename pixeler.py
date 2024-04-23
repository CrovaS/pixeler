from PIL import Image
import numpy as np
import cv2

def pixelate_image(input_image_path, horizontal_len, vertical_len):
    # 이미지를 로드
    img = Image.open(input_image_path)

    # 이미지 크기와 비율을 계산
    width, height = img.size
    target_aspect = horizontal_len / vertical_len

    # 중앙을 기준으로 이미지를 자르기
    source_aspect = width / height

    if source_aspect > target_aspect:
        # 너비가 너무 길 때
        new_width = int(target_aspect * height)
        offset = (width - new_width) / 2
        img = img.crop((offset, 0, width - offset, height))
    else:
        # 높이가 너무 길 때
        new_height = int(width / target_aspect)
        offset = (height - new_height) / 2
        img = img.crop((0, offset, width, height - offset))

    # 이미지를 픽셀화
    img = img.resize((horizontal_len, vertical_len), Image.NEAREST)

    # 픽셀화된 이미지 저장
    img.save("pixelated_image.png")

    # HSB 변환
    img_cv = np.array(img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

    return hue, sat, val

def scale_hsb(hue_matrix, saturation_matrix, brightness_matrix, max_value=39):
    # Hue의 경우, 최대값이 179이므로 39로 스케일링
    hue_scaled = np.round(hue_matrix * (max_value / 179))
    # Saturation과 Brightness의 경우, 최대값이 255이므로 39로 스케일링
    saturation_scaled = np.round(saturation_matrix * (max_value / 255))
    brightness_scaled = np.round(brightness_matrix * (max_value / 255))
    
    return hue_scaled.astype(int), saturation_scaled.astype(int), brightness_scaled.astype(int)

def save_hsb_matrices(hue_matrix, saturation_matrix, brightness_matrix, prefix="hsb_output"):
    # Hue 저장
    np.savetxt(f"{prefix}_hue.txt", hue_matrix, fmt="%d")
    # Saturation 저장
    np.savetxt(f"{prefix}_saturation.txt", saturation_matrix, fmt="%d")
    # Brightness 저장
    np.savetxt(f"{prefix}_brightness.txt", brightness_matrix, fmt="%d")

def save_combined_hsb(combined_hsb, filename="combined_hsb_output.txt"):
    # 3차원 배열을 2차원 배열로 변환: 각 행이 하나의 픽셀의 [Hue, Saturation, Brightness]를 가지도록 함
    reshaped_array = combined_hsb.reshape(-1, combined_hsb.shape[-1])
    np.savetxt(filename, reshaped_array, fmt="%d")

def print_hsb_block(combined_hsb, start_row, start_col, end_row, end_col):
    end_row = end_row + 1
    end_col = end_col + 1
    hsb_block = combined_hsb[start_row:end_row, start_col:end_col]
    print("HSB block display:")
    for row in hsb_block:
        for values in row:
            print(f"[{values[0]} {values[1]} {values[2]}]", end=' ')
        print()

# 사용 예
hue, sat, brightness = pixelate_image("test1.jpg", 50, 30)
# print("Hue Matrix:")
# print(hue)
# print("Saturation Matrix:")
# print(sat)
# print("Brightness Matrix:")
# print(brightness)
hue_scaled, sat_scaled, brightness_scaled = scale_hsb(hue, sat, brightness)
combined_hsb = np.stack((hue_scaled, sat_scaled, brightness_scaled), axis=-1)
save_hsb_matrices(hue_scaled, sat_scaled, brightness_scaled, prefix="testA")
save_combined_hsb(combined_hsb, "testA_all.txt")
# print(combined_hsb[10,10])
print_hsb_block(combined_hsb, 10, 40, 19, 49)