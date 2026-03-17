import os
import cv2
import numpy as np

def single_scale_retinex(img, sigma):
    blurred_img = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1) - np.log10(blurred_img + 1)
    return retinex, blurred_img

def ssr(img, sigma):
    img = img.astype(np.float32) / 255.0
    retinex, illumination = single_scale_retinex(img, sigma)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    retinex = retinex.astype(np.uint8)

    illumination = (illumination - np.min(illumination)) / (np.max(illumination) - np.min(illumination)) * 255
    illumination = illumination.astype(np.uint8)

    return retinex, illumination

def process_folder(input_folder, output_folder_retinex, output_folder_illumination, sigma):
    if not os.path.exists(output_folder_retinex):
        os.makedirs(output_folder_retinex)
    if not os.path.exists(output_folder_illumination):
        os.makedirs(output_folder_illumination)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            retinex, illumination = ssr(img, sigma)

            retinex_path = os.path.join(output_folder_retinex, filename)
            illumination_path = os.path.join(output_folder_illumination, filename)

            cv2.imwrite(retinex_path, retinex)
            cv2.imwrite(illumination_path, illumination)
            print(f"Processed {filename}")
