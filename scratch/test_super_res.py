import cv2
import numpy as np
import time

def test_super_res_stacking():
    # Load frames from the last captured test
    # Let's read the video stream to get raw frames
    video_path = "/Users/mireflare/Documents/Codes/mechatronics/stream/temp_output.mp4"
    if not os.path.exists(video_path):
        # Fallback: if video is missing, we can't run the test
        print("Video stream missing. Please run capture first.")
        return

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()

    if len(frames) == 0:
        print("No frames found.")
        return

    ref_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = ref_frame.shape
    
    # 2x Super Resolution setup
    h_2x, w_2x = h * 2, w * 2
    stacked_float_2x = np.zeros((h_2x, w_2x), dtype=np.float32)
    
    # Scale ref frame to 2x using Lanczos interpolation
    ref_frame_2x = cv2.resize(ref_frame, (w_2x, h_2x), interpolation=cv2.INTER_LANCZOS4)
    stacked_float_2x += ref_frame_2x.astype(np.float32)

    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    aligned_count = 1

    print("Aligning at 1x, warping at 2x...")
    for idx in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            # 1. Align at 1x (Fast)
            cc, warp_matrix = cv2.findTransformECC(
                ref_frame, curr_frame, warp_matrix, warp_mode, criteria, None, 5
            )
            
            # 2. Scale warp matrix to 2x (Scale translation terms)
            warp_matrix_2x = warp_matrix.copy()
            warp_matrix_2x[0, 2] *= 2.0  # Scale Tx
            warp_matrix_2x[1, 2] *= 2.0  # Scale Ty
            
            # 3. Scale frame to 2x
            curr_frame_2x = cv2.resize(curr_frame, (w_2x, h_2x), interpolation=cv2.INTER_LANCZOS4)
            
            # 4. Warp at 2x
            aligned_frame_2x = cv2.warpAffine(
                curr_frame_2x,
                warp_matrix_2x,
                (w_2x, h_2x),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            stacked_float_2x += aligned_frame_2x.astype(np.float32)
            aligned_count += 1
        except cv2.error:
            continue

    print(f"Aligned {aligned_count}/{len(frames)} frames.")
    stacked_2x = stacked_float_2x / aligned_count
    stacked_2x = np.clip(stacked_2x, 0, 255).astype(np.uint8)

    # 5. Contrast Enhancement & Unsharp Masking in 2x domain
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_2x = clahe.apply(stacked_2x)
    
    gaussian_blur = cv2.GaussianBlur(clahe_2x, (9, 9), 1.5)
    result_img_2x = cv2.addWeighted(clahe_2x, 2.2, gaussian_blur, -1.2, 0)

    cv2.imwrite("/Users/mireflare/Documents/Codes/mechatronics/data/super_res_test_result.png", result_img_2x)
    print("Super-Resolution test image written to data/super_res_test_result.png")

import os
if __name__ == "__main__":
    test_super_res_stacking()
