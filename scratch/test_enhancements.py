import cv2
import numpy as np

# Load the current stacked result
img = cv2.imread("/Users/mireflare/Documents/Codes/mechatronics/data/stacked_test_result.png", cv2.IMREAD_GRAYSCALE)

if img is not None:
    # 1. Apply CLAHE to boost text contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    
    # 2. Strong Unsharp Masking
    gaussian = cv2.GaussianBlur(clahe_img, (5, 5), 1.5)
    enhanced1 = cv2.addWeighted(clahe_img, 2.2, gaussian, -1.2, 0)
    
    # 3. Adaptive Thresholding (Alternative for maximum binary readability)
    thresh = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    
    cv2.imwrite("/Users/mireflare/Documents/Codes/mechatronics/data/test_enhanced_unsharp.png", enhanced1)
    cv2.imwrite("/Users/mireflare/Documents/Codes/mechatronics/data/test_enhanced_thresh.png", thresh)
    print("Test images written successfully.")
else:
    print("Failed to read image.")
