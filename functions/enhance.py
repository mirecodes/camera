import cv2
import numpy as np
import os

def super_resolution_stacking(video_path, output_path, max_frames=90, apply_sharpening=True, strength="strong"):
    """
    Stacks multiple video frames to eliminate digital noise and average out sensor vibration,
    then applies an edge-enhancing sharpening filter to restore crisp text boundaries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return False

    frames = []
    count = 0

    print(f"Extracting up to {max_frames} frames from video: {video_path}...")
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for cleaner text/ocr processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1

    cap.release()

    if len(frames) == 0:
        print("Error: No frames extracted from video.")
        return False

    print(f"Synthesizing {len(frames)} frames to clear noise (Temporal Averaging)...")
    
    # 1. Multi-frame stacking: convert to 32-bit float for precision, average, and convert back to 8-bit
    all_frames = np.array(frames, dtype=np.float32)
    stacked_image = np.mean(all_frames, axis=0)
    stacked_image = np.clip(stacked_image, 0, 255).astype(np.uint8)

    # 2. Image Sharpening
    if apply_sharpening:
        print(f"Applying {strength} sharpening filter...")
        if strength == "strong":
            sharpening_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
        else:
            # Standard laplacian sharpening kernel
            sharpening_kernel = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0]
            ])
        enhanced_image = cv2.filter2D(stacked_image, -1, sharpening_kernel)
    else:
        enhanced_image = stacked_image

    # 3. Save result
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, enhanced_image)
    print(f"💾 Enhanced high-resolution image saved to: {output_path}")
    return True
