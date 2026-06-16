#!/usr/bin/env python3
import os
import sys
import time
import cv2
import yaml
import numpy as np
import ssl
import layoutparser as lp

# Bypass SSL certificates verification for model downloads (crucial for macOS environments)
ssl._create_default_https_context = ssl._create_unverified_context

class WebcamCapturer:
    def __init__(self, config_path="cfg/capture_cfg.yaml"):
        # Resolve paths relative to project root (two levels up from functions/capture.py)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(file_dir)
        
        if not os.path.isabs(config_path):
            self.config_path = os.path.join(project_root, config_path)
        else:
            self.config_path = config_path

        print(f"Loading configuration from: {self.config_path}")
        self.config = self.load_config()

        # Parse config parameters
        self.camera_index = self.config['camera']['index']
        self.width = self.config['camera']['resolution']['width']
        self.height = self.config['camera']['resolution']['height']
        self.fps = self.config['camera'].get('fps', 30)
        self.record_stream = self.config['camera'].get('record_stream', True)

        # Resolve output directories
        video_out_rel = self.config['paths']['video_output']
        self.video_output_path = os.path.join(project_root, video_out_rel) if not os.path.isabs(video_out_rel) else video_out_rel

        image_out_rel = self.config['paths']['image_output_dir']
        self.image_output_dir = os.path.join(project_root, image_out_rel) if not os.path.isabs(image_out_rel) else image_out_rel

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.video_output_path), exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = session_timestamp
        self.session_dir = os.path.join(self.image_output_dir, session_timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self.capture_count = 0

        # Load LayoutParser model (ppyolov2_r50vd_dcn_365e_publaynet)
        print("Loading LayoutParser model (ppyolov2)...")
        try:
            self.layout_model = lp.PaddleDetectionLayoutModel(
                'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config'
            )
            print("LayoutParser model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Failed to load LayoutParser model: {e}. Fallback to manual split.", file=sys.stderr)
            self.layout_model = None

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def is_blurry(self, image, threshold=10.0):
        """Helper to check blurriness of captured frame."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var, laplacian_var < threshold

    def find_dynamic_split_line(self, layout_result, img_width):
        """
        Dynamically finds the optimal vertical line to split a double-page document spread
        based on the horizontal distribution (histogram) of detected text blocks.
        """
        # 1. Create a 1D text accumulation histogram across the width
        x_histogram = np.zeros(img_width, dtype=np.int32)
        accepted_types = {'text', 'paragraph', 'body'}
        body_blocks = [b for b in layout_result if b.type.lower() in accepted_types]

        if not body_blocks:
            default_split = img_width // 2
            print(f"[Dynamic Split] No text blocks found. Fallback split point: {default_split}")
            return default_split

        # Accumulate coordinate intervals into histogram
        for b in body_blocks:
            x_min = int(max(0, min(b.coordinates[0], img_width)))
            x_max = int(max(0, min(b.coordinates[2], img_width)))
            x_histogram[x_min:x_max] += 1

        # 2. Limit the valley search to the central 40% - 60% ROI of the image
        roi_start = int(img_width * 0.40)
        roi_end = int(img_width * 0.60)
        roi_histogram = x_histogram[roi_start:roi_end]

        # 3. Find the lowest overlap index closest to the center
        min_overlap = np.min(roi_histogram)
        min_indices = np.where(roi_histogram == min_overlap)[0] + roi_start
        center = img_width // 2
        
        # Select the candidate index that minimizes distance to the physical center
        best_split_x = min_indices[np.argmin(np.abs(min_indices - center))]

        # 4. Fallback condition: If the valley is not clear (min_overlap > 1)
        if min_overlap <= 1:
            split_x = int(best_split_x)
            print(f"[Dynamic Split] Detected split line at x={split_x} (overlap={min_overlap}).")
        else:
            split_x = img_width // 2
            print(f"[Dynamic Split] WARNING: No clear valley (min_overlap={min_overlap}). Fallback to center x={split_x}.")

        # 5. Output ASCII text density histogram in console logs
        print("\n[Dynamic Split Log] X-Histogram ASCII density (40% - 60% range):")
        step = max(1, (roi_end - roi_start) // 20)
        for i in range(roi_start, roi_end, step):
            val = x_histogram[i]
            bar = "#" * val
            print(f"  x={i:4d} | {bar:<10} ({val})")
        print()

        return split_x

    def run(self, record_stream=False):
        """
        Runs the webcam live stream.
        :param record_stream: If True, saves the entire stream to video_output_path.
        """
        print(f"Opening camera index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual camera resolution: {actual_width}x{actual_height}")

        out = None
        if record_stream:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            print(f"Full stream recording enabled. Saving to: {self.video_output_path}")
            out = cv2.VideoWriter(self.video_output_path, fourcc, self.fps, (actual_width, actual_height))

        if not cap.isOpened():
            print("Error: Could not open webcam stream.", file=sys.stderr)
            return

        print("\n=== Control Commands ===")
        print(" - Press 'c' on the video window to CAPTURE the current frame.")
        print(" - Press 'q' on the video window to QUIT.")
        print("========================\n")
        
        # Load configurable max frames from config
        enhance_cfg = self.config.get('enhancement', {})
        stacking_cfg = enhance_cfg.get('stacking', {})
        max_frames_cfg = max(2, int(stacking_cfg.get('max_frames', 50)))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.", file=sys.stderr)
                    break

                # Draw info overlay on a copy for display (optional)
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press 'c' to Capture & Stack | 'q' to Quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show stream
                cv2.imshow('Webcam Live Stream', display_frame)

                # Save to full stream video if recording is enabled
                if out is not None:
                    out.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit command received.")
                    break
                elif key == ord('c') or key == ord('s'):
                    # Stacking Capture Triggered!
                    capture_dir = os.path.join(self.session_dir, f"{self.capture_count:04d}")
                    os.makedirs(capture_dir, exist_ok=True)
                    
                    raw_save_path = os.path.join(capture_dir, "raw.png")
                    enhanced_save_path = os.path.join(capture_dir, "enhanced.png")
                    
                    print(f"\n[Stacking Capture] Gathering {max_frames_cfg} frames from this moment...")
                    
                    # Gather frames starting now
                    captured_frames = []
                    
                    for _ in range(max_frames_cfg):
                        ret_f, f = cap.read()
                        if ret_f:
                            captured_frames.append(f)
                            # Display overlay status on the window
                            overlay = f.copy()
                            cv2.putText(overlay, f"Capturing: {len(captured_frames)}/{max_frames_cfg}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow('Webcam Live Stream', overlay)
                            cv2.waitKey(1)

                    # Merge frames in grayscale
                    all_stack_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in captured_frames]
                    
                    if len(all_stack_frames) > 0:
                        enhance_cfg = self.config.get('enhancement', {})
                        stacking_cfg = enhance_cfg.get('stacking', {})
                        scale = max(1, int(stacking_cfg.get('scale_factor', 2)))
                        
                        print(f"Aligning and Stacking {len(all_stack_frames)} frames using ECC at {scale}x resolution...")
                        
                        ref_idx = int(np.argmax([cv2.Laplacian(img, cv2.CV_64F).var() for img in all_stack_frames]))
                        ref_frame = all_stack_frames[ref_idx]
                        print(f" -> Chosen frame {ref_idx}/{len(all_stack_frames)} as reference (Sharpness: {cv2.Laplacian(ref_frame, cv2.CV_64F).var():.2f})")
                        
                        # Save raw image
                        cv2.imwrite(raw_save_path, captured_frames[ref_idx])
                        print(f"💾 Saved raw reference frame to: {raw_save_path}")
                        
                        h, w = ref_frame.shape
                        
                        h_scaled, w_scaled = h * scale, w * scale
                        stacked_float_scaled = np.zeros((h_scaled, w_scaled), dtype=np.float32)
                        
                        ref_frame_scaled = cv2.resize(ref_frame, (w_scaled, h_scaled), interpolation=cv2.INTER_LANCZOS4)
                        stacked_float_scaled += ref_frame_scaled.astype(np.float32)
                        
                        aligned_count = 1
                        warp_mode = cv2.MOTION_EUCLIDEAN
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
                        
                        for idx in range(1, len(all_stack_frames)):
                            curr_frame = all_stack_frames[idx]
                            warp_matrix = np.eye(2, 3, dtype=np.float32)
                            try:
                                cc, warp_matrix = cv2.findTransformECC(
                                    ref_frame,
                                    curr_frame,
                                    warp_matrix,
                                    warp_mode,
                                    criteria,
                                    None,
                                    5
                                )
                                warp_matrix_scaled = warp_matrix.copy()
                                warp_matrix_scaled[0, 2] *= float(scale)
                                warp_matrix_scaled[1, 2] *= float(scale)
                                
                                curr_frame_scaled = cv2.resize(curr_frame, (w_scaled, h_scaled), interpolation=cv2.INTER_LANCZOS4)
                                
                                aligned_frame_scaled = cv2.warpAffine(
                                    curr_frame_scaled,
                                    warp_matrix_scaled,
                                    (w_scaled, h_scaled),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE
                                )
                                stacked_float_scaled += aligned_frame_scaled.astype(np.float32)
                                aligned_count += 1
                            except cv2.error:
                                continue
                                
                        print(f" -> Successfully aligned {aligned_count}/{len(all_stack_frames)} frames.")
                        stacked_scaled = stacked_float_scaled / aligned_count
                        stacked_scaled = np.clip(stacked_scaled, 0, 255).astype(np.uint8)

                        # Apply sharpening if enabled in configuration
                        if enhance_cfg.get('enabled', True) and enhance_cfg.get('sharpening', True):
                            strength = enhance_cfg.get('strength', 'strong')
                            if strength == "unsharp":
                                print("Applying CLAHE and unsharp masking filter to stacked image...")
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                                clahe_scaled = clahe.apply(stacked_scaled)
                                ksize = 5 if scale <= 2 else (9 if scale == 3 else 13)
                                gaussian_blur = cv2.GaussianBlur(clahe_scaled, (ksize, ksize), 1.5)
                                process_frame = cv2.addWeighted(clahe_scaled, 2.2, gaussian_blur, -1.2, 0)
                            else:
                                print(f"Applying real-time {strength} sharpening to stacked image...")
                                if strength == "strong":
                                    kernel = np.array([
                                        [-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]
                                    ])
                                else:
                                    kernel = np.array([
                                        [ 0, -1,  0],
                                        [-1,  5, -1],
                                        [ 0, -1,  0]
                                    ])
                                process_frame = cv2.filter2D(stacked_scaled, -1, kernel)
                        else:
                            process_frame = stacked_scaled

                        # Save high resolution frame
                        cv2.imwrite(enhanced_save_path, process_frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                        print(f"💾 Saved stacked & sharpened frame to: {enhanced_save_path}")

                        # Run Layout-aware splitting and OCR
                        print("Running Document AI layout analysis & split on captured frame...")
                        try:
                            from functions.ocr.pipeline import run_ocr_pipeline, draw_and_save

                            verbose = self.config.get('logging', {}).get('verbose', True)
                            track_stats = self.config.get('logging', {}).get('resource_tracking', True)

                            # Check config toggle for dynamic split
                            use_dynamic_split = self.config.get('layout', {}).get('dynamic_split', True)

                            if use_dynamic_split:
                                h_proc, w_proc = process_frame.shape[:2]
                                split_x = w_proc // 2
                                y1, y2 = int(h_proc * 0.12), int(h_proc * 0.88)
                                detection_success = False

                                if self.layout_model is not None:
                                    try:
                                        layout = self.layout_model.detect(process_frame)
                                        split_x = self.find_dynamic_split_line(layout, w_proc)
                                        accepted_types = {'text', 'paragraph', 'body'}
                                        body_blocks = [b for b in layout if b.type.lower() in accepted_types]
                                        if body_blocks:
                                            y1_list = [int(b.coordinates[1]) for b in body_blocks]
                                            y2_list = [int(b.coordinates[3]) for b in body_blocks]
                                            y1 = max(0, min(y1_list))
                                            y2 = min(h_proc, max(y2_list))
                                            detection_success = True
                                            print(f"[LayoutParser] Detected main content bounds: y1={y1}, y2={y2}")
                                        else:
                                            print("[LayoutParser] No body blocks detected. Using fallback vertical bounds.")
                                    except Exception as e:
                                        print(f"WARNING: LayoutParser detection failed: {e}. Using fallback bounds.")
                                else:
                                    print("[LayoutParser] Layout model not loaded. Using fallback split and vertical bounds.")

                                # Slice into Left and Right cropped regions
                                left_cropped = process_frame[y1:y2, 0:split_x]
                                right_cropped = process_frame[y1:y2, split_x:w_proc]

                                # Save page crops
                                left_crop_path = os.path.join(capture_dir, "left.png")
                                right_crop_path = os.path.join(capture_dir, "right.png")
                                cv2.imwrite(left_crop_path, left_cropped)
                                cv2.imwrite(right_crop_path, right_cropped)
                                print(f"💾 Saved left page crop to: {left_crop_path}")
                                print(f"💾 Saved right page crop to: {right_crop_path}")

                                # Run OCR sequentially
                                ocr_start_time = time.time()
                                print("[OCR] Running PaddleOCR on Left Page...")
                                results_l, prep_l, text_l, stats_l = run_ocr_pipeline(
                                    left_cropped,
                                    self.config,
                                    verbose=verbose,
                                    track_stats=track_stats
                                )
                                print("[OCR] Running PaddleOCR on Right Page...")
                                results_r, prep_r, text_r, stats_r = run_ocr_pipeline(
                                    right_cropped,
                                    self.config,
                                    verbose=verbose,
                                    track_stats=track_stats
                                )
                                ocr_duration = time.time() - ocr_start_time
                                from functions.ocr.pipeline import reconstruct_paragraphs
                                paragraphs_l, last_is_open = reconstruct_paragraphs(results_l, self.config, is_left_page=True)
                                paragraphs_r, _ = reconstruct_paragraphs(results_r, self.config, is_left_page=False)
                                
                                if last_is_open and paragraphs_l and paragraphs_r:
                                    merged_para = paragraphs_l[-1] + " " + paragraphs_r[0]
                                    paragraphs = paragraphs_l[:-1] + [merged_para] + paragraphs_r[1:]
                                else:
                                    paragraphs = paragraphs_l + paragraphs_r
                                
                                final_text = "\n\n".join(paragraphs)

                                # Construct bbox.txt content for split page mode
                                bbox_lines = []
                                bbox_lines.append("--- Left Page ---")
                                if results_l:
                                    for i, res in enumerate(results_l, 1):
                                        bbox_lines.append(f'{i}: "{res[1]}"')
                                else:
                                    bbox_lines.append("[No text detected]")
                                bbox_lines.append("")
                                bbox_lines.append("--- Right Page ---")
                                if results_r:
                                    for i, res in enumerate(results_r, 1):
                                        bbox_lines.append(f'{i}: "{res[1]}"')
                                else:
                                    bbox_lines.append("[No text detected]")
                                bbox_text = "\n".join(bbox_lines)

                                # Save marked/context images if configured
                                if self.config.get('logging', {}).get('save_marked', True):
                                    if results_l:
                                        left_marked_path = os.path.join(capture_dir, "left_context.png")
                                        draw_and_save(np.array(prep_l), results_l, left_marked_path, verbose=verbose)
                                    if results_r:
                                        right_marked_path = os.path.join(capture_dir, "right_context.png")
                                        draw_and_save(np.array(prep_r), results_r, right_marked_path, verbose=verbose)
                            else:
                                # Single page mode (No splitting)
                                print("[OCR] Running single-page mode (dynamic split disabled)...")
                                ocr_start_time = time.time()
                                results, prep, final_text, stats = run_ocr_pipeline(
                                    process_frame,
                                    self.config,
                                    verbose=verbose,
                                    track_stats=track_stats
                                )
                                ocr_duration = time.time() - ocr_start_time

                                # Construct bbox.txt content for single page mode
                                bbox_lines = []
                                if results:
                                    for i, res in enumerate(results, 1):
                                        bbox_lines.append(f'{i}: "{res[1]}"')
                                else:
                                    bbox_lines.append("[No text detected]")
                                bbox_text = "\n".join(bbox_lines)

                                # Save marked/context image if configured
                                if self.config.get('logging', {}).get('save_marked', True) and results:
                                    marked_path = os.path.join(capture_dir, "context.png")
                                    draw_and_save(np.array(prep), results, marked_path, verbose=verbose)

                            # Print OCR result to terminal
                            print("\n" + "="*40)
                            print("★ OCR RESULT ★")
                            print("="*40)
                            if final_text.strip():
                                print(final_text)
                            else:
                                print("[No text detected]")
                            print("="*40 + "\n")

                            # Save OCR result to txt file
                            txt_save_path = os.path.join(capture_dir, "ocr.txt")
                            with open(txt_save_path, 'w', encoding='utf-8') as f:
                                f.write(final_text)
                            print(f"💾 OCR text saved to: {txt_save_path}")

                            # Save bounding boxes text file
                            bbox_save_path = os.path.join(capture_dir, "bbox.txt")
                            with open(bbox_save_path, 'w', encoding='utf-8') as f:
                                f.write(bbox_text)
                            print(f"💾 Bounding box text saved to: {bbox_save_path}")

                            # Save text payload to stream/page_{page_num:04d}.json and update stream/metadata.json atomically
                            try:
                                import json
                                func_dir = os.path.dirname(os.path.abspath(__file__))
                                proj_root = os.path.dirname(func_dir)
                                stream_dir = os.path.join(proj_root, "stream")
                                os.makedirs(stream_dir, exist_ok=True)
                                
                                page_json_path = os.path.join(stream_dir, f"page_{self.capture_count:04d}.json")
                                payload = {
                                    "capture_id": f"{self.session_timestamp}_{self.capture_count:04d}",
                                    "text": final_text,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                with open(page_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(payload, f, ensure_ascii=False, indent=4)
                                print(f"💾 Saved page payload to: {page_json_path}")
                                
                                # Write metadata.json atomically using temp file and replace
                                meta_path = os.path.join(stream_dir, "metadata.json")
                                temp_meta_path = os.path.join(stream_dir, "metadata.tmp.json")
                                meta_payload = {
                                    "latest_page": self.capture_count,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                with open(temp_meta_path, 'w', encoding='utf-8') as f:
                                    json.dump(meta_payload, f, ensure_ascii=False, indent=4)
                                os.replace(temp_meta_path, meta_path)
                                print(f"💾 Atomically updated metadata to: {meta_path}")
                            except Exception as e:
                                print(f"WARNING: Failed to save monitor JSON payload or metadata: {e}")

                            if track_stats:
                                print(f" - Pure OCR Time: {ocr_duration:.2f}s")

                        except Exception as e:
                            print(f"Failed to run OCR layout split pipeline: {e}", file=sys.stderr)
                    else:
                        print("Error: No frames collected for stacking.", file=sys.stderr)

                    self.capture_count += 1
                    
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            print("Resources released. Stream stopped.")

