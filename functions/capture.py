#!/usr/bin/env python3
import os
import sys
import time
import queue
import threading
import json
import multiprocessing
from collections import deque
import cv2
import yaml
import numpy as np
import ssl

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

        # Check if development.reprocess_latest is enabled
        dev_cfg = self.config.get('development', {})
        self.reprocess_mode = dev_cfg.get('reprocess_latest', False)
        self.custom_split_x = None

        # Parse flip option
        self.flip = self.config.get('camera', {}).get('flip', 0)

        image_out_rel = self.config.get('paths', {}).get('image_output_dir', 'data/')
        self.image_output_dir = os.path.join(project_root, image_out_rel) if not os.path.isabs(image_out_rel) else image_out_rel

        if self.reprocess_mode:
            print("[Offline Mode] Initializing in offline reprocessing mode. Camera and background worker initialization skipped.")
            return

        # Parse config parameters
        self.camera_index = self.config.get('camera', {}).get('index', 0)
        self.width = self.config.get('camera', {}).get('resolution', {}).get('width', 1920)
        self.height = self.config.get('camera', {}).get('resolution', {}).get('height', 1080)
        self.fps = self.config.get('camera', {}).get('fps', 30)
        self.record_stream = self.config.get('camera', {}).get('record_stream', True)

        # Resolve output directories
        video_out_rel = self.config.get('paths', {}).get('video_output', 'stream/temp_output.mp4')
        self.video_output_path = os.path.join(project_root, video_out_rel) if not os.path.isabs(video_out_rel) else video_out_rel

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.video_output_path), exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = session_timestamp
        self.session_dir = os.path.join(self.image_output_dir, session_timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self.capture_count = 0

        # Load configurable max frames from config to setup rolling frame buffer
        enhance_cfg = self.config.get('enhancement', {})
        stacking_cfg = enhance_cfg.get('stacking', {})
        self.max_frames = max(2, int(stacking_cfg.get('max_frames', 50)))

        if self.reprocess_mode:
            print("[Offline Mode] Initializing in offline reprocessing mode. Camera and background worker initialization skipped.")
            return

        # Async Threading Queue & Worker State
        self.task_queue = queue.Queue()
        self.status_queue = multiprocessing.Queue()
        self.ocr_queue = multiprocessing.Queue()
        self.q_size = 0
        self.processing_status = "Idle"
        
        # Start background OCR process
        self.ocr_process = multiprocessing.Process(
            target=ocr_worker_proc, 
            args=(self.ocr_queue, self.status_queue, self.config), 
            daemon=True
        )
        self.ocr_process.start()
        
        # Start background saver thread
        self.worker_thread = threading.Thread(
            target=self.saver_worker_proc, 
            daemon=True
        )
        self.worker_thread.start()

    def load_config(self):
        """Loads and parses the YAML configuration file from the defined config path."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def is_blurry(self, image, threshold=10.0):
        """
        Determines if an image is blurry by calculating the variance of its Laplacian.
        Returns:
            (laplacian_var, is_blurry_bool)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var, laplacian_var < threshold

    def apply_flip(self, frame):
        """
        Applies rotation or flip options to the frame based on camera configuration.
        Supported flips: 90, 180, 270, 'horizontal', 'vertical'.
        """
        if frame is None:
            return frame
        if self.flip == 180 or self.flip == "180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.flip == 90 or self.flip == "90":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.flip == 270 or self.flip == "270":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.flip == "horizontal":
            return cv2.flip(frame, 1)
        elif self.flip == "vertical":
            return cv2.flip(frame, 0)
        return frame

    def reprocess_latest_data(self):
        """
        Locates the latest session directory under data/, clears all intermediate products
        (everything except raw.png) in all capture subfolders (0000, 0001, etc.), and reprocesses them.
        """
        import glob
        session_dirs = sorted([d for d in glob.glob(os.path.join(self.image_output_dir, "*")) if os.path.isdir(d)])
        if not session_dirs:
            print("Error: No session directories found in image output directory.", file=sys.stderr)
            return

        target_session_dir = session_dirs[-1]
        session_timestamp = os.path.basename(target_session_dir)
        print(f"[Reprocess Session] Processing all capture folders in latest session: {target_session_dir}")

        capture_dirs = sorted([d for d in glob.glob(os.path.join(target_session_dir, "*")) if os.path.isdir(d)])
        if not capture_dirs:
            print(f"Error: No capture directories found in session {target_session_dir}", file=sys.stderr)
            return

        for cap_dir in capture_dirs:
            try:
                cap_idx = int(os.path.basename(cap_dir))
            except ValueError:
                continue

            raw_path = os.path.join(cap_dir, "raw.png")
            if not os.path.exists(raw_path):
                print(f"[Reprocess Session] Skipping {cap_dir} - raw.png not found.")
                continue

            print(f"\n[Reprocess Session] Clearing intermediate files in capture {cap_idx:04d}...")
            # Delete intermediate files except raw.png, left.png, and right.png
            for f in os.listdir(cap_dir):
                f_path = os.path.join(cap_dir, f)
                if os.path.isfile(f_path) and f not in ("raw.png", "left.png", "right.png"):
                    try:
                        os.remove(f_path)
                    except Exception as e:
                        print(f"  -> Failed to delete {f}: {e}")

            img = cv2.imread(raw_path)
            if img is None:
                print(f"Error: Could not read image at {raw_path}", file=sys.stderr)
                continue

            # raw.png is already saved with the flip/rotation applied, so do not apply it again here.
            print(f"[Reprocess Session] Running pipeline on {raw_path}...")
            self.execute_capture_task(
                capture_index=cap_idx,
                captured_frames=[img],
                config=self.config,
                session_dir=target_session_dir,
                session_timestamp=session_timestamp
            )
        print("[Reprocess Latest] Finished batch reprocessing.")

    def run(self, record_stream=False):
        """
        Runs the webcam live stream with instant non-blocking capture and overlay metrics.
        :param record_stream: If True, saves the entire stream to video_output_path.
        """
        # Check if development.reprocess_latest is enabled
        dev_cfg = self.config.get('development', {})
        if dev_cfg.get('reprocess_latest', False):
            print("\n[Reprocess Latest] Running in offline development mode.")
            self.reprocess_latest_data()
            return

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
        print(" - CLICK on the video window to select a CUSTOM vertical split line.")
        print("========================\n")
        
        cv2.namedWindow('Webcam Live Stream')
        self.custom_split_x = None
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.custom_split_x = x
                print(f"[Split Line Selection] User selected split line at x={x}")
        cv2.setMouseCallback('Webcam Live Stream', mouse_callback)

        # Initialize rolling frame buffer
        frame_buffer = deque(maxlen=self.max_frames)
        consecutive_failures = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # Allow up to 30 consecutive failed frames (~1 second)
                        print("Error: Failed to grab frame consecutively. Exiting loop.", file=sys.stderr)
                        break
                    cv2.waitKey(33)  # Wait ~33ms and try again
                    continue
                
                consecutive_failures = 0

                # Apply camera flip/rotation configuration
                frame = self.apply_flip(frame)

                # Maintain history of frames for stacking
                frame_buffer.append(frame.copy())

                # Check for worker status updates non-blockingly
                try:
                    while not self.status_queue.empty():
                        status = self.status_queue.get_nowait()
                        if status[0] == "done":
                            self.q_size = max(0, self.q_size - 1)
                            self.processing_status = "Idle"
                        elif status[0] == "started":
                            self.processing_status = f"Processing #{status[1]}"
                except Exception:
                    pass

                # Draw info overlay on a copy for display (optional)
                display_frame = frame.copy()
                
                # Overlay layout
                cv2.putText(display_frame, "Press 'c' to Capture & Stack | 'q' to Quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Print queue size and processing status
                status_color = (0, 255, 255) if self.q_size > 0 else (0, 255, 0)
                status_msg = f"Queue Size: {self.q_size} | Worker: {self.processing_status}"
                cv2.putText(display_frame, status_msg, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Draw split line
                if self.custom_split_x is not None:
                    cv2.line(display_frame, (self.custom_split_x, 0), (self.custom_split_x, actual_height), (0, 0, 255), 2)
                else:
                    cv2.line(display_frame, (actual_width // 2, 0), (actual_width // 2, actual_height), (0, 255, 255), 1)

                # Show stream
                cv2.imshow('Webcam Live Stream', display_frame)

                # Check if the window was closed (returns < 1 or -1 on close)
                if cv2.getWindowProperty('Webcam Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user.")
                    self.task_queue.put(None)
                    break

                # Save to full stream video if recording is enabled
                if out is not None:
                    out.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit command received.")
                    # Put sentinel to stop worker process
                    self.task_queue.put(None)
                    break
                elif key == ord('c') or key == ord('s'):
                    # Capture current snapshot of the rolling frames
                    captured_frames = list(frame_buffer)
                    if len(captured_frames) > 0:
                        task_info = {
                            'capture_index': self.capture_count,
                            'frames': captured_frames,
                            'custom_split_x': self.custom_split_x
                        }
                        self.task_queue.put(task_info)
                        self.q_size += 1
                        print(f"[Queue] Enqueued capture #{self.capture_count} (Queue Size: {self.q_size})")
                        self.capture_count += 1
                    else:
                        print("Error: Frame buffer is empty, cannot capture.", file=sys.stderr)
                    
        except KeyboardInterrupt:
            print("\n[SIGINT] KeyboardInterrupt received. Terminating background workers immediately...")
            if hasattr(self, 'ocr_process') and self.ocr_process.is_alive():
                self.ocr_process.terminate()
                self.ocr_process.join()
            if hasattr(self, 'task_queue'):
                try:
                    while not self.task_queue.empty():
                        self.task_queue.get_nowait()
                except Exception:
                    pass
                self.task_queue.put(None)
        finally:
            # Signal worker thread to stop
            if hasattr(self, 'task_queue'):
                self.task_queue.put(None)
            
            # Wait for worker thread to finish enqueued saves (if not already terminated)
            if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
                print("Finishing remaining image enqueues & saves...")
                self.worker_thread.join(timeout=2.0)
            
            # Wait for background OCR process to finish (if not already terminated)
            if hasattr(self, 'ocr_process') and self.ocr_process.is_alive():
                if self.q_size > 0:
                    print(f"Waiting for remaining {self.q_size} OCR tasks to finish. Press Ctrl+C to force terminate...")
                    try:
                        self.ocr_process.join()
                    except KeyboardInterrupt:
                        print("\n[Force Quit] Terminating background OCR process...")
                        self.ocr_process.terminate()
                        self.ocr_process.join()
                else:
                    self.ocr_process.join(timeout=2.0)
            
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            print("Resources released. Stream stopped.")

    def saver_worker_proc(self):
        """Saver thread function running in main process: stacks, pre-processes, and saves images, then enqueues OCR tasks."""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    # Signal background OCR process to exit
                    self.ocr_queue.put(None)
                    break
                
                capture_index = task['capture_index']
                captured_frames = task['frames']
                custom_split_x = task.get('custom_split_x', None)
                
                # Perform fast saving, stacking, and splitting
                preprocess_results = WebcamCapturer.save_and_preprocess_task(
                    capture_index, captured_frames, self.config, self.session_dir, self.session_timestamp, custom_split_x
                )
                if preprocess_results:
                    # Enqueue the path info to the multiprocessing OCR queue
                    self.ocr_queue.put(preprocess_results)
            except Exception as e:
                print(f"[Saver Thread Error] Failed to stack/save capture: {e}", file=sys.stderr)

    @staticmethod
    def save_and_preprocess_task(capture_index, captured_frames, config, session_dir, session_timestamp, custom_split_x=None):
        """Performs fast frame stacking, alignment, bleed-through removal, layout splitting, and saving of images on disk."""
        task_start_t = time.time()
        stack_t = 0.0
        enhance_t = 0.0
        split_t = 0.0
        
        capture_dir = os.path.join(session_dir, f"{capture_index:04d}")
        os.makedirs(capture_dir, exist_ok=True)
        
        raw_save_path = os.path.join(capture_dir, "raw.png")
        enhanced_save_path = os.path.join(capture_dir, "enhanced.png")
        
        if len(captured_frames) == 0:
            print("Error: No frames collected for stacking.", file=sys.stderr)
            return None
            
        enhance_cfg = config.get('enhancement', {})
        stacking_cfg = enhance_cfg.get('stacking', {})
        stacking_enabled = stacking_cfg.get('enabled', True)
        scale = max(1, int(stacking_cfg.get('scale_factor', 2)))
        
        if stacking_enabled and len(captured_frames) > 1:
            t0 = time.time()
            all_stack_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in captured_frames]
            print(f"\n[Stacking - Async #{capture_index}] Aligning and Stacking {len(all_stack_frames)} frames using ECC...")
            
            ref_idx = int(np.argmax([cv2.Laplacian(img, cv2.CV_64F).var() for img in all_stack_frames]))
            ref_frame = all_stack_frames[ref_idx]
            
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
            stack_t = time.time() - t0
        else:
            t0 = time.time()
            print(f"\n[Stacking - Async #{capture_index}] Stacking disabled. Using single frame.")
            single_frame = captured_frames[-1]
            cv2.imwrite(raw_save_path, single_frame)
            print(f"💾 Saved raw single frame to: {raw_save_path}")
            
            gray_frame = cv2.cvtColor(single_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray_frame.shape
            h_scaled, w_scaled = h * scale, w * scale
            stacked_scaled = cv2.resize(gray_frame, (w_scaled, h_scaled), interpolation=cv2.INTER_LANCZOS4)
            stack_t = time.time() - t0

        # Apply bleed-through removal if configured
        if enhance_cfg.get('bleed_through_removal', False):
            print("Applying background subtraction and grayscale normalization to remove bleed-through...")
            bt_cfg = enhance_cfg.get('bleed_through', {})
            k_size = int(bt_cfg.get('kernel_size', 21))
            if k_size % 2 == 0: k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            background = cv2.dilate(stacked_scaled, kernel)
            stacked_scaled = cv2.divide(stacked_scaled, background, scale=255)

        # Apply sharpening if enabled
        t0 = time.time()
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

        # Save enhanced frame
        cv2.imwrite(enhanced_save_path, process_frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        print(f"💾 Saved stacked & sharpened frame to: {enhanced_save_path}")
        enhance_t = time.time() - t0

        # Run Layout-aware splitting
        use_dynamic_split = config.get('layout', {}).get('dynamic_split', True)
        left_crop_path = os.path.join(capture_dir, "left.png")
        right_crop_path = os.path.join(capture_dir, "right.png")
        
        if use_dynamic_split:
            t0 = time.time()
            h_proc, w_proc = stacked_scaled.shape[:2]
            if custom_split_x is not None:
                split_x = int(custom_split_x * scale)
            else:
                split_x = w_proc // 2
            
            left_cropped_raw = stacked_scaled[:, 0:split_x]
            right_cropped_raw = stacked_scaled[:, split_x:w_proc]
            
            cv2.imwrite(left_crop_path, left_cropped_raw)
            print(f"💾 Saved raw left page crop to: {left_crop_path}")
            cv2.imwrite(right_crop_path, right_cropped_raw)
            print(f"💾 Saved raw right page crop to: {right_crop_path}")
            split_t = time.time() - t0
            
        return {
            'capture_index': capture_index,
            'capture_dir': capture_dir,
            'session_timestamp': session_timestamp,
            'use_dynamic_split': use_dynamic_split,
            'left_crop_path': left_crop_path if use_dynamic_split else None,
            'right_crop_path': right_crop_path if use_dynamic_split else None,
            'enhanced_save_path': enhanced_save_path,
            'task_start_t': task_start_t,
            'stack_t': stack_t,
            'enhance_t': enhance_t,
            'split_t': split_t if use_dynamic_split else 0.0
        }

    @staticmethod
    def execute_capture_task(capture_index, captured_frames, config, session_dir, session_timestamp, custom_split_x=None):
        """Executes stacking/preprocessing and OCR sequentially (used in reprocessing mode)."""
        preprocess_results = WebcamCapturer.save_and_preprocess_task(
            capture_index, captured_frames, config, session_dir, session_timestamp, custom_split_x
        )
        if preprocess_results:
            execute_ocr_task(preprocess_results, config)


def ocr_worker_proc(ocr_queue, status_queue, config):
    """Worker process function: runs background OCR tasks and pre-warms the OCR engine in a separate process."""
    # Restrict thread pools inside background process to prevent CPU choking/leaks
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    try:
        print("\n[OCR Pre-warm] Pre-warming PaddleOCR engine in background process...")
        from functions.ocr.pipeline import get_engine
        engine = get_engine(config, verbose=True)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        engine.read_text(dummy_img)
        print("[OCR Pre-warm] Background process pre-warming completed. Ready for captures!\n")
    except Exception as e:
        print(f"[OCR Pre-warm] Failed to pre-warm engine in background process: {e}", file=sys.stderr)

    while True:
        try:
            task = ocr_queue.get()
            if task is None:
                break
            
            capture_index = task['capture_index']
            status_queue.put(("started", capture_index))
            execute_ocr_task(task, config)
            status_queue.put(("done", capture_index))
        except Exception as e:
            print(f"[Worker Process Error] Failed to process OCR: {e}", file=sys.stderr)


def execute_ocr_task(preprocess_results, config):
    """Performs the OCR portion of the capture task, running in the background process."""
    capture_index = preprocess_results['capture_index']
    capture_dir = preprocess_results['capture_dir']
    session_timestamp = preprocess_results['session_timestamp']
    use_dynamic_split = preprocess_results['use_dynamic_split']
    left_crop_path = preprocess_results.get('left_crop_path')
    right_crop_path = preprocess_results.get('right_crop_path')
    enhanced_save_path = preprocess_results.get('enhanced_save_path')
    task_start_t = preprocess_results['task_start_t']
    stack_t = preprocess_results['stack_t']
    enhance_t = preprocess_results['enhance_t']
    split_t = preprocess_results['split_t']
    
    ocr_left_t = 0.0
    ocr_right_t = 0.0
    ocr_post_t = 0.0
    
    try:
        from functions.ocr.pipeline import run_ocr_pipeline, draw_and_save

        verbose = config.get('logging', {}).get('verbose', True)
        track_stats = config.get('logging', {}).get('resource_tracking', True)

        if use_dynamic_split:
            left_cropped = cv2.imread(left_crop_path)
            right_cropped = cv2.imread(right_crop_path)
            
            if left_cropped is None or right_cropped is None:
                print(f"Error: Could not read split pages at {left_crop_path} or {right_crop_path}", file=sys.stderr)
                return
            
            img_cfg = config.get('image', {})
            if img_cfg.get('preprocess', True):
                max_size = img_cfg.get('max_size', 1500)
                for crop_img in (left_cropped, right_cropped):
                    if crop_img is not None and max(crop_img.shape[:2]) > max_size:
                        h_c, w_c = crop_img.shape[:2]
                        scale_c = max_size / max(h_c, w_c)
                        if crop_img is left_cropped:
                            left_cropped = cv2.resize(left_cropped, (int(w_c * scale_c), int(h_c * scale_c)), interpolation=cv2.INTER_AREA)
                        else:
                            right_cropped = cv2.resize(right_cropped, (int(w_c * scale_c), int(h_c * scale_c)), interpolation=cv2.INTER_AREA)

            print("[OCR] Running PaddleOCR on Left Page...")
            t0 = time.time()
            results_l_text, prep_l, text_l, stats_l = run_ocr_pipeline(
                left_cropped,
                config,
                verbose=verbose,
                track_stats=track_stats
            )
            ocr_left_t = time.time() - t0

            print("[OCR] Running PaddleOCR on Right Page...")
            t0 = time.time()
            results_r_text, prep_r, text_r, stats_r = run_ocr_pipeline(
                right_cropped,
                config,
                verbose=verbose,
                track_stats=track_stats
            )
            ocr_right_t = time.time() - t0

            t0 = time.time()
            from functions.ocr.pipeline import reconstruct_paragraphs
            paragraphs_l, last_is_open = reconstruct_paragraphs(results_l_text, config, is_left_page=True, image_height=left_cropped.shape[0])
            paragraphs_r, _ = reconstruct_paragraphs(results_r_text, config, is_left_page=False, image_height=right_cropped.shape[0])
            
            if last_is_open and paragraphs_l and paragraphs_r:
                merged_para = paragraphs_l[-1] + " " + paragraphs_r[0]
                paragraphs = paragraphs_l[:-1] + [merged_para] + paragraphs_r[1:]
            else:
                paragraphs = paragraphs_l + paragraphs_r
            
            final_text = "\n\n".join(paragraphs)

            bbox_lines = []
            bbox_lines.append("--- Left Page ---")
            if results_l_text:
                for i, res in enumerate(results_l_text, 1):
                    t_val = res[1][0] if isinstance(res[1], tuple) else res[1]
                    bbox_lines.append(f'{i}: "{t_val}"')
            else:
                bbox_lines.append("[No text detected]")
            bbox_lines.append("")
            bbox_lines.append("--- Right Page ---")
            if results_r_text:
                for i, res in enumerate(results_r_text, 1):
                    t_val = res[1][0] if isinstance(res[1], tuple) else res[1]
                    bbox_lines.append(f'{i}: "{t_val}"')
            else:
                bbox_lines.append("[No text detected]")
            bbox_text = "\n".join(bbox_lines)

            if config.get('logging', {}).get('save_marked', True):
                left_marked_path = os.path.join(capture_dir, "left_context.png")
                draw_and_save(np.array(prep_l), results_l_text, left_marked_path, verbose=verbose)
                
                right_marked_path = os.path.join(capture_dir, "right_context.png")
                draw_and_save(np.array(prep_r), results_r_text, right_marked_path, verbose=verbose)
            ocr_post_t = time.time() - t0
        else:
            print("[OCR] Running single-page mode (dynamic split disabled)...")
            enhanced_frame = cv2.imread(enhanced_save_path)
            if enhanced_frame is None:
                print(f"Error: Could not read enhanced image at {enhanced_save_path}", file=sys.stderr)
                return
                
            t0 = time.time()
            results, prep, final_text, stats = run_ocr_pipeline(
                enhanced_frame,
                config,
                verbose=verbose,
                track_stats=track_stats
            )
            ocr_left_t = time.time() - t0

            t0 = time.time()
            bbox_lines = []
            if results:
                for i, res in enumerate(results, 1):
                    t_val = res[1][0] if isinstance(res[1], tuple) else res[1]
                    bbox_lines.append(f'{i}: "{t_val}"')
            else:
                bbox_lines.append("[No text detected]")
            bbox_text = "\n".join(bbox_lines)

            if config.get('logging', {}).get('save_marked', True) and results:
                marked_path = os.path.join(capture_dir, "context.png")
                draw_and_save(np.array(prep), results, marked_path, verbose=verbose)
            ocr_post_t = time.time() - t0

        print("\n" + "="*40)
        print(f"★ OCR RESULT (Task #{capture_index}) ★")
        print("="*40)
        if final_text.strip():
            print(final_text)
        else:
            print("[No text detected]")
        print("="*40 + "\n")

        txt_save_path = os.path.join(capture_dir, "ocr.txt")
        with open(txt_save_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"💾 OCR text saved to: {txt_save_path}")

        bbox_save_path = os.path.join(capture_dir, "bbox.txt")
        with open(bbox_save_path, 'w', encoding='utf-8') as f:
            f.write(bbox_text)
        print(f"💾 Bounding box text saved to: {bbox_save_path}")

        try:
            func_dir = os.path.dirname(os.path.abspath(__file__))
            proj_root = os.path.dirname(func_dir)
            stream_dir = os.path.join(proj_root, "stream")
            os.makedirs(stream_dir, exist_ok=True)
            
            page_json_path = os.path.join(stream_dir, f"page_{capture_index:04d}.json")
            payload = {
                "capture_id": f"{session_timestamp}_{capture_index:04d}",
                "text": final_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(page_json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)
            print(f"💾 Saved page payload to: {page_json_path}")
            
            meta_path = os.path.join(stream_dir, "metadata.json")
            temp_meta_path = os.path.join(stream_dir, "metadata.tmp.json")
            meta_payload = {
                "latest_page": capture_index,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(temp_meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=4)
            os.replace(temp_meta_path, meta_path)
            print(f"💾 Atomically updated metadata to: {meta_path}")
        except Exception as e:
            print(f"WARNING: Failed to save monitor JSON payload or metadata: {e}")

        total_duration = time.time() - task_start_t
        print("\n" + "="*50)
        print(f"⏱️  PERFORMANCE PROFILE (Task #{capture_index}) ⏱️")
        print("="*50)
        print(f" - Image Stacking/Alignment: {stack_t:.3f}s")
        print(f" - Image Enhancement (Sharpen): {enhance_t:.3f}s")
        if use_dynamic_split:
            print(f" - Layout Crop & Splitting: {split_t:.3f}s")
            print(f" - OCR Left Page: {ocr_left_t:.3f}s")
            print(f" - OCR Right Page: {ocr_right_t:.3f}s")
            print(f" - OCR Post-processing: {ocr_post_t:.3f}s")
        else:
            print(f" - OCR Single Page: {ocr_left_t:.3f}s")
            print(f" - OCR Post-processing: {ocr_post_t:.3f}s")
        print(f" - Total Execution Time: {total_duration:.3f}s")
        print("="*50 + "\n")

    except Exception as e:
        print(f"Failed to run OCR layout split pipeline: {e}", file=sys.stderr)
