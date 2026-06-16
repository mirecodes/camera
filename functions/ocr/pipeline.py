import os
import sys
import time
import ssl
from typing import List, Tuple, Optional, Any
import numpy as np
import cv2
import psutil
from PIL import Image, ImageOps, ImageFilter

# Setup local model paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# functions/ocr/pipeline.py is 2 levels deep from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(FILE_DIR))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
PADDLE_MODEL_PATH = os.path.abspath(os.path.join(MODEL_ROOT, "paddle"))
EASY_MODEL_PATH = os.path.abspath(os.path.join(MODEL_ROOT, "easyocr"))

def setup_ocr_env():
    """Sets environment variables for redirecting engine model paths."""
    os.makedirs(PADDLE_MODEL_PATH, exist_ok=True)
    os.makedirs(EASY_MODEL_PATH, exist_ok=True)
    
    os.environ['PADDLE_HOME'] = PADDLE_MODEL_PATH
    os.environ['PADDLEX_HOME'] = PADDLE_MODEL_PATH
    os.environ['PADDLE_PDX_HOME'] = PADDLE_MODEL_PATH
    # Override HOME just in case, but avoid execve unless needed, to prevent import issues
    os.environ['HOME'] = PADDLE_MODEL_PATH

# Set environments immediately upon import
setup_ocr_env()

# Global SSL bypass for downloading models on restrictive networks
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Global instances
ENGINE = None
KIWI = None
PROCESS_OBJ = psutil.Process(os.getpid())

def get_kiwi() -> Any:
    """Initializes and returns a singleton instance of the Kiwi analyzer if available."""
    global KIWI
    if KIWI is None:
        try:
            from kiwipiepy import Kiwi
            KIWI = Kiwi()
        except ImportError:
            print("Warning: kiwipiepy is not installed. Spacing correction will be disabled.")
            KIWI = False
    return KIWI

def get_engine(config: dict, verbose: bool = False) -> Any:
    """Initializes and returns the OCR engine specified in the configuration."""
    global ENGINE
    if ENGINE is None:
        engine_type = config.get('ocr', {}).get('engine', 'easyocr').lower()
        if verbose:
            print(f"Initializing OCR Engine: {engine_type.upper()}...")
            
        if engine_type == 'paddleocr':
            from .paddle_ocr_engine import PaddleOCREngine
            ENGINE = PaddleOCREngine(config, PADDLE_MODEL_PATH, verbose=verbose)
        else:
            from .easy_ocr_engine import EasyOCREngine
            ENGINE = EasyOCREngine(config, EASY_MODEL_PATH, verbose=verbose)
    return ENGINE

def preprocess_image(img: Image.Image, verbose: bool = False) -> Image.Image:
    """Enhances text clarity using RGB-based autocontrast and sharpening filters."""
    if verbose:
        print(" -> Preprocessing: RGB Autocontrast + Sharpening")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def prepare_image(image_input: Any, config: dict, verbose: bool = False) -> Tuple[np.ndarray, Image.Image]:
    """Loads, crops, and pre-processes the target image for optimal OCR extraction."""
    img_cfg = config.get('image', {})
    max_size = img_cfg.get('max_size', 4000)
    
    if isinstance(image_input, str):
        if verbose:
            print(f"Loading image from file: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img = Image.fromarray(image_input)
        else:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image input type.")
        
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Apply center-crop based on percentage to remove margins
    crop_cfg = img_cfg.get('crop', {})
    if crop_cfg.get('enabled', False):
        crop_w, crop_h = crop_cfg.get('width_pct', 75.0), crop_cfg.get('height_pct', 90.0)
        new_w, new_h = int(width * (crop_w / 100.0)), int(height * (crop_h / 100.0))
        left, top = (width - new_w) // 2, (height - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        if verbose:
            print(f" -> Cropped to {img.width}x{img.height} ({crop_w}%x{crop_h}%)")
    
    if img_cfg.get('preprocess', True):
        img = preprocess_image(img, verbose=verbose)
        
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        
    return np.array(img), img

def get_resource_usage() -> Tuple[float, float, float]:
    """Returns (Memory MB, Process CPU User Time, Process CPU System Time)."""
    mem_mb = PROCESS_OBJ.memory_info().rss / (1024 * 1024)
    cpu_times = PROCESS_OBJ.cpu_times()
    return mem_mb, cpu_times.user, cpu_times.system

def find_split_line_from_boxes(results: list, W: int) -> int:
    """
    Finds the optimal vertical split line by analyzing horizontal overlap of OCR bounding boxes.
    """
    x_histogram = np.zeros(W, dtype=np.int32)
    for res in results:
        if len(res) < 2 or res[0] is None or len(res[0]) < 4:
            continue
        bbox = res[0]
        try:
            xs = [p[0] for p in bbox]
            x_min = int(max(0, min(xs)))
            x_max = int(max(0, min(max(xs), W)))
            x_histogram[x_min:x_max] += 1
        except Exception:
            continue

    roi_start = int(W * 0.40)
    roi_end = int(W * 0.60)
    roi_histogram = x_histogram[roi_start:roi_end]
    
    if len(roi_histogram) == 0:
        return W // 2
        
    min_overlap = np.min(roi_histogram)
    min_indices = np.where(roi_histogram == min_overlap)[0] + roi_start
    center = W // 2
    best_split_x = min_indices[np.argmin(np.abs(min_indices - center))]
    return int(best_split_x)

def get_page_data(results: list, W: int, H: int, is_left: bool, split_x: int) -> Tuple[list, List[str], Tuple[int, int, int, int]]:
    """
    Filters, shifts coordinates, classifies, and computes the body text crop box for a single page.
    Returns: (shifted_results, labels, (x1, y1, x2, y2) relative to process_frame)
    """
    page_results = []
    for res in results:
        if len(res) < 2 or res[0] is None or len(res[0]) < 4:
            continue
        bbox = res[0]
        try:
            xs = [p[0] for p in bbox]
            x_center = sum(xs) / len(xs)
            
            # Filter based on split line
            if is_left and x_center >= split_x:
                continue
            if not is_left and x_center < split_x:
                continue
                
            # Shift x coordinate relative to the cropped page origin
            x_offset = 0 if is_left else split_x
            shifted_bbox = [[p[0] - x_offset, p[1]] for p in bbox]
            
            page_results.append([shifted_bbox, res[1]])
        except Exception:
            continue
            
    # Classify boxes on this page
    labels = classify_boxes(page_results, H)
    
    # Compute outer bounding box of the body blocks
    body_xs = []
    body_ys = []
    for i, res in enumerate(page_results):
        if labels[i] == "body":
            bbox = res[0]
            body_xs.extend([p[0] for p in bbox])
            body_ys.extend([p[1] for p in bbox])
            
    # Define the crop region relative to process_frame
    page_w = split_x if is_left else (W - split_x)
    x_offset = 0 if is_left else split_x
    
    if body_xs and body_ys:
        margin = 25 # safety padding in pixels
        x1 = max(0, int(min(body_xs)) - margin)
        y1 = max(0, int(min(body_ys)) - margin)
        x2 = min(page_w, int(max(body_xs)) + margin)
        y2 = min(H, int(max(body_ys)) + margin)
        
        # Map back to process_frame coordinates
        crop_box = (x1 + x_offset, y1, x2 + x_offset, y2)
    else:
        # Fallback to full cropped page if no body text
        crop_box = (x_offset, 0, x_offset + page_w, H)
        
    return page_results, labels, crop_box

def draw_and_save_labeled(crop_img_np: np.ndarray, page_results: list, labels: List[str], crop_box: Tuple[int, int, int, int], is_left: bool, split_x: int, output_path: str, verbose: bool = False):
    """Generates visual debug image with bounding boxes on the cropped page."""
    img_cv = cv2.cvtColor(crop_img_np, cv2.COLOR_RGB2BGR)
    x_offset = 0 if is_left else split_x
    
    # crop_box is (x1_glob, y1_glob, x2_glob, y2_glob) relative to process_frame
    crop_x1 = crop_box[0] - x_offset
    crop_y1 = crop_box[1]
    
    for i, res in enumerate(page_results):
        if len(res) < 2 or res[0] is None or len(res[0]) < 1: continue
        bbox, text = res[0], res[1]
        label = labels[i]
        
        # Shift bbox relative to crop origin
        shifted_bbox = [[p[0] - crop_x1, p[1] - crop_y1] for p in bbox]
        
        # Color coding: body = green (0, 255, 0), header/footer = orange (0, 165, 255)
        color = (0, 255, 0)
        if label in ("header", "footer"):
            color = (0, 165, 255) # Orange in BGR
            
        pts = np.array(shifted_bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_cv, [pts], True, color, 2)
        
        try:
            label_text = f"{i+1}({label[0].upper()})"
            cv2.putText(img_cv, label_text, (int(shifted_bbox[0][0]), int(shifted_bbox[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except: pass
            
    cv2.imwrite(output_path, img_cv)
    if verbose:
        print(f" -> Result visualization saved to: {output_path}")

def classify_boxes(results: list, H: int) -> List[str]:
    """
    Classifies boxes as 'header', 'footer', or 'body' based on geometric layout.
    """
    labels = ["body"] * len(results)
    if not results or H <= 0:
        return labels
        
    box_infos = []
    for i, res in enumerate(results):
        if len(res) < 2 or res[0] is None or len(res[0]) < 4:
            box_infos.append(None)
            continue
        bbox = res[0]
        try:
            ys = [p[1] for p in bbox]
            y_top, y_bottom = min(ys), max(ys)
            height = y_bottom - y_top
            y_center = (y_top + y_bottom) / 2.0
            text = res[1][0] if isinstance(res[1], tuple) else res[1]
            box_infos.append({
                'index': i,
                'y_top': y_top,
                'y_bottom': y_bottom,
                'y_center': y_center,
                'height': height,
                'text': text
            })
        except Exception:
            box_infos.append(None)
            
    # Candidates
    header_candidates = []
    footer_candidates = []
    body_indices = []
    
    for info in box_infos:
        if info is None:
            continue
        if info['y_center'] < 0.12 * H:
            header_candidates.append(info)
        elif info['y_center'] > 0.88 * H:
            footer_candidates.append(info)
        else:
            body_indices.append(info['index'])
            
    # If there is no body text, we do not classify headers/footers to avoid deleting the only text on the page
    if not body_indices:
        return labels
        
    body_boxes = [box_infos[idx] for idx in body_indices]
    min_body_y = min(b['y_top'] for b in body_boxes)
    max_body_y = max(b['y_bottom'] for b in body_boxes)
    
    # Check header candidates
    for cand in header_candidates:
        gap = min_body_y - cand['y_bottom']
        if gap > 1.8 * cand['height']:
            labels[cand['index']] = "header"
        else:
            body_boxes.append(cand)
            min_body_y = min(b['y_top'] for b in body_boxes)
            
    # Check footer candidates
    for cand in footer_candidates:
        gap = cand['y_top'] - max_body_y
        if gap > 1.8 * cand['height']:
            labels[cand['index']] = "footer"
            
    return labels

def draw_and_save(image_np: np.ndarray, results: list, output_path: str, verbose: bool = False):
    """Generates a visual debug image with bounding boxes and detection indices."""
    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    H = image_np.shape[0]
    labels = classify_boxes(results, H)
    
    for i, res in enumerate(results):
        if len(res) < 2 or res[0] is None or len(res[0]) < 1: continue
        bbox, text = res[0], res[1]
        label = labels[i]
        
        # Color coding: body = green (0, 255, 0), header/footer = orange (0, 165, 255)
        color = (0, 255, 0)
        if label in ("header", "footer"):
            color = (0, 165, 255) # Orange in BGR
            
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_cv, [pts], True, color, 2)
        
        try:
            label_text = f"{i+1}({label[0].upper()})"
            cv2.putText(img_cv, label_text, (int(bbox[0][0]), int(bbox[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except: pass
            
    cv2.imwrite(output_path, img_cv)
    if verbose:
        print(f" -> Result visualization saved to: {output_path}")

def reconstruct_paragraphs(results: list, config: dict, is_left_page: bool = False, image_height: Optional[int] = None) -> Tuple[List[str], bool]:
    """
    Groups OCR bounding boxes into lines, sorts them, detects paragraph breaks
    based on layout properties (line width and vertical gaps), and applies spacing correction.
    
    Returns:
        (paragraphs_list, last_is_open_bool)
    """
    if not results:
        return [], False
        
    labels = ["body"] * len(results)
    if image_height is not None:
        labels = classify_boxes(results, image_height)
    
    valid_items = []
    for i, res in enumerate(results):
        if len(res) >= 2 and res[0] is not None and len(res[0]) >= 4:
            # Filter out headers and footers
            if labels[i] in ("header", "footer"):
                continue
                
            text = res[1]
            if isinstance(text, tuple):
                text = text[0]
            if text.strip():
                bbox = res[0]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_left, x_right = min(xs), max(xs)
                y_top, y_bottom = min(ys), max(ys)
                height = y_bottom - y_top
                width = x_right - x_left
                valid_items.append({
                    'text': text,
                    'x_left': x_left,
                    'x_right': x_right,
                    'y_top': y_top,
                    'y_bottom': y_bottom,
                    'height': height,
                    'width': width
                })
                
    if not valid_items:
        return [], False
        
    # Sort items top-to-bottom
    valid_items.sort(key=lambda item: item['y_top'])
    
    # Group into lines
    lines = []
    for item in valid_items:
        placed = False
        for line in lines:
            line_y_top = min(it['y_top'] for it in line)
            line_y_bottom = max(it['y_bottom'] for it in line)
            line_height = line_y_bottom - line_y_top
            
            overlap = min(item['y_bottom'], line_y_bottom) - max(item['y_top'], line_y_top)
            min_h = min(item['height'], line_height)
            if min_h > 0 and overlap > 0.45 * min_h:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
            
    # Sort lines by y_top
    lines.sort(key=lambda line: min(item['y_top'] for item in line))
    
    # Sort items within each line left-to-right
    for line in lines:
        line.sort(key=lambda item: item['x_left'])
        
    # Merge line representations
    merged_lines = []
    for line in lines:
        line_text = " ".join(item['text'] for item in line)
        line_x_left = min(item['x_left'] for item in line)
        line_x_right = max(item['x_right'] for item in line)
        line_y_top = min(item['y_top'] for item in line)
        line_y_bottom = max(item['y_bottom'] for item in line)
        merged_lines.append({
            'text': line_text,
            'x_left': line_x_left,
            'x_right': line_x_right,
            'y_top': line_y_top,
            'y_bottom': line_y_bottom,
            'height': line_y_bottom - line_y_top,
            'width': line_x_right - line_x_left
        })
        
    # Find paragraph breaks
    widths = [line['width'] for line in merged_lines]
    max_width = max(widths) if widths else 1.0
    
    paragraphs = []
    current_para = []
    last_is_open = False
    
    for i, line in enumerate(merged_lines):
        current_para.append(line['text'])
        
        is_para_end = False
        if i == len(merged_lines) - 1:
            if is_left_page:
                # For the last line of the left page, check if it meets the criteria of paragraph end
                is_para_end = False
                if line['width'] < 0.82 * max_width:
                    text_stripped = line['text'].strip()
                    if text_stripped and text_stripped[-1] in ('.', '?', '!', '"', '”', '`', '’'):
                        is_para_end = True
                    elif line['width'] < 0.70 * max_width:
                        is_para_end = True
                if not is_para_end:
                    last_is_open = True
            else:
                is_para_end = True
        else:
            next_line = merged_lines[i+1]
            gap = next_line['y_top'] - line['y_bottom']
            if gap > 1.45 * line['height']:
                is_para_end = True
            elif line['width'] < 0.82 * max_width:
                text_stripped = line['text'].strip()
                if text_stripped and text_stripped[-1] in ('.', '?', '!', '"', '”', '`', '’'):
                    is_para_end = True
                elif line['width'] < 0.70 * max_width:
                    is_para_end = True
                    
        if is_para_end or (i == len(merged_lines) - 1):
            # Combine items of current paragraph
            para_text = " ".join(current_para)
            
            # Spacing correction on this paragraph
            pp_cfg = config.get('post_process', {})
            if pp_cfg.get('kiwi_spacing', False):
                kiwi = get_kiwi()
                if kiwi:
                    para_text = kiwi.space(para_text, reset_whitespace=pp_cfg.get('reset_whitespace', True))
            
            paragraphs.append(para_text)
            current_para = []
            
    return paragraphs, last_is_open

def run_ocr_pipeline(image_input: Any, config: dict, verbose: bool = False, track_stats: bool = False) -> Tuple[list, Image.Image, str, dict]:
    """
    Executes the complete OCR pipeline.
    
    Args:
        image_input: File path, PIL Image, or Numpy Array.
        config: Configuration dictionary.
        verbose: Print logging info.
        track_stats: Measure CPU/Memory usage.
        
    Returns:
        (results, preprocessed_pil, final_text, stats_dict)
    """
    start_time = time.time()
    mem_start, u_start, s_start = get_resource_usage() if track_stats else (0, 0, 0)
    
    # 1. Image preparation
    image_np, image_pil = prepare_image(image_input, config, verbose=verbose)
    
    # 2. Get OCR Engine & read text
    engine = get_engine(config, verbose=verbose)
    
    ocr_start_time = time.time()
    results = engine.read_text(image_np)
    ocr_end_time = time.time()
    
    mem_end, u_end, s_end = get_resource_usage() if track_stats else (0, 0, 0)
    
    # Check if the OCR engine produced a preprocessed/rotated image (e.g. PaddleOCR's internal preprocessor)
    if hasattr(engine, 'last_preprocessed_image') and engine.last_preprocessed_image is not None:
        if verbose:
            print(" -> OCR Engine returned an internally preprocessed/rotated image. Using it for PIL return.")
        # PaddleOCR's returned image is BGR, so convert to RGB for PIL Image compatibility
        output_np = engine.last_preprocessed_image
        image_pil = Image.fromarray(cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB))
    
    # 3. Post-process text using paragraph reconstruction
    paragraphs, _ = reconstruct_paragraphs(results, config, is_left_page=False, image_height=image_np.shape[0])
    final_text = "\n\n".join(paragraphs)
            
    stats = {}
    if track_stats:
        stats['ocr_time'] = ocr_end_time - ocr_start_time
        stats['total_time'] = time.time() - start_time
        stats['cpu_used'] = (u_end - u_start) + (s_end - s_start)
        stats['peak_memory'] = mem_end
        
    return results, image_pil, final_text, stats
