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

def draw_and_save(image_np: np.ndarray, results: list, output_path: str, verbose: bool = False):
    """Generates a visual debug image with bounding boxes and detection indices."""
    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for i, res in enumerate(results, 1):
        if len(res) < 2 or res[0] is None or len(res[0]) < 1: continue
        bbox, text = res[0], res[1]
        
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_cv, [pts], True, (0, 255, 0), 2)
        
        try:
            cv2.putText(img_cv, str(i), (int(bbox[0][0]), int(bbox[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except: pass
            
    cv2.imwrite(output_path, img_cv)
    if verbose:
        print(f" -> Result visualization saved to: {output_path}")

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
    
    # 3. Post-process text
    pp_cfg = config.get('post_process', {})
    full_text = " ".join([res[1] for res in results if len(res) >= 2])
    
    final_text = full_text
    if pp_cfg.get('kiwi_spacing', False):
        kiwi = get_kiwi()
        if kiwi:
            final_text = kiwi.space(full_text, reset_whitespace=pp_cfg.get('reset_whitespace', True))
            
    stats = {}
    if track_stats:
        stats['ocr_time'] = ocr_end_time - ocr_start_time
        stats['total_time'] = time.time() - start_time
        stats['cpu_used'] = (u_end - u_start) + (s_end - s_start)
        stats['peak_memory'] = mem_end
        
    return results, image_pil, final_text, stats
