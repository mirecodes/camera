import easyocr
from typing import List, Tuple
from .base import OCREngine

class EasyOCREngine(OCREngine):
    """
    OCR Engine implementation using the EasyOCR library.
    Best for varied text environments and natural scenes.
    """

    def __init__(self, config: dict, model_path: str, verbose: bool = False):
        """
        Initializes the EasyOCR reader.
        
        Args:
            config: System configuration dictionary.
            model_path: Local path to store/load EasyOCR models.
            verbose: Enable detailed logging during initialization.
        """
        ocr_cfg = config.get('ocr', {})
        self.reader = easyocr.Reader(
            ocr_cfg.get('languages', ['ko', 'en']),
            gpu=ocr_cfg.get('gpu', False),
            model_storage_directory=model_path,
            verbose=verbose
        )
        self.paragraph = ocr_cfg.get('paragraph', True)

    def read_text(self, image_np) -> List[Tuple[List[List[int]], str, float]]:
        """
        Performs OCR using EasyOCR and returns standardized results.
        """
        return self.reader.readtext(image_np, paragraph=self.paragraph)
