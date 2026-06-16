import os
from paddleocr import PaddleOCR
from typing import List, Tuple
from .base import OCREngine

class PaddleOCREngine(OCREngine):
    """
    OCR Engine implementation using the PaddleOCR (v3.5.0/PaddleX) library.
    Highly optimized for document and book scanning.
    """

    def __init__(self, config: dict, model_path: str, verbose: bool = False):
        """
        Initializes the PaddleOCR predictor.
        
        Args:
            config: System configuration dictionary.
            model_path: Local path used for environment variable redirection.
            verbose: Enable detailed logging during initialization.
        """
        ocr_cfg = config.get('ocr', {})
        # Map languages to PaddleOCR expected format
        lang = 'korean' if 'ko' in ocr_cfg.get('languages', []) else 'en'
        
        # Note: Model path is handled via HOME environment variable set in pipeline.py/ocr.py
        self.ocr = PaddleOCR(
            use_textline_orientation=True, 
            lang=lang,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True
        )
        self.last_preprocessed_image = None

    def read_text(self, image_np) -> List[Tuple[List[List[int]], str, float]]:
        """
        Performs OCR using PaddleOCR and parses the dictionary-based output into 
        a standardized (bbox, text, confidence) format.
        """
        raw_results = self.ocr.ocr(image_np)
        
        self.last_preprocessed_image = None
        results = []
        if not raw_results:
            return results

        # Iterate through the dictionary-based result for each image
        for res in raw_results:
            # Retrieve the preprocessed image from PaddleOCR's internal pipeline
            doc_prep = res.get('doc_preprocessor_res', {})
            if doc_prep and 'output_img' in doc_prep and doc_prep['output_img'] is not None:
                self.last_preprocessed_image = doc_prep['output_img']
            elif doc_prep and 'rot_img' in doc_prep and doc_prep['rot_img'] is not None:
                self.last_preprocessed_image = doc_prep['rot_img']

            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            polys = res.get('dt_polys', [])
            
            # Map PaddleX outputs to the standard OCR internal format
            for text, score, poly in zip(texts, scores, polys):
                results.append((poly, text, float(score)))
                
        return results
