from abc import ABC, abstractmethod
from typing import List, Tuple

class OCREngine(ABC):
    """
    Abstract Base Class for OCR Engines.
    Provides a standardized interface for different OCR implementations.
    """

    @abstractmethod
    def read_text(self, image_np) -> List[Tuple[List[List[int]], str, float]]:
        """
        Extracts text from the given image.
        
        Args:
            image_np: Image data as a numpy array (RGB).
            
        Returns:
            A list of tuples containing (bounding_box, text, confidence).
            bounding_box: List of [x, y] coordinates for the four corners.
            text: Extracted string.
            confidence: Recognition confidence score (0.0 to 1.0).
        """
        pass
