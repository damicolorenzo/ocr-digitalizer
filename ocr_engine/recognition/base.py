
from abc import ABC, abstractmethod

import numpy as np


class OCRRecognizer(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """Recognize text from an image region."""
        raise NotImplementedError
