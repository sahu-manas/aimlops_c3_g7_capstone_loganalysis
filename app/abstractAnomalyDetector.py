import abc
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class AbstractAnomalyDetector(abc.ABC):
    """
    Abstract base class for anomaly detection.
    """
    def __init__(self):
        """
        Initialize the anomaly detector

        
        """
        
    @abc.abstractmethod
    def computeAnomaly(self, line: str) -> bool:
        """
        Abstract method to compute Anomaly.
        Must be implemented by subclasses.
        """
        pass
