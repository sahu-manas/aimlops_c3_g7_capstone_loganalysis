import abc
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

class AbstractLogParser(abc.ABC):
    """
    Abstract base class for log file parsing.
    """
    def __init__(self):
        """
        Initialize the log parser with a specific log file path.

        :param log_file_path: Full path to the log file to be parsed
        """

    def _validate_log_file(self, log_file_path: str) -> None:
        """
        Validate the log file path before parsing.

        :param log_file_path: Path to the log file
        :raises FileNotFoundError: If the log file does not exist
        :raises PermissionError: If the log file cannot be read
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        if not os.access(log_file_path, os.R_OK):
            raise PermissionError(f"Cannot read log file: {log_file_path}")

    @abc.abstractmethod
    def parseFile(self, log_file_path: str) -> Optional[pd.DataFrame]:
        """
        Abstract method to parse the entire log file.
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def _parse_lines(self, lines) -> Optional[pd.DataFrame]:
        """
        Abstract method to parse a single log line.
        
        :param line: A single line from the log file
        :return: A dictionary representing the parsed log entry, or None if parsing fails
        """
        pass

    def filter_entries(self, 
                       start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None, 
                       log_level: Optional[str] = None, 
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Filter log entries based on various criteria.

        :return: Filtered list of log entries
        """
        filtered_entries = self.log_entries.copy()

        if start_time:
            filtered_entries = [entry for entry in filtered_entries 
                                if entry.get('timestamp', datetime.min) >= start_time]
        
        if end_time:
            filtered_entries = [entry for entry in filtered_entries 
                                if entry.get('timestamp', datetime.max) <= end_time]
        
        if log_level:
            filtered_entries = [entry for entry in filtered_entries 
                                if entry.get('log_level') == log_level]
        
        # Additional custom filtering
        for key, value in kwargs.items():
            filtered_entries = [entry for entry in filtered_entries 
                                if entry.get(key) == value]
        
        return filtered_entries
