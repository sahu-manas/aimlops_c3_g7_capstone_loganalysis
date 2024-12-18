import re
from datetime import datetime
from typing import Dict, Any, Optional
from ..abstractLogParser import AbstractLogParser

class NginxLogParser(AbstractLogParser):
    """
    Parser for Nginx access log files.
    """
    # Typical Nginx log format:
    # 192.168.1.1 - - [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 3874 "http://referer.com" "User-Agent"
    _LOG_PATTERN = re.compile(
        r'(\S+) (\S+) (\S+) \[([^]]+)\] "([^"]*)" (\d+) (\d+) "([^"]*)" "([^"]*)"'
    )

    def parse(self) -> None:
        """
        Parse the entire Nginx log file.
        """
        self.log_entries.clear()
        
        with open(self.log_file_path, 'r') as file:
            for line in file:
                parsed_entry = self._parse_line(line.strip())
                if parsed_entry:
                    self.log_entries.append(parsed_entry)

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single Nginx log line.

        :param line: A single line from the Nginx log file
        :return: Parsed log entry or None if parsing fails
        """
        match = self._LOG_PATTERN.match(line)
        if not match:
            return None

        try:
            # Parse timestamp
            timestamp = datetime.strptime(
                match.group(4), 
                '%d/%b/%Y:%H:%M:%S %z'
            )

            return {
                'ip_address': match.group(1),
                'identity': match.group(2),
                'user': match.group(3),
                'timestamp': timestamp,
                'request': match.group(5),
                'status_code': int(match.group(6)),
                'bytes_sent': int(match.group(7)),
                'referer': match.group(8),
                'user_agent': match.group(9)
            }
        except (ValueError, TypeError):
            return None
