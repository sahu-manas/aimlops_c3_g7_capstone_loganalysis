import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import pandas as pd
import logging
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional
from config.config import config
from abstractLogParser import AbstractLogParser
from logparser import Drain

logging.basicConfig(level=config.app_config.logging_level,
                    format=config.app_config.logging_format)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

class HDFSLogParser(AbstractLogParser):
    """
    Parser for Nginx access log files.
    """
    # Typical Nginx log format:
    # 192.168.1.1 - - [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 3874 "http://referer.com" "User-Agent"
    def __init__(self):
        super().__init__()
        
        self._LOG_PATTERN = config.app_config.hdfs_log_pattern
        self.regex = [
                 r"(?<=blk_)[-\d]+", # block_id
                 r'\d+\.\d+\.\d+\.\d+',  # IP
                 r"(/[-\w]+)+",  # file path
                 #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
                ]
        
        self.templates_dir  = os.path.join(config.app_config.rootPath,config.app_config.hdfs_templates_dir)
        self.log_templates_file = os.path.join(self.templates_dir,config.app_config.hdfs_templates_file)
        self.log_templates_json_file = os.path.join(self.templates_dir,config.app_config.hdfs_templates_json_file)
    
    def hdfs_sampling(self, df, window='session'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."

        with open(self.log_templates_json_file, "r") as f:
            event_num = json.load(f)
        
        df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

        data_dict = defaultdict(list) #preserve insertion order of items
        for idx, row in tqdm(df.iterrows()):
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                data_dict[blk_Id].append(row["EventId"])

        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    
        logger.info("hdfs sampling done")

        return data_df
    
    def generate_data(self, data_df):
        seq = data_df["EventSequence"]
        logger.info("generate data done")
        #print(seq.head())
        return seq
    
    def parseFile(self, log_file_path: str) -> Optional[pd.DataFrame]:
        """
        Parse the entire Nginx log file.
        """
        try:
            self._validate_log_file(log_file_path)
        except:
            raise
        
        log_messages = []
        linecount = 0
        cnt = 0
        
        try:
        
            with open(log_file_path, 'r') as file:
                for line in file.readlines():
                    cnt += 1
                    log_messages.append(line)
                    linecount += 1
        
            log_entries = self._parse_lines(log_messages)
        
            logger.info("parsed the log file....")
            logger.info(log_entries.head())
            return log_entries
        
        except Exception as e:
            raise

    def _parse_lines(self, lines: list) -> Optional[pd.DataFrame]:
        """
        Parse a sequence of HDFS log line for a single block.

        :param line: A list of sequence of line from the HDFS log file
        :return: Parsed log entry as a dataframe or None if parsing fails
        """
        
        st = 0.5  # Similarity threshold
        depth = 5  # Depth of all leaf nodes

        try:
            # Parse timestamp
            parser = Drain.LogParser(self._LOG_PATTERN, depth=depth, st=st, rex=self.regex, keep_para=False)
            df_log = parser.parseInMemory(lines)
            logger.info("Parsed the log lines.....")
            logger.info(df_log.head())
            
            data_df = self.hdfs_sampling(df_log)
            parsed_entry = self.generate_data(data_df)
            logger.info("Created the sequence log keys....")
            logger.info(parsed_entry.head())
            logger.info("Parsing log lines completed....")
            
            return parsed_entry
        except Exception as e:
            logger.error(str(e))
            logger.error("Regular Expression Failure. Unable to match regular expression to log format")
            raise
