import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd

from app.parsers.hdfsLogParser import HDFSLogParser
from app.detectors.hdfsAnomalyDetection import HDFSAnomalyDetector

def main():
    # Path to your Apache log file
    #log_file_path = os.path.join(root,'hdfs_analysis/data/HDFS_Anomaly_1.txt')
    log_file_path = os.path.join(root,'hdfs_analysis/data/HDFS_Normal_1.txt')
    
    # Create parser instance
    parser = HDFSLogParser()
    
    # Parse the log file
    log_entries = parser.parseFile(log_file_path)
    
    #log_entries = parser.log_entries
    # Get basic statistics
    print("Total log entries:", log_entries.head())
    
    print("Predicting....")
    detector = HDFSAnomalyDetector()
    for line in log_entries:
        print(line)
        is_anomaly = detector.computeAnomaly(line)
        print(is_anomaly)
    
if __name__ == '__main__':
    main()
