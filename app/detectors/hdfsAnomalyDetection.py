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

import torch

from config.config import config
from abstractAnomalyDetector import AbstractAnomalyDetector
from logbert_pytorch.predict_log import Predictor
from logbert_pytorch.dataset.utils import seed_everything

logging.basicConfig(level=config.app_config.logging_level,
                    format=config.app_config.logging_format)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

class HDFSAnomalyDetector(AbstractAnomalyDetector):
    """
    Anomaly Detector for HDFS log files.
    """
    def __init__(self):
        super().__init__()
        
    def createOptions(self) -> dict:
        options = dict()

        options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        options["output_dir"] = os.path.join(config.app_config.rootPath,config.app_config.hdfs_templates_dir)
        options["model_dir"] = os.path.join(config.app_config.rootPath,config.app_config.hdfs_model_dir)
        options["model_path"] = os.path.join(options["model_dir"],config.app_config.hdfs_model_name)
        options["train_vocab"] = os.path.join(config.app_config.rootPath,config.app_config.train_vocab)
        options["vocab_path"] = os.path.join(config.app_config.rootPath,config.app_config.vocab_path)

        options["window_size"] = config.app_config.window_size
        options["adaptive_window"] = config.app_config.adaptive_window
        options["seq_len"] = config.app_config.seq_len
        options["max_len"] = config.app_config.max_len # for position embedding
        options["min_len"] = config.app_config.min_len
        options["mask_ratio"] = config.app_config.mask_ratio
        # sample ratio
        options["train_ratio"] = config.app_config.train_ratio
        options["valid_ratio"] = 0.1
        options["test_ratio"] = 1

        # features
        options["is_logkey"] = config.app_config.is_logkey
        options["is_time"] = config.app_config.is_time

        options["hypersphere_loss"] = config.app_config.hypersphere_loss
        options["hypersphere_loss_test"] = config.app_config.hypersphere_loss_test

        options["scale"] = config.app_config.scale # MinMaxScaler()
        options["scale_path"] = os.path.join(options["model_dir"],"scale.pkl")

        # model
        options["hidden"] = config.app_config.hidden # embedding size
        options["layers"] = config.app_config.layers
        options["attn_heads"] = config.app_config.attn_heads

        options["epochs"] = config.app_config.epochs
        options["n_epochs_stop"] = config.app_config.n_epochs_stop
        options["batch_size"] = config.app_config.batch_size

        options["corpus_lines"] = config.app_config.corpus_lines
        options["on_memory"] = config.app_config.on_memory
        options["num_workers"] = config.app_config.num_workers
        options["lr"] = config.app_config.lr
        options["adam_beta1"] = config.app_config.adam_beta1
        options["adam_beta2"] = config.app_config.adam_beta2
        options["adam_weight_decay"] = config.app_config.adam_weight_decay
        options["with_cuda"]= config.app_config.with_cuda
        options["cuda_devices"] = config.app_config.cuda_devices
        options["log_freq"] = config.app_config.log_freq

        # predict
        options["num_candidates"] = config.app_config.num_candidates
        options["gaussian_mean"] = config.app_config.gaussian_mean
        options["gaussian_std"] = config.app_config.gaussian_std

        seed_everything(seed=1234)
        
        return options
        
    def computeAnomaly(self, line: str) -> bool:
        """
        Abstract method to compute Anomaly.
        Must be implemented by subclasses.
        """
        options = self.createOptions()
        res = ' '.join([str(s) for s in line])
        logger.info(res)
        total_results = Predictor(options).predictInMemory(res)
        
        if total_results > 0:
            return True
        else:
            return False
        