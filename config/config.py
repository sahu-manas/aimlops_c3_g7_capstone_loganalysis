# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

from typing import ClassVar
import logging

import logbert_pytorch
# Project Directories
PACKAGE_ROOT = Path(logbert_pytorch.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    logging_level: str
    logging_format: str
    hdfs_templates_dir: str
    hdfs_templates_json_file: str
    hdfs_templates_file: str
    hdfs_model_dir: str
    hdfs_model_name: str
    hdfs_center: str
    hdfs_total_dist: str
    hdfs_st: float
    hdfs_depth: int
    hdfs_log_pattern: str
    device: str
    output_dir: str
    model_dir: str
    model_path: str
    train_vocab: str
    vocab_path: str
    window_size: int
    adaptive_window: bool
    seq_len: int
    max_len: int
    min_len: int
    mask_ratio: float
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    is_logkey: bool
    is_time: bool
    hypersphere_loss: bool
    hypersphere_loss_test: bool
    scale: str
    scale_path: str
    hidden: int
    layers: int
    attn_heads: int
    epochs: int
    n_epochs_stop: int
    batch_size: int
    corpus_lines: str
    on_memory: bool
    num_workers: int
    lr: float
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: int
    with_cuda: bool
    cuda_devices: str
    log_freq: str
    num_candidates: int
    gaussian_mean: int
    gaussian_std: int
    
    rootPath: ClassVar[str]
    rootPath = ROOT


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        #model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()