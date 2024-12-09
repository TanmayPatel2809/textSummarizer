from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionconfig:
    root_dir: Path
    hf_dataset_name: str
    local_data_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass
class DataTransformationconfig:
    root_dir: Path
    data_path: Path