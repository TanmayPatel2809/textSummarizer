import os
from datasets import load_dataset
from src import logger
from src.entity.config_entity import (DataIngestionconfig)


class DataIngestion:
    def __init__(self, config:DataIngestionconfig):
        self.config=config
    
    def download_dataset(self):
        """
        Downloads the dataset from Hugging Face and saves it to a local zip file.
        """
        if not os.path.exists(self.config.local_data_file):
            dataset = load_dataset(self.config.hf_dataset_name)
            dataset.save_to_disk(self.config.local_data_file)
            logger.info(f"Dataset download! with following info: ")
        else:
            logger.info(f"File already exists.")