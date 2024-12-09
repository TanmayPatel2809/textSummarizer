import os
from src import logger
from datasets import load_from_disk
from src.entity.config_entity import DataTransformationconfig

class DataTransformation:
    def __init__(self, config: DataTransformationconfig):
        self.config = config

    def data_cleaning(self):
        dataset = load_from_disk(self.config.data_path)
        dataset['train'] = dataset['train'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['test'] = dataset['test'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['validation'] = dataset['validation'].filter(lambda example: example['dialogue'].strip() != "")

        dataset.save_to_disk(self.config.data_path)
