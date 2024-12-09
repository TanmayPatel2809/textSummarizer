import os
from src import logger
from datasets import load_from_disk
from transformers import BartTokenizer
from src.entity.config_entity import DataTransformationconfig

class DataTransformation:
    def __init__(self, config: DataTransformationconfig):
        self.config = config
        self.tokenizer= BartTokenizer.from_pretrained(self.config.tokenizer_name)
    def data_cleaning(self):
        dataset = load_from_disk(self.config.data_path)
        dataset['train'] = dataset['train'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['test'] = dataset['test'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['validation'] = dataset['validation'].filter(lambda example: example['dialogue'].strip() != "")

        dataset.save_to_disk(self.config.filtered_data_path)

    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )

        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.filtered_data_path))