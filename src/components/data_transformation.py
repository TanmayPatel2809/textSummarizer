import os
from src import logger
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.entity.config_entity import DataTransformationconfig

class DataTransformation:
    def __init__(self, config: DataTransformationconfig):
        self.config = config
        self.tokenizer_bart = AutoTokenizer.from_pretrained(self.config.tokenizer1_name)
        self.tokenizer_pegasus = AutoTokenizer.from_pretrained(self.config.tokenizer2_name)
    def data_cleaning(self):
        dataset = load_from_disk(self.config.data_path)
        dataset['train'] = dataset['train'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['test'] = dataset['test'].filter(lambda example: example['dialogue'].strip() != "")
        dataset['validation'] = dataset['validation'].filter(lambda example: example['dialogue'].strip() != "")

        dataset.save_to_disk(self.config.filtered_data_path)

    def convert_examples_to_features(self,example_batch,model_type):
        try: 
            if model_type == 'bart':
                input_encodings = self.tokenizer_bart(example_batch['dialogue'], max_length=1024, truncation=True, padding=True)

                with self.tokenizer_pegasus.as_target_tokenizer():
                    target_encodings = self.tokenizer_pegasus(example_batch['summary'], max_length=128, truncation=True, padding=True)

                return {
                    'input_ids': input_encodings['input_ids'],
                    'attention_mask': input_encodings['attention_mask'],
                    'labels': target_encodings['input_ids']
                }

            elif model_type == 'pegasus':
                input_encodings = self.tokenizer_pegasus(example_batch['dialogue'], max_length=1024, truncation=True, padding=True)

                with self.tokenizer_pegasus.as_target_tokenizer():
                    target_encodings = self.tokenizer_pegasus(example_batch['summary'], max_length=128, truncation=True, padding=True)

                return {
                    'input_ids': input_encodings['input_ids'],
                    'attention_mask': input_encodings['attention_mask'],
                    'labels': target_encodings['input_ids']
                }
        except Exception as e:
            print(e)
    
    def save(self, model_type):
        try: 
            dataset_samsum = load_from_disk(self.config.filtered_data_path)
            dataset_samsum_pt = dataset_samsum.map(lambda x: self.convert_examples_to_features(x, model_type=model_type), batched=True)
            output_path = self.config.root_dir + f"/tokenized_{model_type}"
            dataset_samsum_pt.save_to_disk(output_path)
        except Exception as e:
            print(e)