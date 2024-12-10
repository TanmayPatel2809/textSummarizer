from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.entity.config_entity import ModelTrainerConfig
import nltk
nltk.download('punkt')

from evaluate import load
from transformers import pipeline
import numpy as np

# Load ROUGE metric
rouge_metric = load("rouge")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def compute_metrics(self,eval_pred):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        predictions, labels = eval_pred# Obtaining predictions and true labels
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()} # Extracting some results

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        #loading the data
        dataset = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir = self.config.output_dir,
            evaluation_strategy = self.config.evaluation_strategy,
            metric_for_best_model = self.config.metric_for_best_model,
            learning_rate=float(self.config.learning_rate),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.weight_decay,
            save_strategy= self.config.save_strategy,
            fp16=True
        ) 
        trainer = Trainer(model=model,
            args=trainer_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        torch.cuda.empty_cache()
        trainer.train()

        ## Save model
        model.save_pretrained(os.path.join(self.config.output_dir,"samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.output_dir,"tokenizer"))


