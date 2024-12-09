from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.entity.config_entity import ModelTrainerConfig
from src.components.compute_metrics import compute_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

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
            compute_metrics=compute_metrics,
        )
        
        trainer.train()

        ## Save model
        model.save_pretrained(os.path.join(self.config.output_dir,"samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.output_dir,"tokenizer"))


