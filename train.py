import argparse
from transformers import AutoTokenizer
import torch
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

import evaluate
rouge = evaluate.load("rouge")

if torch.cuda.is_available():
    device = torch.device('cuda:0')  
    torch.cuda.set_device(device)  
else:
    device = torch.device('cpu')

from huggingface_hub import notebook_login
notebook_login()

class PegasusTrainer:
    def __init__(self, dataset_path, model_name, datafile):
        self.dataset_path = dataset_path
        self.datafile = datafile
        self.model_name = model_name

    def train(self):
        prefix = "summarize: "

        # Load dataset
        data_files = {"train": self.datafile}
        dataset = load_dataset(self.dataset_path, data_files=data_files)
        checkpoint = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
        # Preprocessing function
        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["content"]]
            model_inputs = tokenizer(inputs, max_length=8192, truncation=True)

            labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        import numpy as np



        # Split dataset
        dataset_train = dataset['train'].train_test_split(test_size=0.1)
        dataset_train = dataset_train.map(preprocess_function, batched=True)



        # Load tokenizer and model

       

        training_args = Seq2SeqTrainingArguments(
            output_dir="summarization_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=8,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train["train"],
            eval_dataset=dataset_train["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pegasus model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file', required=True)
    parser.add_argument('--datafile', type=str, help='Path to the dataset file', required=True)
    parser.add_argument('--model', type=str, help='Name of the pre-trained Pegasus model', required=True)

    args = parser.parse_args()

    pegasus_trainer = PegasusTrainer(args.dataset, args.model, args.datafile)
    pegasus_trainer.train()
