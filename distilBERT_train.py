import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score

# Load CSV
df = pd.read_csv("track-a.csv")

# Convert to Hugging Face Dataset format
labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
df['labels'] = df[labels].values.tolist()
dataset = Dataset.from_pandas(df[['text', 'labels']])


train_test = dataset.train_test_split(test_size=0.2)
train_ds = train_test['train']
val_ds = train_test['test']

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels),
    problem_type="multi_label_classification"
)

# Trainer args
training_args = TrainingArguments(
    output_dir="./distilbert-emotions",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=4,
    learning_rate=2e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Metrics



def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5
    labels = p.label_ids

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save final model
model.save_pretrained("saved_distilbert_model")
tokenizer.save_pretrained("saved_distilbert_model")
