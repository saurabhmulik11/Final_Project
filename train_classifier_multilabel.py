import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support

DATA_DIR = 'data'
TRAIN = os.path.join(DATA_DIR, 'train_clf.jsonl')
VAL = os.path.join(DATA_DIR, 'val_clf.jsonl')
OUTPUT_DIR = 'models/roberta_risk_multilabel'
BASE = os.environ.get('BASE_CLS_MODEL', 'roberta-base')


def compute_metrics(pred):
    logits = pred.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = pred.label_ids
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f)}


def main():
    dataset = load_dataset('json', data_files={'train': TRAIN, 'val': VAL})
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    sample = next(iter(dataset['train']))
    num_labels = len(sample['labels'])
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=num_labels, problem_type='multi_label_classification')

    def preprocess(examples):
        toks = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        toks['labels'] = examples['labels']
        return toks

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        num_train_epochs=3,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == '__main__':
    main()
