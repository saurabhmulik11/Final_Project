# import os
# from datasets import load_dataset, load_metric
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# DATA_DIR = 'data'
# TRAIN = os.path.join(DATA_DIR, 'train_summarization.jsonl')
# VAL = os.path.join(DATA_DIR, 'val_summarization.jsonl')
# OUTPUT_DIR = 'models/t5_summarizer'
# BASE = os.environ.get('BASE_SUM_MODEL', 't5-small')


# def main():
#     dataset = load_dataset('json', data_files={'train': TRAIN, 'val': VAL})
#     tokenizer = T5Tokenizer.from_pretrained(BASE)
#     model = T5ForConditionalGeneration.from_pretrained(BASE)

#     max_input_length = 1024
#     max_target_length = 150

#     def preprocess(examples):
#         inputs = ['summarize: ' + t for t in examples['text']]
#         model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
#         labels = tokenizer(examples['summary'], max_length=max_target_length, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs

#     tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)
#     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#     training_args = Seq2SeqTrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         predict_with_generate=True,
#         evaluation_strategy='epoch',
#         save_total_limit=2,
#         num_train_epochs=3,
#         logging_steps=200,
#     )

#     rouge = load_metric('rouge')

#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in lab] for lab in labels]
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#         result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
#         return result

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized['train'],
#         eval_dataset=tokenized['val'],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics
#     )

#     trainer.train()
#     trainer.save_model(OUTPUT_DIR)

# if __name__ == '__main__':
#     main()
import os
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# ===========================================
# CONFIGURATION
# ===========================================
BASE_SUM_MODEL = os.getenv("BASE_SUM_MODEL", "t5-base")  # Larger model
DATA_DIR = "data"      # Folder containing *_summarization.jsonl
OUTPUT_DIR = "models/t5_base_summarizer"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 2          # Reduce if you get CUDA OOM
EPOCHS = 3
LEARNING_RATE = 5e-5

# ===========================================
# DEVICE SETUP
# ===========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🧠 Using device: {device}")
print(f"📘 Base Model: {BASE_SUM_MODEL}")

# ===========================================
# LOAD DATASETS
# ===========================================
train_path = os.path.join(DATA_DIR, "train_summarization.jsonl")
val_path = os.path.join(DATA_DIR, "val_summarization.jsonl")

data_files = {
    "train": train_path,
    "validation": val_path
}

dataset = load_dataset("json", data_files=data_files)
print(f"✅ Dataset loaded: {dataset}")

# ===========================================
# TOKENIZER & MODEL
# ===========================================
tokenizer = T5Tokenizer.from_pretrained(BASE_SUM_MODEL)
model = T5ForConditionalGeneration.from_pretrained(BASE_SUM_MODEL).to(device)

# ===========================================
# PREPROCESS FUNCTION
# ===========================================
def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]
    
    model_inputs = tokenizer(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, padding="max_length"
    )
    
    labels = tokenizer(
        targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text", "summary", "id"]
)

print(f"🧾 Tokenization complete! Sample keys: {list(tokenized_datasets['train'].features.keys())}")

# ===========================================
# TRAINER SETUP
# ===========================================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ===========================================
# TRAIN MODEL
# ===========================================
print("\n🚀 Training started with t5-base...\n")
trainer.train()

# ===========================================
# SAVE MODEL
# ===========================================
print("\n✅ Training completed. Saving model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"📂 Model saved to: {OUTPUT_DIR}")
