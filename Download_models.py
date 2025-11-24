from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import os

# --- List of models to download ---
# (I identified these from your log file)
models_to_download = {
    "summarizer": "sshleifer/distilbart-cnn-12-6",
    "legal_bert": "nlpaueb/legal-bert-base-uncased"
}

# --- Directory to save models ---
SAVE_DIRECTORY = "models_final"

if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

# --- Download loop ---
for model_type, model_name in models_to_download.items():
    print(f"--- Downloading {model_type}: {model_name} ---")
    
    # Define the save path
    save_path = os.path.join(SAVE_DIRECTORY, model_name.split('/')[-1]) # e.g., "models/distilbart-cnn-12-6"
    
    # Set the correct AutoModel class
    if model_type == "summarizer":
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForSequenceClassification

    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Download model
        model = model_class.from_pretrained(model_name)
        
        # Save them to the local directory
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"Successfully saved {model_name} to {save_path}\n")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}\n")

print("--- Model download complete. ---")