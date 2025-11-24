
import pandas as pd
import torch
import re
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoConfig
import evaluate  # Hugging Face's evaluate library
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay, hamming_loss
from tqdm import tqdm

# --- 1. Define Constants ---
MODELS_DIR = "./legal-ai/models_final"  # UPDATED PATH
DATA_DIR = "./legal-ai/data"            # UPDATED PATH

# --- Model Paths ---
SUMMARIZER_PATH = os.path.join(MODELS_DIR, "t5_base_summarizer")
RISK_PATH = os.path.join(MODELS_DIR, "distilbert_risk_multilabel")
CLAUSE_PT_PATH = os.path.join(MODELS_DIR, "bert_clause_classifier.pt") # .pt file
BASELINE_SUMMARIZER_PATH = os.path.join(MODELS_DIR, "distilbart-cnn-12-6")
BASELINE_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "legal-bert-base-uncased")

# --- Data Paths ---
TEST_CLF_FILE = os.path.join(DATA_DIR, "test_clf.jsonl")
TEST_SUMMARIZATION_FILE = os.path.join(DATA_DIR, "test_summarization.jsonl")
RISK_LABEL_MAPPING_FILE = os.path.join(DATA_DIR, "label_mapping.json")

# Set device
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")


# --- 2. Data Loading Functions ---

def load_all_data():
    """Loads all test data and labels from the /data directory."""
    
    # --- Load Risk Labels ---
    try:
        with open(RISK_LABEL_MAPPING_FILE, 'r', encoding='utf-8') as f: # Added encoding
            RISK_LABELS = json.load(f)
        print(f"Loaded {len(RISK_LABELS)} risk labels: {RISK_LABELS}")
    except FileNotFoundError:
        print(f"Error: {RISK_LABEL_MAPPING_FILE} not found. Cannot perform risk evaluation.")
        RISK_LABELS = []

    # --- Load Classification Data (test_clf.jsonl) ---
    test_data_clf = []
    CLASSIFICATION_LABELS = []
    try:
        with open(TEST_CLF_FILE, 'r', encoding='utf-8') as f: # Added encoding
            for line in f:
                test_data_clf.append(json.loads(line))
        print(f"Loaded {len(test_data_clf)} classification test samples from {TEST_CLF_FILE}")
        
        # Dynamically get classification labels from the test data
        CLASSIFICATION_LABELS = sorted(list(set([d['clause_label'] for d in test_data_clf])))
        print(f"Found {len(CLASSIFICATION_LABELS)} clause classes: {CLASSIFICATION_LABELS}")

    except FileNotFoundError:
        print(f"Error: {TEST_CLF_FILE} not found. Skipping classification tasks.")
    except KeyError:
        print(f"Error: 'clause_label' not found in {TEST_CLF_FILE}. Check data format.")
        
    # --- Load Summarization Data (test_summarization.jsonl) ---
    test_data_summarization = []
    try:
        with open(TEST_SUMMARIZATION_FILE, 'r', encoding='utf-8') as f: # Added encoding
            for line in f:
                test_data_summarization.append(json.loads(line))
        print(f"Loaded {len(test_data_summarization)} summarization test samples from {TEST_SUMMARIZATION_FILE}")
    except FileNotFoundError:
        print(f"Error: {TEST_SUMMARIZATION_FILE} not found. Skipping summarization task.")
        
    return test_data_summarization, test_data_clf, RISK_LABELS, CLASSIFICATION_LABELS


# --- 3. Model Loading Functions ---

def load_all_models(CLASSIFICATION_LABELS):
    """Loads all custom and baseline models for evaluation."""
    print("\nLoading models...")
    models = {}

    # --- Custom Risk Classifier (Hugging Face format) ---
    try:
        models["custom_risk_classifier"] = pipeline("text-classification", model=RISK_PATH, tokenizer=RISK_PATH, return_all_scores=True, device=device)
        print("Successfully loaded custom risk classifier.")
    except Exception as e:
        print(f"Error loading custom risk classifier: {e}")
        models["custom_risk_classifier"] = None

    # --- Custom Summarizer (Hugging Face format) ---
    try:
        models["custom_summarizer"] = pipeline("summarization", model=SUMMARIZER_PATH, tokenizer=SUMMARIZER_PATH, device=device)
        print(f"Successfully loaded custom summarizer from {SUMMARIZER_PATH}")
    except Exception as e:
        print(f"Error loading custom summarizer: {e}")
        models["custom_summarizer"] = None

    # --- Baseline Classifier (Hugging Face format) ---
    try:
        baseline_classifier_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_CLASSIFIER_PATH)
        baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_CLASSIFIER_PATH)
        
        models["baseline_clause_classifier"] = pipeline("text-classification", model=baseline_classifier_model, tokenizer=baseline_tokenizer, device=device)
        models["baseline_risk_classifier"] = pipeline("text-classification", model=baseline_classifier_model, tokenizer=baseline_tokenizer, return_all_scores=True, device=device)
        print("Successfully loaded baseline classifiers.")
    except Exception as e:
        print(f"Error loading baseline classifiers: {e}")
        models["baseline_clause_classifier"] = None
        models["baseline_risk_classifier"] = None

    # --- Baseline Summarizer (Hugging Face format) ---
    try:
        models["baseline_summarizer"] = pipeline("summarization", model=BASELINE_SUMMARIZER_PATH, tokenizer=BASELINE_SUMMARIZER_PATH, device=device)
        print(f"Successfully loaded baseline summarizer from {BASELINE_SUMMARIZER_PATH}")
    except Exception as e:
        print(f"Error loading baseline summarizer: {e}")
        models["baseline_summarizer"] = None

    # --- (!!!) ACTION REQUIRED: Custom Clause Classifier (.pt file) ---
    print(f"\nLoading custom clause classifier from {CLAUSE_PT_PATH}...")
    try:
        # --- This is no longer a STUB. This is a functional loader. ---
        
        # 1. DEFINE THE BASE MODEL YOU USED FOR TRAINING
        # (e.g., 'bert-base-uncased', 'nlpaueb/legal-bert-base-uncased')
        # (!!!) YOU MUST CHECK THIS (!!!)
        base_model_name = 'nlpaueb/legal-bert-base-uncased' 
        
        # 2. LOAD THE BASE MODEL STRUCTURE
        # We load the config first to tell it the number of labels
        config = AutoConfig.from_pretrained(
            base_model_name, 
            num_labels=len(CLASSIFICATION_LABELS)
        )
        model = AutoModelForSequenceClassification.from_config(config)
        
        # 3. LOAD YOUR SAVED WEIGHTS (.pt file)
        # This assumes you saved the model's `state_dict()`
        model.load_state_dict(torch.load(CLAUSE_PT_PATH, map_location=torch.device(device)))
        
        # 4. LOAD THE TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 5. CREATE THE PIPELINE
        models["custom_clause_classifier"] = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        print("Successfully loaded custom .pt clause classifier.")

    except Exception as e:
        print(f"Error loading custom clause classifier from {CLAUSE_PT_PATH}: {e}")
        print("This often happens if 'base_model_name' is wrong or the .pt file is not a state_dict.")
        models["custom_clause_classifier"] = None
        
    return models

# --- 4. Evaluation Functions ---

def evaluate_summarization(models, test_data):
    """Calculates ROUGE and BLEU scores for summarization models."""
    print("\n--- 1. Evaluating Summarization ---")
    if not test_data:
        print("No summarization data found. Skipping.")
        return

    if not models.get("custom_summarizer") or not models.get("baseline_summarizer"):
        print("Missing summarization models. Skipping.")
        return

    # Prepare data
    texts = [item['text'] for item in test_data]
    references = [item['reference_summary'] for item in test_data]
    
    # Load metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    
    results = {}
    all_predictions = {}
    
    # Run predictions
    for model_name, model in [("Custom (T5)", models["custom_summarizer"]), ("Baseline (DistilBART)", models["baseline_summarizer"])]:
        print(f"Running {model_name} predictions...")
        predictions = []
        for text in tqdm(texts):
            predictions.append(model(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text'])
        all_predictions[model_name] = predictions

    # Calculate metrics
    for model_name, predictions in all_predictions.items():
        print(f"Calculating scores for {model_name}...")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        bleu_scores = bleu.compute(predictions=predictions, references=references, max_order=4)
        
        results[model_name] = {
            "ROUGE-1 (F1)": rouge_scores['rouge1'],
            "ROUGE-2 (F1)": rouge_scores['rouge2'],
            "ROUGE-L (F1)": rouge_scores['rougeL'],
            "BLEU": bleu_scores['bleu']
        }
    
    # --- Display & Plot ---
    df_results = pd.DataFrame(results).T
    print("\nSummarization Results:")
    print(df_results.to_markdown(floatfmt=".4f"))
    
    # Plot
    df_plot = df_results[['ROUGE-1 (F1)', 'ROUGE-2 (F1)', 'ROUGE-L (F1)']].reset_index().melt('index', var_name='Metric', value_name='F1 Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Metric", y="F1 Score", hue="index")
    plt.title("Summarization Model ROUGE Comparison")
    plt.ylim(0, max(df_plot['F1 Score']) * 1.2)
    plt.legend(title="Model")
    plt.savefig("summarization_metrics_comparison.png")
    print("Saved summarization plot to summarization_metrics_comparison.png")
    plt.close()


def evaluate_clause_classification(models, test_data, class_labels):
    """Calculates classification report and plots confusion matrix."""
    print("\n--- 2. Evaluating Clause Classification (Multiclass) ---")
    if not test_data:
        print("No classification data found. Skipping.")
        return

    if not models.get("custom_clause_classifier") or not models.get("baseline_clause_classifier"):
        print("Missing clause classification models. Skipping.")
        return

    true_labels = [item["clause_label"] for item in test_data]
    texts = [item["text"] for item in test_data]
    
    # Get predictions
    print("Running custom model predictions...")
    pred_custom = [models["custom_clause_classifier"](text)[0]['label'] for text in tqdm(texts)]
    
    print("Running baseline model predictions...")
    pred_baseline = [models["baseline_clause_classifier"](text)[0]['label'] for text in tqdm(texts)]

    # Generate reports
    report_custom = classification_report(true_labels, pred_custom, labels=class_labels, output_dict=True, zero_division=0)
    report_baseline = classification_report(true_labels, pred_baseline, labels=class_labels, output_dict=True, zero_division=0)
    
    # Format for table
    results = {
        "Custom (BERT .pt)": {
            "Accuracy": report_custom["accuracy"],
            "Macro Precision": report_custom["macro avg"]["precision"],
            "Macro Recall": report_custom["macro avg"]["recall"],
            "Macro F1-Score": report_custom["macro avg"]["f1-score"]
        },
        "Baseline (Legal-BERT)": {
            "Accuracy": report_baseline["accuracy"],
            "Macro Precision": report_baseline["macro avg"]["precision"],
            "Macro Recall": report_baseline["macro avg"]["recall"],
            "Macro F1-Score": report_baseline["macro avg"]["f1-score"]
        }
    }
    
    df_results_clf = pd.DataFrame(results).T
    print("\nClause Classification Results:")
    print(df_results_clf.to_markdown(floatfmt=".4f"))

    # --- Generate Confusion Matrix Plot ---
    for model_name, preds in [("Custom_BERT_pt", pred_custom), ("Baseline_LegalBERT", pred_baseline)]:
        cm = confusion_matrix(true_labels, preds, labels=class_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix (Clause ID) - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_filename = f"confusion_matrix_clause_{model_name.lower()}.png"
        plt.savefig(plot_filename)
        print(f"Saved confusion matrix to {plot_filename}")
        plt.close()


def evaluate_risk_analysis(models, test_data, risk_labels):
    """Calculates multilabel metrics and plots Precision-Recall curve."""
    print("\n--- 3. Evaluating Risk Analysis (Multilabel) ---")
    if not test_data:
        print("No classification data found. Skipping.")
        return
        
    if not models.get("custom_risk_classifier") or not models.get("baseline_risk_classifier"):
        print("Missing risk analysis models. Skipping.")
        return
        
    if not risk_labels:
        print("RISK_LABELS not found. Skipping risk analysis.")
        return

    # Re-format true_labels from JSON into a binary numpy array
    texts = []
    true_labels_list = []
    try:
        for item in test_data:
            texts.append(item['text'])
            label_dict = item['risk_labels']
            true_labels_list.append([label_dict.get(label, 0) for label in risk_labels])
    except KeyError:
        print(f"Error: 'risk_labels' key not found in {TEST_CLF_FILE}. Check data format.")
        return
        
    true_labels_binary = np.array(true_labels_list)
    
    results_risk = {}
    
    # Create a single figure for P-R curves
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for model_name, model in [("Custom (DistilBERT)", models["custom_risk_classifier"]), ("Baseline (Legal-BERT)", models["baseline_risk_classifier"])]:
        print(f"Running {model_name} predictions...")
        
        raw_preds = model(texts)
        
        # Get the label mapping from this model's config
        model_config = AutoConfig.from_pretrained(model.model.config.name_or_path)
        model_labels = [model_config.id2label[i] for i in range(len(model_config.id2label))]
        
        pred_scores = []
        for res_list in raw_preds:
            score_dict = {res['label']: res['score'] for res in res_list}
            # Order scores according to the master RISK_LABELS list
            ordered_scores = [score_dict.get(label, 0.0) for label in risk_labels]
            pred_scores.append(ordered_scores)
            
        pred_scores_array = np.array(pred_scores)
        pred_scores_binary = (pred_scores_array > 0.5).astype(int)
        
        report = classification_report(true_labels_binary, pred_scores_binary, target_names=risk_labels, output_dict=True, zero_division=0)
        
        results_risk[model_name] = {
            "Micro F1": report["micro avg"]["f1-score"],
            "Macro F1": report["macro avg"]["f1-score"],
            "Hamming Loss (↓)": hamming_loss(true_labels_binary, pred_scores_binary)
        }

        # --- Add to Precision-Recall Curve Plot ---
        # Plot for the first label (e.g., 'High-Risk') as an example
        label_index = 0
        label_name = risk_labels[label_index]
        y_true = true_labels_binary[:, label_index]
        y_scores = pred_scores_array[:, label_index]
        
        PrecisionRecallDisplay.from_predictions(y_true, y_scores, name=f"{model_name} ({label_name})", ax=ax)
    
    # Save the combined P-R plot
    plt.title(f'Precision-Recall Curve for "{risk_labels[0]}"')
    plot_filename = "precision_recall_curve_risk.png"
    plt.savefig(plot_filename)
    print(f"Saved P-R curve plot to {plot_filename}")
    plt.close()

    df_results_risk = pd.DataFrame(results_risk).T
    print("\nRisk Analysis Results:")
    print(df_results_risk.to_markdown(floatfmt=".4f"))


# --- 5. Main Execution ---

def main():
    # Load data
    test_data_summarization, test_data_clf, RISK_LABELS, CLASSIFICATION_LABELS = load_all_data()
    
    # Load models
    models = load_all_models(CLASSIFICATION_LABELS)
    
    # Run evaluations
    evaluate_summarization(models, test_data_summarization)
    evaluate_clause_classification(models, test_data_clf, CLASSIFICATION_LABELS)
    evaluate_risk_analysis(models, test_data_clf, RISK_LABELS)
    
    print("\n--- Evaluation Complete ---")
    print("Plots saved to current directory.")

if __name__ == "__main__":
    main()