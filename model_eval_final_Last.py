

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import os
import pdfplumber  # For PDF reading
import io          # To handle file stream

# --- 1. Model Path Configuration ---
# --- ⚠️ IMPORTANT: EDIT THIS LINE ---
#
# Set this variable to the ABSOLUTE path of your "models_final" folder.
# Use forward slashes (/) or double backslashes (\\).
#
# EXAMPLE 1 (Windows):
# MODEL_DIR = r"F:\Final_Year_Project\models_final"
#
# EXAMPLE 2 (Windows, alternative):
# MODEL_DIR = "F:/Final_Year_Project/models_final"
#
# EXAMPLE 3 (Linux/macOS):
# MODEL_DIR = "/home/user/my_project/models_final"

MODEL_DIR = r"F:\Final_Year_Project\legal-ai\models_final"  # <-- EDIT THIS PATH (Used raw string r"..." for safety)

# --- End of update ---


# --- Paths for your CUSTOM models ---
CUSTOM_SUMMARIZER_PATH = os.path.join(MODEL_DIR, "t5_base_summarizer")
CUSTOM_RISK_PATH = os.path.join(MODEL_DIR, "distilbert_risk_multilabel")
CUSTOM_CLAUSE_PATH = os.path.join(MODEL_DIR, "bert_clause_classifier.pt") # .pt file

# --- Paths for your downloaded BASELINE models ---
BASELINE_SUMMARIZER_PATH = os.path.join(MODEL_DIR, "distilbart-cnn-12-6")
BASELINE_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "legal-bert-base-uncased")

st.set_page_config(layout="wide", page_title="Legal Document Analyzer")

# --- 2. PDF Extraction Function ---

def extract_text_from_pdf(file_io):
    """Extracts text from an in-memory PDF file."""
    text = ""
    try:
        with pdfplumber.open(file_io) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        if not text:
            st.error("Could not extract text from PDF. The file might be image-based or corrupted.")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# --- 3. Generic Model Loading Functions ---

@st.cache_resource
def load_summarization_pipeline(model_path):
    """Loads any summarization pipeline from a local path."""
    if not os.path.exists(model_path):
        st.error(f"Summarizer model not found at path: {model_path}")
        return None
    try:
        # --- FIX for t5_base_summarizer ---
        # If the model is t5, we explicitly load the t5-base config first
        # to prevent the state_dict mismatch error.
        if "t5_base_summarizer" in model_path:
            from transformers import T5Config
            config = T5Config.from_pretrained("t5-base")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading summarizer from {model_path}: {e}")
        return None

@st.cache_resource
def load_classification_pipeline(model_path, return_all_scores=False):
    """Loads any classification pipeline from a local path."""
    if not os.path.exists(model_path):
        st.error(f"Classifier model not found at path: {model_path}")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=return_all_scores)
    except Exception as e:
        st.error(f"Error loading classifier from {model_path}: {e}")
        return None

@st.cache_resource
def load_clause_classifier_stub(model_path):
    """
    STUB function to load your .pt file.
    This needs to be replaced with your actual model's class definition.
    """
    if not os.path.exists(BASELINE_CLASSIFIER_PATH):
        st.error(f"Baseline model for stub not found at: {BASELINE_CLASSIFIER_PATH}")
        return None
    if not os.path.exists(model_path):
        st.warning(f"Note: Custom clause model file '{os.path.basename(model_path)}' not found. Using baseline model as a placeholder.")
    
    try:
        # This is a PLACEHOLDER.
        # We load the baseline Legal-BERT model as a stand-in.
        tokenizer = AutoTokenizer.from_pretrained(BASELINE_CLASSIFIER_PATH)
        
        # --- FIX for Classifier Error ---
        # We load the baseline model (which has 2 labels) but tell it to
        # expect 10 labels for your custom task.
        # ignore_mismatched_sizes=True prevents an error by ignoring the
        # final classification layer, which is what you want when loading
        # a .pt file to replace it anyway.
        model = AutoModelForSequenceClassification.from_pretrained(
            BASELINE_CLASSIFIER_PATH, 
            num_labels=10, # Assuming 10 clause types
            ignore_mismatched_sizes=True # <-- THIS IS THE FIX
        )
        
        # --- This is where you would load your custom weights ---
        # if os.path.exists(model_path):
        #     try:
        #         # 1. DEFINE YOUR MODEL CLASS FIRST
        #         #    my_model = YourBertModelClass() 
        #         # 2. LOAD THE STATE DICT
        #         #    my_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #         # 3. SET model = my_model
        #         st.info(f"Successfully loaded custom weights from {model_path}")
        #     except Exception as load_err:
        #         st.warning(f"Could not load custom weights from {model_path}: {load_err}. Using base model structure.")
        # else:
        st.info(f"Loaded STUB clause classifier. Replace this function with your full .pt loading logic.")
        # --------------------------------------------------------

        return pipeline("text-classification", model=model, tokenizer=tokenizer)
        
    except Exception as e:
        st.error(f"Error loading clause classifier stub: {e}")
        return None

# --- 4. Utility Functions ---

def split_into_clauses(text):
    """Simple heuristic to split a legal document into clauses."""
    clauses = re.split(r'\n\s*\n+', text) # Split by one or more empty lines
    return [clause.strip() for clause in clauses if clause.strip()]

def get_risk_labels(results, threshold=0.5):
    """Processes the output of a multi-label classifier."""
    labels = []
    if not results or not results[0]: return "N/A"
    for res in results[0]:
        if res['score'] > threshold:
            labels.append(f"{res['label']} ({res['score']:.2f})")
    return ", ".join(labels) if labels else "No Significant Risk"



# --- 5. Load All Models on Startup ---
st.markdown("""
    <header style="background-color: #1F2937; color: white; padding: 24px; border-radius: 8px; text-align: center;">
        <h1 style="font-size: 2rem; font-weight: bold;">Legal Document Analyzer</h1>
    </header>
""", unsafe_allow_html=True)

st.markdown(f"Attempting to load models from: `{MODEL_DIR}`")

# Validate model folder
if not os.path.exists(MODEL_DIR):
    st.error(f"FATAL ERROR: Model directory not found.\nPlease update the `MODEL_DIR` path in the script.")
    st.stop()

# --- Load Custom Models Only ---
with st.spinner("Loading models, please wait..."):
    custom_summarizer = load_summarization_pipeline(CUSTOM_SUMMARIZER_PATH)
    custom_risk_classifier = load_classification_pipeline(CUSTOM_RISK_PATH, return_all_scores=True)
    custom_clause_classifier = load_clause_classifier_stub(CUSTOM_CLAUSE_PATH)

ALL_MODELS_LOADED = all([custom_summarizer, custom_risk_classifier, custom_clause_classifier])

if not ALL_MODELS_LOADED:
    st.error("One or more models failed to load. Please check file paths and model names.")
    st.stop()

# --- 6. PDF Upload ---
uploaded_file = st.file_uploader("Upload your legal document (PDF)", type=["pdf"])

document_text = ""
if uploaded_file:
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    with st.spinner("Extracting text from PDF..."):
        document_text = extract_text_from_pdf(file_bytes)
    if document_text:
        st.success("PDF text extracted successfully!")

# --- Analyze Button ---
analyze_button = st.button("Analyze Document", type="primary", disabled=not document_text)
st.divider()

# --- 7. Analysis Section ---
if analyze_button:
    clauses = split_into_clauses(document_text)
    if not clauses:
        st.warning("Could not find any clauses. Please check document formatting.")
        st.stop()

    st.header("Legal Document Analysis (Custom Models)")

    # --- 1. Summary ---
    st.subheader(f"1. Document Summary (`{os.path.basename(CUSTOM_SUMMARIZER_PATH)}`)")
    with st.spinner("Generating summary..."):
        try:
            custom_summary = custom_summarizer(document_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.info(custom_summary)
        except Exception as e:
            st.error(f"Error during summarization: {e}")

    # --- 2. Clause Classification ---
    st.subheader(f"2. Clause Classification (`{os.path.basename(CUSTOM_CLAUSE_PATH)}`)")
    with st.spinner("Classifying clauses..."):
        try:
            clause_data = []
            for clause in clauses:
                clause_type = custom_clause_classifier(clause)[0]['label']
                clause_data.append({"Clause": clause, "Identified Type": clause_type})
            st.dataframe(pd.DataFrame(clause_data))
        except Exception as e:
            st.error(f"Error during clause classification: {e}")

    # --- 3. Risk Analysis ---
    st.subheader(f"3. Risk Analysis (`{os.path.basename(CUSTOM_RISK_PATH)}`)")
    with st.spinner("Analyzing risk..."):
        try:
            risk_data = []
            for clause in clauses:
                risk_labels = get_risk_labels(custom_risk_classifier(clause))
                risk_data.append({"Clause": clause, "Risk Profile": risk_labels})
            st.dataframe(pd.DataFrame(risk_data))
        except Exception as e:
            st.error(f"Error during risk analysis: {e}")


# # --- 5. Load All Models on Startup ---
# # st.title("Legal Document Analyzer 📄")
# st.markdown("""
#     <header style="background-color: #1F2937; color: white; padding: 24px; border-radius: 8px; text-align: center;">
#         <h1 style="font-size: 2rem; font-weight: bold;">Legal Document Analyzer</h1>
#     </header>
# """, unsafe_allow_html=True)
# st.markdown(f"Attempting to load models from: `{MODEL_DIR}`") # Show the path for debugging

# # --- NEW VALIDATION CHECK ---
# # Stop the app if the main model directory doesn't exist
# if not os.path.exists(MODEL_DIR):
#     st.error(f"FATAL ERROR: Model directory not found.")
#     st.error(f"The script is looking for your 'models_final' folder at this path: {MODEL_DIR}")
#     st.error("Please update the `MODEL_DIR` variable at the top of the `app.py` script to the correct, absolute path of your 'models_final' folder.")
#     st.stop()
# # --- END NEW CHECK ---

# with st.spinner("Loading all models, please wait..."):
#     # Load custom models
#     custom_summarizer = load_summarization_pipeline(CUSTOM_SUMMARIZER_PATH)
#     custom_risk_classifier = load_classification_pipeline(CUSTOM_RISK_PATH, return_all_scores=True)
#     custom_clause_classifier = load_clause_classifier_stub(CUSTOM_CLAUSE_PATH)
    
#     # Load baseline models
#     baseline_summarizer = load_summarization_pipeline(BASELINE_SUMMARIZER_PATH)
#     baseline_clause_classifier = load_classification_pipeline(BASELINE_CLASSIFIER_PATH)
#     baseline_risk_classifier = load_classification_pipeline(BASELINE_CLASSIFIER_PATH, return_all_scores=True)

# # Check if all models loaded
# ALL_MODELS_LOADED = all([
#     custom_summarizer, custom_risk_classifier, custom_clause_classifier,
#     baseline_summarizer, baseline_clause_classifier, baseline_risk_classifier
# ])

# if ALL_MODELS_LOADED:
#     st.success("All models loaded successfully!")
# else:
#     st.error("One or more models failed to load. Please check file paths and errors above. Verify your 'models_final' folder contains all required model subfolders.")
#     st.stop()


# # # --- 6. UI Layout ---
# # st.header("Document Input")
# # uploaded_file = st.file_uploader("Upload your legal document (PDF)", type=["pdf"])

# # document_text = ""
# # if uploaded_file:
# #     file_bytes = io.BytesIO(uploaded_file.getvalue())
# #     with st.spinner("Extracting text from PDF..."):
# #         document_text = extract_text_from_pdf(file_bytes)
# #     if document_text:
# #         st.success("PDF text extracted successfully!")
# #         with st.expander("Show Extracted Text (First 500 characters)"):
# #             st.text(document_text[:500] + "...")

# # # Analysis Options
# # st.header("Analysis Options")
# # compare_baseline = st.checkbox("Compare with Baseline Models")
# # analyze_button = st.button("Analyze Document", type="primary", disabled=not document_text)

# # st.divider()

# # # --- 7. Analysis & Display ---
# # if analyze_button:
# #     clauses = split_into_clauses(document_text)
# #     if not clauses:
# #         st.warning("Could not find any clauses. Please check document formatting.")
# #         st.stop()

# #     if compare_baseline:
# #         # --- COMPARISON VIEW ---
# #         st.header("Comparative Analysis")
# #         col_custom, col_baseline = st.columns(2)
# #         with col_custom: st.subheader("Your Custom Models")
# #         with col_baseline: st.subheader("Baseline (Built-in) Models")
        
# #         # --- 1. Summarization ---
# #         st.divider()
# #         with col_custom: st.markdown("#### 1. Document Summary")
# #         with col_baseline: st.markdown("#### 1. Document Summary")
# #         with st.spinner("Generating summaries..."):
# #             try:
# #                 custom_summary = custom_summarizer(document_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
# #                 with col_custom: st.info(custom_summary)
# #             except Exception as e:
# #                 with col_custom: st.error(f"Error in custom summarizer: {e}")
            
# #             try:
# #                 baseline_summary = baseline_summarizer(document_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
# #                 with col_baseline: st.info(baseline_summary)
# #             except Exception as e:
# #                 with col_baseline: st.error(f"Error in baseline summarizer: {e}")

# #         # --- 2. Clause Classification ---
# #         st.divider()
# #         with col_custom: st.markdown("#### 2. Clause Classification")
# #         with col_baseline: st.markdown("#### 2. Clause Classification")
# #         with st.spinner("Classifying clauses..."):
# #             custom_clause_data = []
# #             baseline_clause_data = []
# #             try:
# #                 for clause in clauses:
# #                     custom_type = custom_clause_classifier(clause)[0]['label']
# #                     custom_clause_data.append({"Clause": clause, f"Type ({os.path.basename(CUSTOM_CLAUSE_PATH)})": custom_type})
                    
# #                     baseline_type = baseline_clause_classifier(clause)[0]['label']
# #                     baseline_clause_data.append({"Clause": clause, f"Type ({os.path.basename(BASELINE_CLASSIFIER_PATH)})": baseline_type})
                
# #                 with col_custom: st.dataframe(pd.DataFrame(custom_clause_data))
# #                 with col_baseline: st.dataframe(pd.DataFrame(baseline_clause_data))
# #             except Exception as e:
# #                 st.error(f"Error during clause classification: {e}")

# #         # --- 3. Risk Analysis ---
# #         st.divider()
# #         with col_custom: st.markdown("#### 3. Risk Analysis")
# #         with col_baseline: st.markdown("#### 3. Risk Analysis")
# #         with st.spinner("Analyzing risk..."):
# #             custom_risk_data = []
# #             baseline_risk_data = []
# #             try:
# #                 for clause in clauses:
# #                     custom_risk = get_risk_labels(custom_risk_classifier(clause))
# #                     custom_risk_data.append({"Clause": clause, f"Risk ({os.path.basename(CUSTOM_RISK_PATH)})": custom_risk})
                    
# #                     baseline_risk = get_risk_labels(baseline_risk_classifier(clause))
# #                     baseline_risk_data.append({"Clause": clause, f"Risk ({os.path.basename(BASELINE_CLASSIFIER_PATH)})": baseline_risk})

# #                 with col_custom: st.dataframe(pd.DataFrame(custom_risk_data))
# #                 with col_baseline: st.dataframe(pd.DataFrame(baseline_risk_data))
# #             except Exception as e:
# #                 st.error(f"Error during risk analysis: {e}")

# #     else:
# #         # --- CUSTOM MODELS ONLY VIEW ---
# #         st.header("Analysis Results (Custom Models)")
        
# #         # --- 1. Summary ---
# #         st.subheader(f"1. Generated Summary (`{os.path.basename(CUSTOM_SUMMARIZER_PATH)}`)")
# #         with st.spinner("Generating summary..."):
# #             try:
# #                 custom_summary = custom_summarizer(document_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
# #                 st.info(custom_summary)
# #             except Exception as e:
# #                 st.error(f"Error during summarization: {e}")

# #         # --- 2. Clause Classification ---
# #         st.subheader(f"2. Clause Classification (`{os.path.basename(CUSTOM_CLAUSE_PATH)}`)")
# #         with st.spinner("Classifying clauses..."):
# #             try:
# #                 clause_data = []
# #                 for clause in clauses:
# #                     clause_type = custom_clause_classifier(clause)[0]['label']
# #                     clause_data.append({"Clause": clause, "Identified Type": clause_type})
# #                 st.dataframe(pd.DataFrame(clause_data))
# #             except Exception as e:
# #                 st.error(f"Error during clause classification: {e}")

# #         # --- 3. Risk Analysis ---
# #         st.subheader(f"3. Risk Analysis (`{os.path.basename(CUSTOM_RISK_PATH)}`)")
# #         with st.spinner("Analyzing risk..."):
# #             try:
# #                 risk_data = []
# #                 for clause in clauses:
# #                     risk_labels = get_risk_labels(custom_risk_classifier(clause))
# #                     risk_data.append({"Clause": clause, "Risk Profile": risk_labels})
# #                 st.dataframe(pd.DataFrame(risk_data))
# #             except Exception as e:
# #                 st.error(f"Error during risk analysis: {e}")


# # --- 6. UI Layout ---
# st.header("Document Input")
# uploaded_file = st.file_uploader("Upload your legal document (PDF)", type=["pdf"])

# document_text = ""
# if uploaded_file:
#     file_bytes = io.BytesIO(uploaded_file.getvalue())
#     with st.spinner("Extracting text from PDF..."):
#         document_text = extract_text_from_pdf(file_bytes)
#     if document_text:
#         st.success("PDF text extracted successfully!")

# # Analyze Button
# analyze_button = st.button("Analyze Document", type="primary", disabled=not document_text)
# st.divider()

# # --- 7. Analysis & Display ---
# if analyze_button:
#     clauses = split_into_clauses(document_text)
#     if not clauses:
#         st.warning("Could not find any clauses. Please check document formatting.")
#         st.stop()

#     # --- CUSTOM MODELS ONLY VIEW ---
#     st.header("Analysis Results (Custom Models)")

#     # --- 1. Summary ---
#     st.subheader(f"1. Document Summary (`{os.path.basename(CUSTOM_SUMMARIZER_PATH)}`)")
#     with st.spinner("Generating summary..."):
#         try:
#             custom_summary = custom_summarizer(
#                 document_text, max_length=150, min_length=30, do_sample=False
#             )[0]['summary_text']
#             st.info(custom_summary)
#         except Exception as e:
#             st.error(f"Error during summarization: {e}")

#     # --- 2. Clause Classification ---
#     st.subheader(f"2. Clause Classification (`{os.path.basename(CUSTOM_CLAUSE_PATH)}`)")
#     with st.spinner("Classifying clauses..."):
#         try:
#             clause_data = []
#             for clause in clauses:
#                 clause_type = custom_clause_classifier(clause)[0]['label']
#                 clause_data.append({"Clause": clause, "Identified Type": clause_type})
#             st.dataframe(pd.DataFrame(clause_data))
#         except Exception as e:
#             st.error(f"Error during clause classification: {e}")

#     # --- 3. Risk Analysis ---
#     st.subheader(f"3. Risk Analysis (`{os.path.basename(CUSTOM_RISK_PATH)}`)")
#     with st.spinner("Analyzing risk..."):
#         try:
#             risk_data = []
#             for clause in clauses:
#                 risk_labels = get_risk_labels(custom_risk_classifier(clause))
#                 risk_data.append({"Clause": clause, "Risk Profile": risk_labels})
#             st.dataframe(pd.DataFrame(risk_data))
#         except Exception as e:
#             st.error(f"Error during risk analysis: {e}")
