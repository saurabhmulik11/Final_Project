# Legal AI - Summarization & Multi-label Risk Detection 

This repository is a runnable starter for a Legal AI pipeline that:
- extracts text from PDFs,
- splits clauses,
- generates weak labels (heuristic) for multi-label risk classification,
- trains a T5 summarizer (PyTorch/Hugging Face) and a RoBERTa multi-label classifier,
- serves inference via FastAPI (PDF upload → summary + clause-level risk detection).

**What I created for you:**
- data prep scripts that take a folder of PDFs (e.g. `dataset/2010/*.pdf` ... `dataset/2025/*.pdf`) and extract text
- automatic split into train/val/test (default 80/10/10)
- heuristic weak-labeler producing multi-label vectors per document (customize keywords)
- training scripts (summarizer + multi-label classifier)
- inference API (`app.py`) that returns summary, top risks, and clause-level detections
- Dockerfile and requirements.txt

## Quick steps (local)

1. Put your PDFs under `dataset/` (you said you have `dataset\2010\*.pdf` through 2025). The script expects `dataset/` root.
2. Create and activate virtualenv, then install requirements:
```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. Prepare dataset (extract text, weak-label, split):
```bash
python prepare_dataset.py --src dataset --out data --weak-label
```
This creates `data/train_summarization.jsonl`, `data/val_summarization.jsonl`, and `data/train_clf.jsonl`, `data/val_clf.jsonl` etc.
- Summarization entries will have extracted `text`. If you have reference summaries, merge them into the JSONL `summary` field.
- Weak labels are created heuristically based on keywords (customize `LABEL_KEYWORDS` in `prepare_dataset.py`).

4. Train models:
```bash
python train_summarizer.py
python train_classifier_multilabel.py
```
5. Run API (point env vars to saved model dirs):
```bash
export SUM_MODEL_DIR=models/t5_summarizer
export CLS_MODEL_DIR=models/roberta_risk_multilabel
python app.py
```
6. Test endpoint:
```bash
curl -F "pdf_file=@examples/sample.pdf" http://localhost:8000/analyze
```

## Notes
- The heuristic weak-labeler is a starting point. You should replace it with human labels for production.
- For long docs, training uses chunking or LongT5/LED models — see comments in the training scripts.
- Use GPU for training and inference if available.

