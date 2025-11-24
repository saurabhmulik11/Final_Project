import argparse
import os
from pathlib import Path
import random
from pdf_to_text import extract_text_from_pdf
from clause_splitter import split_into_clauses, clause_has_keyword
from utils import write_jsonl
import json

# Label keywords for weak labeling
LABEL_KEYWORDS = {
    'liability': ['liabilit', 'indemnif', 'hold harmless', 'damages'],
    'termination': ['terminate', 'termination', 'expire', 'expiry'],
    'payment': ['payment', 'fee', 'invoice', 'due', 'payable'],
    'confidentiality': ['confidential', 'non-disclos', 'nda', 'proprietary'],
}

LABEL_ORDER = list(LABEL_KEYWORDS.keys())


def weak_label_text(text):
    """Return multi-label binary vector for given text using keyword heuristics."""
    lower = text.lower()
    labels = [0] * len(LABEL_ORDER)
    for i, lab in enumerate(LABEL_ORDER):
        for kw in LABEL_KEYWORDS[lab]:
            if kw in lower:
                labels[i] = 1
                break
    return labels


def prepare(src_dir, out_dir, weak_label=False, split=(0.8, 0.1, 0.1), seed=42):
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find all PDFs recursively in subfolders
    pdfs = list(src.rglob('*.pdf'))
    print(f"Total PDFs found: {len(pdfs)}")
    for p in pdfs[:5]:  # show first 5 PDFs as sample
        print(f"Sample PDF: {p}")

    if len(pdfs) == 0:
        print("No PDFs found! Check the --src path.")
        return

    random.Random(seed).shuffle(pdfs)
    n = len(pdfs)
    n_train = int(split[0]*n)
    n_val = int(split[1]*n)
    train_files = pdfs[:n_train]
    val_files = pdfs[n_train:n_train+n_val]
    test_files = pdfs[n_train+n_val:]

    def process_list(lst, split_name):
        print(f"\nStarting processing {split_name} PDFs... ({len(lst)} files)")
        out_list = []
        for i, p in enumerate(lst, 1):
            try:
                text = extract_text_from_pdf(str(p))
            except Exception as e:
                print(f"Failed to extract {p}: {e}")
                continue
            item = {'id': p.stem, 'text': text, 'summary': ''}
            if weak_label:
                item['labels'] = weak_label_text(text)
            out_list.append(item)

            if i % 10 == 0 or i == len(lst):
                print(f"Processed {i}/{len(lst)} {split_name} PDFs")

        # Write JSONL files
        write_jsonl(out / f"{split_name}_summarization.jsonl",
                    [{'id': it['id'], 'text': it['text'], 'summary': it['summary']} for it in out_list])
        if weak_label:
            write_jsonl(out / f"{split_name}_clf.jsonl",
                        [{'id': it['id'], 'text': it['text'], 'labels': it['labels']} for it in out_list])

    # Process train, val, test splits
    process_list(train_files, 'train')
    process_list(val_files, 'val')
    process_list(test_files, 'test')

    # Write label mapping
    with open(out / 'label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': LABEL_ORDER}, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Created JSONL files in {out}")


if __name__ == '__main__':
    print("Script started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source dataset root containing PDFs')
    parser.add_argument('--out', required=True, help='Output data folder')
    parser.add_argument('--weak-label', action='store_true', help='Generate weak labels using heuristics')
    args = parser.parse_args()
    prepare(args.src, args.out, weak_label=args.weak_label)
