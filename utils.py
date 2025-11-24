import json
from typing import Iterable

def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path: str, iterable: Iterable[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in iterable:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
