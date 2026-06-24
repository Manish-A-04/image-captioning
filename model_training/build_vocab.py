import json
import os
from collections import Counter
import re

ANNOTATIONS_PATH = os.path.join('..', 'data', 'raw', 'tiny-trcap-en.json')
VOCAB_SAVE_PATH  = 'vocab.json'

MIN_FREQ = 1

SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    return text.split()


def build_vocab(
    annotations_path=ANNOTATIONS_PATH,
    vocab_save_path=VOCAB_SAVE_PATH,
    min_freq=MIN_FREQ,
):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data['annotations']

    counter = Counter()
    for ann in annotations:
        tokens = tokenize(ann['caption'])
        counter.update(tokens)

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

    for word, freq in sorted(counter.items()):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    with open(vocab_save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    return vocab


if __name__ == '__main__':
    build_vocab(
        annotations_path=ANNOTATIONS_PATH,
        vocab_save_path=VOCAB_SAVE_PATH,
        min_freq=MIN_FREQ,
    )
