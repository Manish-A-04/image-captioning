import random
import os
import re

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    return text.split()


def encode_caption(
    tokens,
    vocab,
    max_len,
):
    sos_idx = vocab['<SOS>']
    eos_idx = vocab['<EOS>']
    pad_idx = vocab['<PAD>']
    unk_idx = vocab['<UNK>']

    body = [vocab.get(tok, unk_idx) for tok in tokens[: max_len - 2]]
    ids  = [sos_idx] + body + [eos_idx]

    ids += [pad_idx] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


class ImageCaptionDataset(Dataset):

    def __init__(
        self,
        annotations,
        image_dir,
        vocab,
        transform=None,
        max_len=128,
        caption_mode='random',
    ):
        self.image_dir    = image_dir
        self.vocab        = vocab
        self.transform    = transform
        self.max_len      = max_len
        self.caption_mode = caption_mode

        if caption_mode not in ('random', 'first', 'all'):
            raise ValueError(
                f"caption_mode must be 'random', 'first', or 'all'; got '{caption_mode}'"
            )

        grouped = {}
        for ann in annotations:
            grouped.setdefault(ann['image_id'], []).append(ann)

        self.samples = []

        if caption_mode == 'all':
            for img_id, anns in grouped.items():
                for ann in anns:
                    self.samples.append({
                        'image_id': img_id,
                        'captions': [ann['caption']],
                        'filename': ann['filename'],
                    })
        else:
            for img_id, anns in grouped.items():
                self.samples.append({
                    'image_id': img_id,
                    'captions': [a['caption'] for a in anns],
                    'filename': anns[0]['filename'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample['filename'])

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return None
        except UnidentifiedImageError:
            return None
        except Exception as exc:
            return None

        if self.transform is not None:
            image = self.transform(image)

        captions = sample['captions']
        if self.caption_mode == 'random':
            caption_text = random.choice(captions)
        else:
            caption_text = captions[0]

        tokens         = tokenize(caption_text)
        caption_tensor = encode_caption(tokens, self.vocab, self.max_len)

        return {
            'image':   image,
            'caption': caption_tensor,
            'img_id':  sample['image_id'],
        }

    def __repr__(self):
        return (
            f"ImageCaptionDataset("
            f"samples={len(self)}, "
            f"caption_mode='{self.caption_mode}', "
            f"max_len={self.max_len})"
        )
