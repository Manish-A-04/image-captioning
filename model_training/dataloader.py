import json
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageCaptionDataset


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images   = torch.stack([item['image']   for item in batch])
    captions = torch.stack([item['caption'] for item in batch])
    img_ids  = [item['img_id'] for item in batch]

    return {
        'image':   images,
        'caption': captions,
        'img_id':  img_ids,
    }


def _split_annotations(
    annotations,
    split_ratios=(0.70, 0.10, 0.20),
    seed=42,
):
    train_frac, val_frac, test_frac = split_ratios

    image_ids = sorted({ann['image_id'] for ann in annotations})
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n       = len(image_ids)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    n_test  = n - n_train - n_val

    train_ids = set(image_ids[:n_train])
    val_ids   = set(image_ids[n_train : n_train + n_val])
    test_ids  = set(image_ids[n_train + n_val :])

    train_anns = [a for a in annotations if a['image_id'] in train_ids]
    val_anns   = [a for a in annotations if a['image_id'] in val_ids]
    test_anns  = [a for a in annotations if a['image_id'] in test_ids]

    return train_anns, val_anns, test_anns


def get_dataloaders(config):
    with open(config['annotations_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = data['annotations']

    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    split_ratios = config.get('split_ratios', (0.70, 0.10, 0.20))
    seed         = config.get('seed', 42)
    train_anns, val_anns, _ = _split_annotations(annotations, split_ratios, seed)

    max_len      = config.get('max_len', 128)
    caption_mode = config.get('caption_mode', 'random')
    image_dir    = config['image_dir']
    augment      = config.get('augment', True)
    num_workers  = config.get('num_workers', 4)
    batch_size   = config['batch_size']
    pin_memory   = torch.cuda.is_available()

    train_ds = ImageCaptionDataset(
        annotations=train_anns,
        image_dir=image_dir,
        vocab=vocab,
        transform=get_train_transform() if augment else get_eval_transform(),
        max_len=max_len,
        caption_mode=caption_mode,
    )
    val_ds = ImageCaptionDataset(
        annotations=val_anns,
        image_dir=image_dir,
        vocab=vocab,
        transform=get_eval_transform(),
        max_len=max_len,
        caption_mode='first',
    )

    val_num_workers = (num_workers + 1) // 2 if num_workers > 0 else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(val_num_workers > 0),
    )
    return train_loader, val_loader


def get_test_dataloader(config):
    with open(config['annotations_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = data['annotations']

    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    split_ratios = config.get('split_ratios', (0.70, 0.10, 0.20))
    seed         = config.get('seed', 42)
    _, _, test_anns = _split_annotations(annotations, split_ratios, seed)

    max_len     = config.get('max_len', 128)
    image_dir   = config['image_dir']
    num_workers = config.get('num_workers', 4)
    batch_size  = config.get('batch_size', 16)
    pin_memory  = torch.cuda.is_available()

    test_ds = ImageCaptionDataset(
        annotations=test_anns,
        image_dir=image_dir,
        vocab=vocab,
        transform=get_eval_transform(),
        max_len=max_len,
        caption_mode='first',
    )

    test_num_workers = (num_workers + 1) // 2 if num_workers > 0 else 0

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(test_num_workers > 0),
    )
