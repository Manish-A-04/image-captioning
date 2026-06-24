import random
import os

from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        annotations,
        image_dir,
        transform=None,
        caption_mode='random',
    ):
        self.image_dir    = image_dir
        self.transform    = transform
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

        return {
            'image':   image,
            'caption_text': caption_text,
            'img_id':  sample['image_id'],
        }

    def __repr__(self):
        return (
            f"ImageCaptionDataset("
            f"samples={len(self)}, "
            f"caption_mode='{self.caption_mode}')"
        )
