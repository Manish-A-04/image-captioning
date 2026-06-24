import os
import time
import json
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model import ImageCaptioningModel
from dataloader import get_test_dataloader

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageCaptioningModel(
        encoder_name=config.get('encoder_name', "google/vit-base-patch16-224-in21k"),
        decoder_name=config.get('decoder_name', "gpt2")
    ).to(device)

    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = get_test_dataloader(config)

    max_samples = config.get('max_samples', None)
    beam_size   = config.get('beam_size', 5)
    max_len     = config.get('max_len', 128)

    hypotheses   = []
    references   = []
    rouge        = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []

    done    = 0
    t_start = time.time()

    def _iter_batches():
        for b in test_loader:
            if b is None:
                continue
            yield b

    batch_iter = tqdm(_iter_batches(), desc="Batches", unit="batch") if USE_TQDM else _iter_batches()

    with torch.no_grad():
        for batch in batch_iter:
            images      = batch['image'].to(device)
            gt_captions = batch['caption']

            # Generate captions using Hugging Face generate
            pred_ids = model.generate(images, max_length=max_len, num_beams=beam_size)
            
            # Decode predictions and ground truths
            pred_strs = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            gt_strs   = model.tokenizer.batch_decode(gt_captions, skip_special_tokens=True)

            for i in range(images.size(0)):
                if max_samples is not None and done >= max_samples:
                    break
                
                pred_str = pred_strs[i].strip()
                gt_str   = gt_strs[i].strip()

                pred_tokens = pred_str.split()
                gt_tokens   = gt_str.split()

                hypotheses.append(pred_tokens)
                references.append([gt_tokens])

                score = rouge.score(gt_str, pred_str)
                rouge_scores.append(score['rougeL'].fmeasure)

                done += 1

            if max_samples is not None and done >= max_samples:
                break

    if not hypotheses:
        return {}

    smoothing = SmoothingFunction().method4
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),             smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),         smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),    smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    elapsed_total = time.time() - t_start
    results = {
        'BLEU-1':      round(bleu1, 4),
        'BLEU-2':      round(bleu2, 4),
        'BLEU-3':      round(bleu3, 4),
        'BLEU-4':      round(bleu4, 4),
        'ROUGE-L':     round(avg_rouge, 4),
        'num_samples': done,
        'elapsed_s':   round(elapsed_total, 1),
    }

    with open(config.get('results_path', 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    config = {
        'encoder_name':     'google/vit-base-patch16-224-in21k',
        'decoder_name':     'gpt2',
        'checkpoint_path':  'checkpoints/best_model.pt',
        'results_path':     'eval_results.json',
        'max_len':          128,
        'beam_size':        5,
        'batch_size':       16,
        'max_samples':      None,
        'annotations_path': os.path.join('..', 'data', 'raw', 'tiny-trcap-en.json'),
        'image_dir':        os.path.join('..', 'data', 'raw', 'images'),
        'split_ratios':     (0.70, 0.10, 0.20),
        'num_workers':      3,
        'seed':             42,
    }
    evaluate(config)