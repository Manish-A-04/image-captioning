import os
import time
import json
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model import ImageCaptioningModel
from inference import generate_caption
from dataloader import get_test_dataloader

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def decode_ids(token_ids, idx2word, special_tokens={'<PAD>', '<SOS>', '<EOS>'}):
    tokens = []
    for tid in token_ids:
        word = idx2word.get(str(tid), '<UNK>')
        if word == '<EOS>':
            break
        if word not in special_tokens:
            tokens.append(word)
    return tokens


def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config['vocab_path'], 'r') as f:
        vocab = json.load(f)

    idx2word = {str(v): k for k, v in vocab.items()}

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 3),
        ff_dim=config.get('ff_dim', 512),
        max_len=config.get('max_len', 128),
        dropout=0.0
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

            for i in range(images.size(0)):
                if max_samples is not None and done >= max_samples:
                    break

                image    = images[i].unsqueeze(0)
                pred_ids = generate_caption(
                    model, image, vocab, idx2word, device,
                    max_len=max_len,
                    beam_size=beam_size,
                )
                pred_tokens = decode_ids(pred_ids, idx2word)
                gt_ids      = gt_captions[i].tolist()
                gt_tokens   = decode_ids(gt_ids, idx2word)

                hypotheses.append(pred_tokens)
                references.append([gt_tokens])

                pred_str = ' '.join(pred_tokens)
                gt_str   = ' '.join(gt_tokens)
                score    = rouge.score(gt_str, pred_str)
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
        'vocab_path':       'vocab.json',
        'checkpoint_path':  'checkpoints/best_model.pt',
        'results_path':     'eval_results.json',
        'embed_dim':        256,
        'num_heads':        8,
        'num_layers':       3,
        'ff_dim':           512,
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