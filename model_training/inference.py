import sys
import json
import torch
from PIL import Image
from torchvision import transforms

from model import ImageCaptioningModel


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def greedy_decode(model, image_features, vocab, idx2word, device, max_len=128):
    sos_idx = vocab['<SOS>']
    eos_idx = vocab['<EOS>']

    tokens = [sos_idx]
    for _ in range(max_len):
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model.decode_step(image_features, input_tensor)
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)
        if next_token == eos_idx:
            break
    return tokens[1:]


def beam_search_decode(model, image_features, vocab, idx2word, device, max_len=128, beam_size=5):
    sos_idx = vocab['<SOS>']
    eos_idx = vocab['<EOS>']

    beams = [([sos_idx], 0.0)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == eos_idx:
                completed.append((seq, score))
                continue
            input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model.decode_step(image_features, input_tensor)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            top_log_probs, top_indices = log_probs.topk(beam_size)
            for log_prob, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_beams.append((seq + [idx], score + log_prob))

        new_beams.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = new_beams[:beam_size]

        if all(seq[-1] == eos_idx for seq, _ in beams):
            completed.extend(beams)
            break

    if not completed:
        completed = beams

    completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
    best_seq = completed[0][0]

    result = []
    for tid in best_seq[1:]:
        if tid == eos_idx:
            break
        result.append(tid)
    return result


def generate_caption(model, image_tensor, vocab, idx2word, device, max_len=128, beam_size=5):
    model.eval()
    with torch.no_grad():
        image_features = model.encode(image_tensor)
    if beam_size > 1:
        token_ids = beam_search_decode(model, image_features, vocab, idx2word, device, max_len, beam_size)
    else:
        token_ids = greedy_decode(model, image_features, vocab, idx2word, device, max_len)
    return token_ids


def caption_image_file(image_path, checkpoint_path, vocab_path, beam_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    idx2word = {str(v): k for k, v in vocab.items()}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('config', {})

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=cfg.get('embed_dim', 256),
        num_heads=cfg.get('num_heads', 8),
        num_layers=cfg.get('num_layers', 3),
        ff_dim=cfg.get('ff_dim', 512),
        max_len=cfg.get('max_len', 128),
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    token_ids = generate_caption(model, image_tensor, vocab, idx2word, device, beam_size=beam_size)

    special = {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}
    caption = ' '.join(idx2word.get(str(tid), '<UNK>') for tid in token_ids if idx2word.get(str(tid), '') not in special)
    return caption


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path] [vocab_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/best_model.pt'
    vocab_path = sys.argv[3] if len(sys.argv) > 3 else 'vocab.json'

    caption = caption_image_file(image_path, checkpoint_path, vocab_path)
    print(f"Caption: {caption}")