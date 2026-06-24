import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import ImageCaptioningModel
from dataloader import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, criterion, device, vocab, pad_idx=0):
    model.train()

    model.encoder.backbone.eval()
    if model.encoder.layer4_is_unfrozen:
        model.encoder.backbone[-1].train()

    total_loss   = 0
    total_tokens = 0

    for batch in dataloader:
        if batch is None:
            continue
        images   = batch['image'].to(device)
        captions = batch['caption'].to(device)

        input_captions  = captions[:, :-1]
        target_captions = captions[:, 1:]

        pad_mask = (input_captions == pad_idx)

        optimizer.zero_grad()
        logits = model(images, input_captions, tgt_key_padding_mask=pad_mask)

        logits_flat  = logits.reshape(-1, logits.size(-1))
        targets_flat = target_captions.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        non_pad       = (targets_flat != pad_idx).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def validate(model, dataloader, criterion, device, pad_idx=0):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)

            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]

            pad_mask = (input_captions == pad_idx)
            logits = model(images, input_captions, tgt_key_padding_mask=pad_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = target_captions.reshape(-1)

            loss = criterion(logits_flat, targets_flat)

            non_pad = (targets_flat != pad_idx).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / total_tokens


def _maybe_unfreeze_layer4(epoch, model, optimizer, config, patience_counter):
    unfreeze_epoch = config.get('unfreeze_layer4_epoch', 12)
    if epoch != unfreeze_epoch or model.encoder.layer4_is_unfrozen:
        return patience_counter

    model.encoder.unfreeze_layer4()
    layer4_params = [p for p in model.encoder.backbone[-1].parameters() if p.requires_grad]
    optimizer.add_param_group({
        'params':       layer4_params,
        'lr':           config.get('layer4_lr', 1e-5),
        'weight_decay': config.get('weight_decay', 1e-4),
    })
    return 0


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config['vocab_path'], 'r') as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    pad_idx = vocab.get('<PAD>', 0)

    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 3),
        ff_dim=config.get('ff_dim', 512),
        max_len=config.get('max_len', 128),
        dropout=config.get('dropout', 0.1)
    ).to(device)

    train_loader, val_loader = get_dataloaders(config)

    encoder_params = (
        list(model.encoder.project.parameters())
        + list(model.encoder.pos_emb.parameters())
        + list(model.encoder.norm.parameters())
    )
    decoder_params = list(model.decoder.parameters())

    optimizer = AdamW([
        {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-4)},
        {'params': decoder_params, 'lr': config.get('decoder_lr', 3e-4)},
    ], weight_decay=config.get('weight_decay', 1e-4))

    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf')
    patience = config.get('patience', 7)
    patience_counter = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        patience_counter = _maybe_unfreeze_layer4(
            epoch, model, optimizer, config, patience_counter
        )

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, vocab, pad_idx)
        val_loss   = validate(model, val_loader, criterion, device, pad_idx)
        scheduler.step()
        print(f"Train loss : {train_loss}")
        print(f"Validation loss : {val_loss}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print("Best model updated")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['checkpoint_dir'], 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(config['checkpoint_dir'], 'last_model.pt'))

    with open(os.path.join(config['checkpoint_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    config = {
        'vocab_path':       'vocab.json',
        'checkpoint_dir':   'checkpoints',
        'embed_dim':        256,
        'num_heads':        8,
        'num_layers':       3,
        'ff_dim':           512,
        'max_len':          128,
        'dropout':          0.1,
        'encoder_lr':       1e-4,
        'decoder_lr':       3e-4,
        'weight_decay':     1e-4,
        'epochs':           50,
        'batch_size':       32,
        'patience':         7,
        'unfreeze_layer4_epoch': 12,
        'layer4_lr':             1e-5,
        'annotations_path': os.path.join('..', 'data', 'raw', 'tiny-trcap-en.json'),
        'image_dir':        os.path.join('..', 'data', 'raw', 'images'),
        'split_ratios':     (0.70, 0.10, 0.20),
        'caption_mode':     'random',
        'augment':          True,
        'num_workers':      0,
        'seed':             42,
    }
    train(config)