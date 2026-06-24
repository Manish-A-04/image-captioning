import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import ImageCaptioningModel
from dataloader import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    
    total_loss   = 0
    total_steps  = 0

    for batch in dataloader:
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        captions = batch['caption'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # In HuggingFace, labels are shifted internally. We just pass input_ids as labels.
        # But we need to ignore pad tokens in loss.
        labels = captions.clone()
        labels[labels == model.tokenizer.pad_token_id] = -100

        optimizer.zero_grad()
        
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)
            
            labels = captions.clone()
            labels[labels == model.tokenizer.pad_token_id] = -100

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageCaptioningModel(
        encoder_name=config.get('encoder_name', "google/vit-base-patch16-224-in21k"),
        decoder_name=config.get('decoder_name', "gpt2")
    ).to(device)

    train_loader, val_loader = get_dataloaders(config)

    # Use a single learning rate for simplicity, or split if needed
    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 5e-5), weight_decay=config.get('weight_decay', 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf')
    patience = config.get('patience', 5)
    patience_counter = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"Train loss      : {train_loss:.4f}")
        print(f"Validation loss : {val_loss:.4f}")
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
                print("Early stopping triggered")
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
        'encoder_name':     'google/vit-base-patch16-224-in21k',
        'decoder_name':     'gpt2',
        'checkpoint_dir':   'checkpoints',
        'max_len':          128,
        'learning_rate':    5e-5,
        'weight_decay':     1e-4,
        'epochs':           20,
        'batch_size':       16,
        'patience':         5,
        'annotations_path': os.path.join('..', 'data', 'raw', 'tiny-trcap-en.json'),
        'image_dir':        os.path.join('..', 'data', 'raw', 'images'),
        'split_ratios':     (0.70, 0.10, 0.20),
        'caption_mode':     'random',
        'augment':          True,
        'num_workers':      0,
        'seed':             42,
    }
    train(config)