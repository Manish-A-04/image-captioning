import torch
import torch.nn as nn
import torchvision.models as models
import math


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        features = self.project(features)
        features = self.norm(features)
        features = self.dropout(features)
        return features


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, captions, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.image_proj(image_features).unsqueeze(1)
        tgt = self.embed(captions)
        tgt = self.pos_enc(tgt)
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.fc_out(output)
        return logits


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3, ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = CaptionDecoder(vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len, dropout)

    def forward(self, images, captions, tgt_key_padding_mask=None):
        image_features = self.encoder(images)
        seq_len = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=images.device)
        logits = self.decoder(image_features, captions, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits

    def encode(self, images):
        return self.encoder(images)

    def decode_step(self, image_features, captions):
        seq_len = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=captions.device)
        logits = self.decoder(image_features, captions, tgt_mask=tgt_mask)
        return logits