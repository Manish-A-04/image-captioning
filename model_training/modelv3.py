import torch
import torch.nn as nn
import torchvision.models as models
import math


class VisualPositionEmbedding(nn.Module):

    def __init__(self, embed_dim, grid_h=7, grid_w=7):
        super().__init__()
        n_tokens = grid_h * grid_w
        self.emb = nn.Embedding(n_tokens, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        B, N, _ = x.shape
        pos = torch.arange(N, device=x.device)
        return x + self.emb(pos).unsqueeze(0)


class ImageEncoder(nn.Module):

    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        # Use EfficientNet-B0 for better performance on small datasets 
        # (fewer parameters than ResNet50, good feature extractor)
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = efficientnet.features
        
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        self.project = nn.Linear(1280, embed_dim) # EfficientNet-B0 outputs 1280 channels
        self.pos_emb = VisualPositionEmbedding(embed_dim, grid_h=7, grid_w=7)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = self.backbone(x)
        # EfficientNet-B0 features shape is (B, 1280, 7, 7) for 224x224 input
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.project(tokens)
        tokens = self.pos_emb(tokens)
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return tokens

    def unfreeze_last_layers(self):
        # Unfreeze the last block of EfficientNet
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

    @property
    def last_layers_unfrozen(self):
        return any(p.requires_grad for p in self.backbone[-1].parameters())


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CaptionDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        max_len,
        dropout=0.3,
    ):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc    = PositionalEncoding(embed_dim, max_len, dropout)
        self.image_proj = nn.Linear(embed_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.fc_out   = nn.Linear(embed_dim, vocab_size)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        image_features,
        captions,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        memory = self.image_proj(image_features)
        tgt    = self.embed(captions)
        tgt    = self.pos_enc(tgt)

        output = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = self.out_norm(output)
        logits = self.fc_out(output)
        return logits


class ImageCaptioningModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4, # Reduced for small dataset to prevent overfitting
        num_layers=2, # Reduced for small dataset to prevent overfitting
        ff_dim=512,
        max_len=128,
        dropout=0.3, # Increased dropout for regularization
    ):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim, dropout)
        self.decoder = CaptionDecoder(
            vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len, dropout
        )

    def forward(
        self,
        images,
        captions,
        tgt_key_padding_mask=None,
    ):
        image_features = self.encoder(images)
        seq_len  = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=images.device, dtype=torch.bool
        )
        return self.decoder(
            image_features, captions,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def encode(self, images):
        return self.encoder(images)

    def decode_step(
        self,
        image_features,
        captions,
    ):
        seq_len  = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=captions.device, dtype=torch.bool
        )
        return self.decoder(image_features, captions, tgt_mask=tgt_mask)
