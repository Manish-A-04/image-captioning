import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder_name="google/vit-base-patch16-224-in21k", decoder_name="gpt2"):
        super().__init__()
        # Initialize the model from pretrained components
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name, decoder_name
        )
        
        # Load the tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(decoder_name)
        
        # GPT2 doesn't have a pad token by default, so we set it to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure the model to match tokenizer tokens
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.eos_token_id
            
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Some generation specific parameters
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.decoder_start_token_id = self.model.config.decoder_start_token_id

    def forward(self, pixel_values, labels=None):
        """
        pixel_values: Tensor of shape (batch_size, num_channels, height, width)
        labels: Tensor of shape (batch_size, sequence_length) with padded tokens as -100
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def generate(self, pixel_values, max_length=128, num_beams=5):
        """
        pixel_values: Tensor of shape (batch_size, num_channels, height, width)
        """
        # Disable gradient calculation for generation
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=False
            )
        return outputs.sequences

def build_criterion(pad_idx=-100):
    # Hugging Face models expect -100 for ignored labels
    return nn.CrossEntropyLoss(ignore_index=pad_idx)

if __name__ == '__main__':
    print("Initializing Model...")
    model = ImageCaptioningModel()
    
    B = 2
    # dummy image based on ViT input size
    images = torch.randn(B, 3, 224, 224) 
    
    # dummy labels
    dummy_text = ["Hello world", "Image captioning with ViT and GPT2"]
    encoded = model.tokenizer(dummy_text, padding=True, return_tensors="pt")
    labels = encoded.input_ids
    labels[labels == model.tokenizer.pad_token_id] = -100 # ignore pad token for loss
    
    print("Testing forward pass...")
    outputs = model(pixel_values=images, labels=labels)
    loss = outputs.loss
    
    print(f"Loss: {loss.item():.4f}")
    
    print("Testing generation...")
    preds = model.generate(images, max_length=10)
    decoded = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(f"Generated text: {decoded}")
