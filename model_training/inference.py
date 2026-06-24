import sys
import torch
from PIL import Image
from torchvision import transforms
from model import ImageCaptioningModel

def get_transform():
    # ViT expects mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def generate_caption(model, image_tensor, device, max_len=128, beam_size=5):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output_ids = model.generate(image_tensor, max_length=max_len, num_beams=beam_size)
    
    caption = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def caption_image_file(image_path, checkpoint_path, beam_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('config', {})

    model = ImageCaptioningModel(
        encoder_name=cfg.get('encoder_name', 'google/vit-base-patch16-224-in21k'),
        decoder_name=cfg.get('decoder_name', 'gpt2')
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    caption = generate_caption(model, image_tensor, device, beam_size=beam_size)
    return caption

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/best_model.pt'

    caption = caption_image_file(image_path, checkpoint_path)
    print(f"Caption: {caption}")