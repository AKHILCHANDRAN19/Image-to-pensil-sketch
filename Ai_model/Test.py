#!/usr/bin/env python

# ==============================================================================
#      Photo-to-Sketch Model INFERENCE SCRIPT
# ==============================================================================
#
#  Author: You & Your AI Assistant
#  Version: 1.0
#
#  Description:
#  This script loads the pre-trained CycleGAN model and uses it to convert
#  a folder of photos into sketches. It's designed to work with the model
#  created by the 'High-Accuracy Photo-to-Sketch Trainer'.
#
# ==============================================================================

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Configuration: Set your paths here ---
MODEL_PATH = "/storage/emulated/0/Download/CycleGAN_PhotoSketch_checkpoint.pth"
INPUT_DIR = "/storage/emulated/0/Test"
OUTPUT_DIR = "/storage/emulated/0/Download"

# --- Model Parameters (MUST MATCH THE TRAINING SCRIPT) ---
IMG_HEIGHT = 128
IMG_WIDTH = 224
CHANNELS = 3

# --- 1. Model Definition (MUST be identical to the training script's Generator) ---
# We only need the Generator for inference, not the Discriminator.
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128*2, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64*2, out_channels, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        u1 = self.up1(d3); u2 = self.up2(torch.cat([u1, d2], 1)); u3 = self.up3(torch.cat([u2, d1], 1))
        return u3

# --- 2. Main Inference Function ---
def main():
    print("="*50)
    print("  Starting Photo-to-Sketch Conversion")
    print("="*50)

    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: Input directory not found at {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device (CPU for mobile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the Photo-to-Sketch Generator
    model = GeneratorUNet().to(device)

    # Load the trained weights
    print(f"Loading trained model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # This key 'G_P2S_state_dict' must match exactly what was saved in the training script
    model.load_state_dict(checkpoint['G_P2S_state_dict'])
    
    # Set the model to evaluation mode. This is crucial for getting correct results.
    model.eval()
    print("✅ Model loaded successfully.")

    # Define the same image transformations used during training
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get list of images to process
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"⚠️ No images found in {INPUT_DIR}. Nothing to do.")
        return

    print(f"\nFound {len(image_files)} images to convert. Processing...")

    # Process each image
    # The 'with torch.no_grad():' block disables gradient calculation,
    # making inference faster and use less memory.
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Converting Images"):
            img_path = os.path.join(INPUT_DIR, filename)
            
            # Open, convert to RGB, and transform the image
            input_image = Image.open(img_path).convert("RGB")
            input_tensor = transform(input_image)
            
            # The model expects a batch of images, so we add a 'batch' dimension
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Run the image through the model
            output_tensor = model(input_tensor)

            # Post-process the output tensor to be a savable image
            # 1. Remove the 'batch' dimension
            output_tensor = output_tensor.squeeze(0)
            # 2. Denormalize: from [-1, 1] back to [0, 1]
            output_tensor = 0.5 * (output_tensor + 1.0)
            # 3. Clamp values to be safe
            output_tensor = torch.clamp(output_tensor, 0, 1)

            # Convert tensor to a PIL Image
            output_image = transforms.ToPILImage()(output_tensor)

            # Save the result
            output_filename = f"sketch_{filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            output_image.save(output_path)

    print("\n" + "="*50)
    print(f"✅ Conversion complete!")
    print(f"All sketches have been saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
