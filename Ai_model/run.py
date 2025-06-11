#!/usr/bin/env python

# ==============================================================================
#          Photo-to-Sketch Inference (Testing) Script
# ==============================================================================
#
#  Description:
#  This script loads a pre-trained generator model and uses it to
#  convert photos from a specified input folder into pencil sketches.
#  The resulting sketches are saved to an output folder.
#
# ==============================================================================

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import sys

# --- Configuration Parameters ---

# 1. Paths
TEST_IMAGE_DIR = "/storage/emulated/0/Test"
OUTPUT_DIR = "/storage/emulated/0/Download"
MODEL_PATH = "/storage/emulated/0/Download/generator_Photo_to_Sketch.pth"

# 2. Model & Image Parameters (MUST match the training script)
IMG_HEIGHT = 128
IMG_WIDTH = 224
CHANNELS = 3

# --- 1. Model Definition (Must be EXACTLY the same as in the training script) ---

# This class structure is required for PyTorch to understand how to load the saved model weights.
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        # A simplified U-Net for mobile
        # NOTE: Using InstanceNorm2d is better for image style-transfer tasks than BatchNorm2d
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128*2, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64*2, out_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3)
        u2 = self.up2(torch.cat([u1, d2], 1))
        u3 = self.up3(torch.cat([u2, d1], 1))
        return u3


# --- 2. Main Testing Function ---
def test_model():
    print("="*50)
    print("     Photo-to-Sketch Model Testing")
    print("="*50)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Check for Model and Input Directory ---
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"FATAL ERROR: Test image folder not found at {TEST_IMAGE_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Initialize and Load the Generator Model ---
    print(f"\nLoading model from: {MODEL_PATH}")
    generator = GeneratorUNet().to(device)

    # Use map_location to ensure model loads correctly even if trained on a different device
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Set the model to evaluation mode. This is VERY IMPORTANT for inference.
    generator.eval()
    print("Model loaded successfully.")

    # --- Define Image Transformations (Must match training) ---
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    # --- Process Images in the Test Folder ---
    print(f"\nProcessing images from: {TEST_IMAGE_DIR}")
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the test directory. Exiting.")
        return

    # Use tqdm for a nice progress bar
    for image_name in tqdm(image_files, desc="Converting images"):
        try:
            input_path = os.path.join(TEST_IMAGE_DIR, image_name)

            # Open image and ensure it's in RGB format
            img = Image.open(input_path).convert("RGB")

            # Apply transformations
            input_tensor = transform(img).to(device)
            # Add a batch dimension (from [C, H, W] to [1, C, H, W])
            input_tensor = input_tensor.unsqueeze(0)

            # Generate the sketch
            # torch.no_grad() is used to prevent calculating gradients, saving memory and speeding up the process
            with torch.no_grad():
                output_tensor = generator(input_tensor)

            # --- Save the Output Sketch ---
            output_filename = f"sketch_{image_name}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # save_image handles the de-normalization from [-1, 1] to [0, 255]
            save_image(output_tensor, output_path, normalize=True)

        except Exception as e:
            print(f"\nSkipping file {image_name} due to an error: {e}")

    print("\n" + "="*50)
    print(f"✅ Conversion complete! Sketches saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    test_model()
