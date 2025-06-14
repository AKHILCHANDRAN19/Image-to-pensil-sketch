# -*- coding: utf-8 -*-
"""
Photo to Sketch Pix2Pix Inference/Testing Script

This script loads a pre-trained Generator model and uses it to convert
a folder of test images into sketches.

--- HOW TO USE ---
1.  Make sure your trained model is at:
    /storage/emulated/0/Datasets/checkpoints/generator_best.pth.tar
2.  Place the images you want to convert into the folder:
    /storage/emulated/0/Test/
3.  Run this script.
4.  The generated sketches will be saved in:
    /storage/emulated/0/Download/
"""
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import os
import glob
from tqdm import tqdm
import warnings

# Ignore common UserWarnings from PIL to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# --- 1. CONFIGURATION ---
class Config:
    """Centralized configuration for testing."""
    # Set device to CPU, as requested for mobile use.
    DEVICE = torch.device("cpu")

    # --- Path Configuration (as per your request) ---
    MODEL_PATH = "/storage/emulated/0/Datasets/checkpoints/generator_best.pth.tar"
    TEST_IMAGE_DIR = "/storage/emulated/0/Test"
    OUTPUT_DIR = "/storage/emulated/0/Download"

    # --- Model & Image Settings (MUST MATCH TRAINING SCRIPT) ---
    IMAGE_SIZE = 256
    # Set to True if you trained with MODEL_LITE=True (the default in your script).
    # If you changed the training script to use the full model, set this to False.
    MODEL_LITE = True


# --- 2. PIX2PIX MODEL ARCHITECTURE (Copied from your training script) ---
# This part MUST be identical to the training script to load the weights.

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"), nn.LeakyReLU(0.2))
        self.down1 = Block(features, features * 2, down=True, act="leaky")
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky")
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky")
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky")
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), nn.ReLU())
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features * 2, 1, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


# --- 3. MAIN INFERENCE FUNCTION ---
def main():
    print("--- Photo to Sketch Inference Initializing ---")
    print(f"Device: {Config.DEVICE}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Input Directory: {Config.TEST_IMAGE_DIR}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")

    # --- Pre-flight Checks ---
    if not os.path.exists(Config.MODEL_PATH):
        print(f"\nFATAL: Model file not found at '{Config.MODEL_PATH}'")
        print("Please ensure your trained model 'generator_best.pth.tar' exists.")
        return

    if not os.path.exists(Config.TEST_IMAGE_DIR):
        print(f"\nFATAL: Test image folder not found at '{Config.TEST_IMAGE_DIR}'")
        print("Please create this folder and add images to it.")
        return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to '{Config.OUTPUT_DIR}'")


    # --- Load Model ---
    print("\n--- Loading Model ---")
    gen_features = 32 if Config.MODEL_LITE else 64
    model = Generator(in_channels=3, features=gen_features).to(Config.DEVICE)
    
    try:
        # Load the state dictionary directly
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    except Exception as e:
        print(f"\nFATAL: Error loading model weights: {e}")
        print("This might happen if the MODEL_LITE flag in this script does not match")
        print("the setting you used for training. Try changing it and run again.")
        return

    # Set the model to evaluation mode (very important!)
    model.eval()
    print("Model loaded successfully.")

    # --- Prepare Image Transformations (must match training) ---
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))),
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # --- Find Test Images ---
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(Config.TEST_IMAGE_DIR, ext)))

    if not image_paths:
        print(f"\nWARNING: No images (.png, .jpg, .jpeg) found in '{Config.TEST_IMAGE_DIR}'.")
        return
        
    print(f"\n--- Found {len(image_paths)} images to process ---")

    # --- Process Images ---
    for img_path in tqdm(image_paths, desc="Generating Sketches"):
        try:
            # 1. Load and Transform Image
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

            # 2. Generate Sketch
            with torch.no_grad(): # Use no_grad for inference
                output_tensor = model(input_tensor)

            # 3. Post-process and Save
            # De-normalize from [-1, 1] to [0, 1]
            output_tensor = output_tensor * 0.5 + 0.5

            # Construct output path
            base_filename = os.path.basename(img_path)
            name, ext = os.path.splitext(base_filename)
            output_filename = f"{name}_sketch.png" # Always save as png
            output_path = os.path.join(Config.OUTPUT_DIR, output_filename)

            # Save the image
            utils.save_image(output_tensor, output_path)

        except Exception as e:
            print(f"\nError processing {os.path.basename(img_path)}: {e}. Skipping.")

    print(f"\n--- Inference Complete ---")
    print(f"All sketches have been saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
