# -*- coding: utf-8 -*-
"""
Photo to Sketch Pix2Pix Inference Script (Mobile & CPU Friendly)

This script loads a pre-trained Generator model and uses it to convert a
folder of color photographs into pencil sketches.

--- HOW TO USE ---
1.  Make sure your trained model is at:
    /storage/emulated/0/Datasets/checkpoints/generator_best.pth.tar
2.  Create a folder for your test images at:
    /storage/emulated/0/Test/
3.  Place all the color images you want to convert into the /Test/ folder.
4.  Copy and paste this entire script and run it.
5.  The final sketch images will be saved in:
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

# Ignore a common UserWarning from PIL about large images to keep the log clean
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# --- 1. CONFIGURATION ---
class Config:
    """Centralized configuration for all settings and paths."""
    # Set device to CPU. This script is optimized for mobile/CPU.
    DEVICE = torch.device("cpu")

    # --- Directory Paths (FIXED) ---
    # Path to the trained generator model file.
    MODEL_PATH = "/storage/emulated/0/Datasets/checkpoints/generator_best.pth.tar" # Corrected path to be inside checkpoints
    
    # Path to the folder containing your test images.
    TEST_IMAGE_DIR = "/storage/emulated/0/Test"
    
    # Path to the folder where sketch images will be saved.
    OUTPUT_DIR = "/storage/emulated/0/Download"

    # --- Model & Image Settings ---
    # These MUST match the settings used during training!
    IMAGE_SIZE = 256
    
    # Set to True if you trained with MODEL_LITE=True, otherwise False.
    # This is critical for loading the model correctly.
    MODEL_LITE = True


# --- 2. PIX2PIX MODEL ARCHITECTURE (Copied from training script) ---
# The script needs to know the model's structure to load the weights.
# Only the Generator and its Block are needed for inference.

class Block(nn.Module):
    """A building block for the U-Net Generator."""
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
    """The U-Net Generator Architecture."""
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

# --- 3. UTILITY FUNCTIONS ---
def load_generator(model_path, gen_features):
    """Loads the generator model from a checkpoint file."""
    print(f"=> Loading generator model from '{model_path}'")
    if not os.path.exists(model_path):
        print(f"FATAL: Model file not found at the specified path.")
        print("Please check the path in the Config class.")
        return None
        
    # Instantiate the model with the correct number of features
    gen = Generator(in_channels=3, features=gen_features).to(Config.DEVICE)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    # Load the state dictionary into the model
    gen.load_state_dict(checkpoint["state_dict"])
    
    # Set the model to evaluation mode (very important!)
    gen.eval()
    print("-> Model loaded successfully.")
    return gen

def process_and_save_images(gen, image_paths, output_dir):
    """Processes each image and saves the sketched result."""
    # Define the same image transformations used for the input photos during training
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))),
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Use no_grad to save memory and speed up inference
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Converting Images"):
            try:
                # Open image and convert to RGB
                image = Image.open(img_path).convert("RGB")
                
                # Apply transformations and add a batch dimension
                input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

                # Generate the sketch
                output_tensor = gen(input_tensor)

                # De-normalize the output from [-1, 1] to [0, 1] range
                output_tensor = output_tensor * 0.5 + 0.5

                # Create a unique output filename
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_filename = f"{base_name}_sketch.png"
                output_path = os.path.join(output_dir, output_filename)

                # Save the resulting image
                utils.save_image(output_tensor, output_path)

            except Exception as e:
                print(f"\nCould not process file {os.path.basename(img_path)}. Error: {e}. Skipping.")

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to run the entire inference process."""
    print("--- Photo2Sketch Inference Initializing ---")
    print(f"Using device: {Config.DEVICE}")
    if Config.MODEL_LITE:
        print("Mode: LITE model selected. Ensure this matches your training setting.")
        gen_features = 32
    else:
        print("Mode: FULL model selected. Ensure this matches your training setting.")
        gen_features = 64
        
    # --- Setup Directories ---
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Test image folder: {Config.TEST_IMAGE_DIR}")
    print(f"Output folder:     {Config.OUTPUT_DIR}")

    # --- Load Model ---
    gen = load_generator(Config.MODEL_PATH, gen_features)
    if gen is None:
        return # Exit if model loading failed

    # --- Find and Process Images ---
    print("\n--- Searching for Test Images ---")
    # Find all common image types
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(Config.TEST_IMAGE_DIR, ext)))

    if not image_paths:
        print(f"FATAL: No images found in '{Config.TEST_IMAGE_DIR}'.")
        print("Please add some images to that folder and try again.")
        return
        
    print(f"Found {len(image_paths)} images to convert.")
    
    print("\n--- Starting Conversion Process ---")
    process_and_save_images(gen, image_paths, Config.OUTPUT_DIR)

    print("\n--- Conversion Finished! ---")
    print(f"All sketches have been saved to your Download folder: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
