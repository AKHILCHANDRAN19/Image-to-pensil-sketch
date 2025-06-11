#base script for training 20 epochs

#!/usr/bin/env python

# ==============================================================================
#           GAN Trainer for Unpaired Photo-to-Sketch Translation
# ==============================================================================
#
#  Description:
#  This script trains a GAN model to convert photos into sketches using
#  UNPAIRED data.
#
#  This version is simplified and optimized to ONLY train the Photo-to-Sketch
#  model, making it significantly faster than a full CycleGAN.
#
#  NOTE: Training a GAN on unpaired data without cycle consistency is challenging
#  and may sometimes result in "mode collapse" (where the generator produces
#  similar-looking sketches for different photos).
#
# ==============================================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
import random

# --- Configuration Parameters ---

# 1. Data and Model Paths
PHOTO_DIR = "/storage/emulated/0/Datasets/photos"
SKETCH_DIR = "/storage/emulated/0/Datasets/sketch"
OUTPUT_DIR = "/storage/emulated/0/Download"
MODEL_NAME = "generator_Photo_to_Sketch.pth" # Only one model is created now

# 2. Training Hyperparameters
# --- FOR MOBILE CPU (Keep these low) ---
IMG_HEIGHT = 128
IMG_WIDTH = 224      # Approximates a 16:9 ratio (128 * 16/9 = 227.5)
CHANNELS = 3
BATCH_SIZE = 1      # MUST be 1 for CPU training on mobile
EPOCHS = 20         # Can be increased as training is faster now
LEARNING_RATE = 0.0002

# --- FOR GPU (You can increase these significantly) ---
# IMG_HEIGHT = 256
# IMG_WIDTH = 448
# BATCH_SIZE = 4 or 8

# --- 1. Model Definitions (No changes needed here) ---

# Generator (based on a U-Net architecture)
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

# Discriminator (for telling real vs. fake sketches apart)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        # A simple "PatchGAN" discriminator
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, img):
        return self.model(img)

# --- 2. Custom Dataset for Unpaired Data (No changes needed here) ---
class UnpairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, transform=None):
        self.transform = transform
        self.files_A = sorted([os.path.join(dir_A, f) for f in os.listdir(dir_A)])
        self.files_B = sorted([os.path.join(dir_B, f) for f in os.listdir(dir_B)])
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise FileNotFoundError("One of the data directories is empty. Check PHOTO_DIR and SKETCH_DIR.")

    def __getitem__(self, index):
        # We need a photo from domain A and a random sketch from domain B
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# --- 3. Main Training Script ---
def main():
    print("="*50)
    print("  GAN Trainer for Unpaired Photo-to-Sketch")
    print("="*50)

    # --- SPEED OPTIMIZATION: Auto-select device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Training on CPU. This will be slow. For faster training, use a GPU.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    # Dataloader
    try:
        # For GPU, you can try setting num_workers > 0 for faster data loading
        num_workers = 2 if device.type == 'cuda' else 0
        dataloader = DataLoader(
            UnpairedImageDataset(PHOTO_DIR, SKETCH_DIR, transform=transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
        )
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)

    # Initialize models (Only ONE Generator and ONE Discriminator now)
    G = GeneratorUNet().to(device)      # Generator: Photo -> Sketch
    D = Discriminator().to(device)      # Discriminator: Real Sketch vs. Fake Sketch

    # Loss function
    criterion_GAN = torch.nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Calculate the discriminator output shape dynamically
    D_output_shape = D(torch.empty(1, CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)).shape

    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")

        for i, batch in enumerate(progress_bar):
            # Get real images
            real_photos = batch["A"].to(device)  # Real Photos
            real_sketches = batch["B"].to(device) # Real Sketches

            # Adversarial ground truths (dynamically sized for robustness)
            valid = torch.ones(real_photos.size(0), *D_output_shape[1:], requires_grad=False).to(device)
            fake = torch.zeros(real_photos.size(0), *D_output_shape[1:], requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake sketches from real photos
            fake_sketches = G(real_photos)
            
            # GAN loss for the generator: It tries to fool the discriminator
            loss_G = criterion_GAN(D(fake_sketches), valid)
            
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()

            # Loss for real sketches
            loss_real = criterion_GAN(D(real_sketches), valid)

            # Loss for fake sketches (use .detach() to not train the generator here)
            loss_fake = criterion_GAN(D(fake_sketches.detach()), fake)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            progress_bar.set_postfix(G_loss=f"{loss_G.item():.3f}", D_loss=f"{loss_D.item():.3f}")

    # --- Save Final Model ---
    print("\nTraining complete!")
    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(f"✅ Model saved to {os.path.join(OUTPUT_DIR, MODEL_NAME)}")

if __name__ == "__main__":
    main()
