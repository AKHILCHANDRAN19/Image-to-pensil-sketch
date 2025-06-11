#!/usr/bin/env python

# ==============================================================================
#      INTERACTIVE GAN Trainer for Unpaired Photo-to-Sketch
# ==============================================================================
#
#  Description:
#  This script provides full control over the GAN training process.
#  It detects completed epochs, asks the user how many more to train,
#  and then resumes training for that specific duration.
#
#  FEATURES:
#  - INTERACTIVE: Asks for the number of epochs to train per session.
#  - STATEFUL: Knows how many epochs are already done.
#  - RESUME TRAINING: Automatically continues from the last checkpoint.
#  - CHECKPOINTING: Saves progress every few epochs and at the end of a session.
#  - IDENTITY LOSS: INCLUDED to force the model to learn sketch properties,
#    which significantly improves output quality.
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
CHECKPOINT_MODEL_NAME = "Photo_to_Sketch_checkpoint.pth"

# 2. Training Hyperparameters
# --- FOR MOBILE CPU ---
IMG_HEIGHT = 128
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 1          # MUST be 1 for CPU training on mobile
LEARNING_RATE = 0.0002
SAVE_INTERVAL = 5       # Save a checkpoint every 5 epochs.
lambda_identity = 5.0   # Weight for the Identity Loss. This is the key quality booster.

# --- 1. Model Definitions (Unaltered) ---
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1),
        )
    def forward(self, img): return self.model(img)

# --- 2. Custom Dataset (Unaltered) ---
class UnpairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, transform=None):
        self.transform = transform
        self.files_A = sorted([os.path.join(dir_A, f) for f in os.listdir(dir_A)])
        self.files_B = sorted([os.path.join(dir_B, f) for f in os.listdir(dir_B)])
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise FileNotFoundError("A data directory is empty. Check PHOTO_DIR and SKETCH_DIR.")
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        return {"A": item_A, "B": item_B}
    def __len__(self): return max(len(self.files_A), len(self.files_B))

# --- 3. Main Interactive Training Script ---
def main():
    print("="*50)
    print("  INTERACTIVE Photo-to-Sketch GAN Trainer")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Initialize models and optimizers ---
    G = GeneratorUNet().to(device)
    D = Discriminator().to(device)
    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss() # For identity loss
    optimizer_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- Check for Checkpoint and Report Status ---
    start_epoch = 0
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME)
    
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ Model has completed {start_epoch} epochs.")
    else:
        old_model_path = os.path.join(OUTPUT_DIR, "generator_Photo_to_Sketch.pth")
        if os.path.exists(old_model_path):
            print(f"\nFound old model file: {old_model_path}")
            G.load_state_dict(torch.load(old_model_path, map_location=device))
            start_epoch = 20
            print(f"✅ Imported old model. Assuming {start_epoch} epochs completed.")
        else:
            print("\nNo checkpoint found. Starting a new training session.")
            print("✅ Model has completed 0 epochs.")

    # --- Ask User for Number of Epochs to Train ---
    additional_epochs = 0
    while True:
        try:
            prompt = f"\nHow many ADDITIONAL epochs do you want to train? (Enter a number) "
            user_input = input(prompt)
            if user_input.lower() in ['q', 'quit', 'exit']: print("Exiting trainer."); sys.exit(0)
            additional_epochs = int(user_input)
            if additional_epochs <= 0: print("Please enter a positive number > 0."); continue
            break
        except ValueError: print("Invalid input. Please enter a whole number.")
            
    target_epoch = start_epoch + additional_epochs
    print(f"\n--> OK! Training for {additional_epochs} more epochs to reach a total of {target_epoch}.")

    # --- Dataloader and Transforms ---
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        dataloader = DataLoader(
            UnpairedImageDataset(PHOTO_DIR, SKETCH_DIR, transform=transform),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
    except FileNotFoundError as e: print(f"FATAL ERROR: {e}"); sys.exit(1)

    D_output_shape = D(torch.empty(1, CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)).shape

    # --- Training Loop ---
    for epoch in range(start_epoch, target_epoch):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{target_epoch}", unit="batch")

        for i, batch in enumerate(progress_bar):
            real_photos = batch["A"].to(device)
            real_sketches = batch["B"].to(device)
            valid = torch.ones(real_photos.size(0), *D_output_shape[1:], requires_grad=False).to(device)
            fake = torch.zeros(real_photos.size(0), *D_output_shape[1:], requires_grad=False).to(device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_sketches = G(real_photos)
            loss_GAN_G = criterion_GAN(D(fake_sketches), valid)
            
            # --- IDENTITY LOSS CALCULATION ---
            loss_identity_G = criterion_identity(G(real_sketches), real_sketches)
            
            # --- TOTAL GENERATOR LOSS ---
            loss_G = loss_GAN_G + (loss_identity_G * lambda_identity)
            
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(D(real_sketches), valid)
            loss_fake = criterion_GAN(D(fake_sketches.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # --- Update Progress Bar ---
            progress_bar.set_postfix(G_adv=f"{loss_GAN_G.item():.3f}", G_id=f"{loss_identity_G.item():.3f}", D_loss=f"{loss_D.item():.3f}")

        # --- Save Checkpoint periodically ---
        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch + 1, 'G_state_dict': G.state_dict(), 'D_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(), 'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)

    # --- Final Save at the end of the session ---
    print(f"\n--- Saving final checkpoint for epoch {target_epoch} ---")
    torch.save({
        'epoch': target_epoch, 'G_state_dict': G.state_dict(), 'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(), 'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)
    
    print("\n" + "="*50)
    print(f"✅ Training session complete! Model has now been trained for a total of {target_epoch} epochs.")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("="*50)

if __name__ == "__main__":
    main()
