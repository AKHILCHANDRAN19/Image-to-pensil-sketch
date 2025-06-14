# -*- coding: utf-8 -*-
"""
Photo to Sketch Pix2Pix Training Script (Mobile & CPU Friendly - CORRECTED)

This script trains a Pix2Pix Generative Adversarial Network (GAN) to convert
color photographs into grayscale pencil sketches.

This version corrects a critical bug in the Generator model that caused an
'UnboundLocalError' in previous attempts.

--- HOW TO USE ---
1.  Ensure your folder structure is EXACTLY:
    - /storage/emulated/0/Datasets/images/ (contains 1.png, 2.png, ...)
    - /storage/emulated/0/Datasets/sketch/ (contains 1.jpg, 2.jpg, ...)
2.  Copy and paste this entire script and run it. It will find your data
    and begin training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    """Centralized configuration for all settings and hyperparameters."""
    # Set device to CPU. This script is optimized for mobile/CPU.
    DEVICE = torch.device("cpu")

    # --- Directory Paths (FIXED) ---
    # Using the exact hardcoded path as requested.
    BASE_DIR = "/storage/emulated/0/Datasets"
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    SKETCH_DIR = os.path.join(BASE_DIR, "sketch")
    SAVE_MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
    SAVE_EXAMPLE_DIR = os.path.join(BASE_DIR, "examples")

    # --- Training Hyperparameters (Mobile Optimized) ---
    LEARNING_RATE = 2e-4
    # BATCH_SIZE must be low for mobile CPUs. Start with 1.
    BATCH_SIZE = 1
    # Total number of training epochs. Increase for better results if you have time.
    NUM_EPOCHS = 25
    IMAGE_SIZE = 256
    L1_LAMBDA = 100

    # --- Model & Checkpoint Settings ---
    # Set to True to use a smaller, faster model. Highly recommended for CPU training.
    MODEL_LITE = True
    # Set to True to resume training from the last saved checkpoint.
    LOAD_MODEL = False

    # Checkpoint file paths
    GEN_CHECKPOINT_LAST = os.path.join(SAVE_MODEL_DIR, "generator_last.pth.tar")
    DISC_CHECKPOINT_LAST = os.path.join(SAVE_MODEL_DIR, "discriminator_last.pth.tar")
    GEN_CHECKPOINT_BEST = os.path.join(SAVE_MODEL_DIR, "generator_best.pth.tar")
    DISC_CHECKPOINT_BEST = os.path.join(SAVE_MODEL_DIR, "discriminator_best.pth.tar")


# --- 2. CUSTOM DATASET ---
class PhotoSketchDataset(Dataset):
    """Custom Dataset for loading paired color photos and pencil sketches."""
    def __init__(self, image_dir, sketch_dir):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.image_files = sorted(os.listdir(self.image_dir))

        self.photo_transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))),
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.sketch_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_file = self.image_files[index]
        sketch_file_base = os.path.splitext(img_file)[0]
        sketch_path_pattern = os.path.join(self.sketch_dir, sketch_file_base + '.*')
        sketch_path_list = glob.glob(sketch_path_pattern)

        if not sketch_path_list:
            raise FileNotFoundError(f"No sketch found for image '{img_file}'")
        sketch_path = sketch_path_list[0]
        img_path = os.path.join(self.image_dir, img_file)

        try:
            photo = Image.open(img_path).convert("RGB")
            sketch = Image.open(sketch_path).convert("L")
        except Exception as e:
            print(f"Error loading image pair: {img_file}, {os.path.basename(sketch_path)}. Error: {e}. Skipping.")
            return torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), torch.zeros(1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

        photo = self.photo_transform(photo)
        sketch = self.sketch_transform(sketch)
        return photo, sketch


# --- 3. PIX2PIX MODEL ARCHITECTURE ---
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
        # *** THIS SECTION IS NOW FIXED ***
        # The code is now sequential to prevent the UnboundLocalError.
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

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x): return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv2d(in_channels + 1, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"), nn.LeakyReLU(0.2))
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

# --- 4. UTILITY FUNCTIONS ---
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def save_checkpoint(model, optimizer, epoch, filename):
    print("=> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    print(f"=> Loading checkpoint from '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", -1) + 1

def save_example_images(gen, loader, epoch, folder):
    try: x, y = next(iter(loader))
    except StopIteration: return
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        x, y = x * 0.5 + 0.5, y * 0.5 + 0.5
        y_rgb, y_fake_rgb = y.repeat(1, 3, 1, 1), y_fake.repeat(1, 3, 1, 1)
        num_images = min(x.shape[0], 4)
        grid = torch.cat([x[:num_images], y_fake_rgb[:num_images], y_rgb[:num_images]], dim=0)
        utils.save_image(grid, os.path.join(folder, f"example_epoch_{epoch+1}.png"), nrow=num_images)
    gen.train()
    print(f"-> Saved example image for epoch {epoch+1}")

# --- 5. TRAINING LOOP ---
def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce_loss):
    loop = tqdm(loader, leave=True, desc="Training")
    total_disc_loss, total_gen_loss = 0.0, 0.0
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        
        # Train Discriminator
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        
        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generator
        D_fake = disc(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * Config.L1_LAMBDA
        G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        total_disc_loss += D_loss.item()
        total_gen_loss += G_loss.item()
        
        if idx % 10 == 0:
            loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    return total_disc_loss / len(loader), total_gen_loss / len(loader)

# --- 6. MAIN EXECUTION ---
def main():
    print("--- Photo2Sketch Training Initializing (CPU/Mobile Optimized) ---")
    print(f"Using device: {Config.DEVICE}")
    print(f"Dataset path: {Config.BASE_DIR}")
    if Config.MODEL_LITE: print("Mode: LITE model enabled for faster CPU training.")

    for path in [Config.SAVE_MODEL_DIR, Config.SAVE_EXAMPLE_DIR]: os.makedirs(path, exist_ok=True)

    if Config.MODEL_LITE: gen_features, disc_features = 32, [32, 64, 128, 256]
    else: gen_features, disc_features = 64, [64, 128, 256, 512]

    disc = Discriminator(in_channels=3, features=disc_features).to(Config.DEVICE)
    gen = Generator(in_channels=3, features=gen_features).to(Config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE, L1_LOSS = nn.BCEWithLogitsLoss(), nn.L1Loss()

    start_epoch = 0
    if Config.LOAD_MODEL and os.path.exists(Config.GEN_CHECKPOINT_LAST):
        start_epoch = load_checkpoint(Config.GEN_CHECKPOINT_LAST, gen, opt_gen)
        _ = load_checkpoint(Config.DISC_CHECKPOINT_LAST, disc, opt_disc)
    else:
        print("\n-> No checkpoint found or LOAD_MODEL=False. Initializing new weights.")
        initialize_weights(disc)
        initialize_weights(gen)

    print("\n--- Preparing Data ---")
    try:
        dataset = PhotoSketchDataset(image_dir=Config.IMAGE_DIR, sketch_dir=Config.SKETCH_DIR)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        print(f"Dataset loaded successfully with {len(dataset)} image pairs.")
        if len(dataset) == 0:
            print(f"\nFATAL: The dataset is empty (0 files found).")
            print(f"Please check that files exist in '{Config.IMAGE_DIR}' and '{Config.SKETCH_DIR}'")
            return
    except Exception as e:
        print(f"\nFATAL: Error creating dataset or dataloader: {e}")
        print("Please ensure your dataset folders are correctly populated and accessible.")
        return

    print("\n--- Starting Training ---")
    print("Note: Training on a CPU will be very slow. Be patient!")
    best_gen_loss = float("inf")

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        avg_disc_loss, avg_gen_loss = train_fn(disc, gen, loader, opt_disc, opt_gen, L1_LOSS, BCE)
        print(f"Epoch {epoch+1} Complete. Avg D Loss: {avg_disc_loss:.4f}, Avg G Loss: {avg_gen_loss:.4f}")

        save_checkpoint(gen, opt_gen, epoch, filename=Config.GEN_CHECKPOINT_LAST)
        save_checkpoint(disc, opt_disc, epoch, filename=Config.DISC_CHECKPOINT_LAST)

        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            save_checkpoint(gen, opt_gen, epoch, filename=Config.GEN_CHECKPOINT_BEST)
            save_checkpoint(disc, opt_disc, epoch, filename=Config.DISC_CHECKPOINT_BEST)
            print(f"-> New best model saved! Generator loss: {best_gen_loss:.4f}")

        save_example_images(gen, loader, epoch, Config.SAVE_EXAMPLE_DIR)

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    main()
