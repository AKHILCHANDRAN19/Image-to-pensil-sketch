#@title 0. Install gdown and Setup
!pip install -q gdown torch torchvision torchaudio matplotlib
!pip install --upgrade --no-cache-dir gdown # Ensure gdown is up-to-date for Drive links

import os
import zipfile
import gdown
import glob
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from google.colab import files # For file uploads

# Hyperparameters (you can tune these)
IMAGE_SIZE = 256
BATCH_SIZE = 1 # Pix2Pix often uses batch_size=1 for instance normalization
LR = 0.0002
BETA1 = 0.5 # Adam optimizer beta1
LAMBDA_L1 = 100 # Weight for L1 loss (reconstruction loss)
MODEL_SAVE_DIR = "saved_models_pix2pix"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Models will be saved in: ./{MODEL_SAVE_DIR}/")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#@title 1. Download and Unzip Datasets
#@markdown Provide the Google Drive File IDs for your datasets:
input_images_zip_id = "1P5i_vNdzrBowpxLjfOAoYcaVp4tg_vzU" #@param {type:"string"}
target_sketches_zip_id = "1PE_RZ90jVsu7cGs5BgVL48_nNwmMAbLi" #@param {type:"string"}

input_zip_path = "input_images.zip"
target_zip_path = "target_sketches.zip"

print("Downloading input images...")
gdown.download(id=input_images_zip_id, output=input_zip_path, quiet=False)
print("\nDownloading target sketches...")
gdown.download(id=target_sketches_zip_id, output=target_zip_path, quiet=False)

# Unzip
input_data_dir = "input_images_data"
target_data_dir = "target_sketches_data"

# Clean up directories before unzipping to avoid issues from previous runs
if os.path.exists(input_data_dir):
    !rm -rf {input_data_dir}
if os.path.exists(target_data_dir):
    !rm -rf {target_data_dir}
os.makedirs(input_data_dir, exist_ok=True)
os.makedirs(target_data_dir, exist_ok=True)


print(f"\nUnzipping {input_zip_path} to {input_data_dir}...")
with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
    zip_ref.extractall(input_data_dir)
print(f"Unzipping {target_zip_path} to {target_data_dir}...")
with zipfile.ZipFile(target_zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_data_dir)

print("Data download and extraction complete.")
print(f"Input images should be in subdirectories like: ./{input_data_dir}/images/")
print(f"Target sketches should be in subdirectories like: ./{target_data_dir}/sketch/")

print("\nListing content of input_images_data:")
!ls -R ./{input_data_dir}
print("\nListing content of target_sketches_data:")
!ls -R ./{target_data_dir}

#@title 2. Define Dataset and DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

class ImageSketchDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        # Assuming input files are 1.png, 2.png, etc.
        self.input_image_files = sorted(
            glob.glob(os.path.join(input_dir, "*.png")),
            key=lambda x: int(re.search(r'(\d+)\.png$', os.path.basename(x)).group(1)) if re.search(r'(\d+)\.png$', os.path.basename(x)) else float('inf')
        )

        print(f"Searching for input images (*.png) in: {input_dir}")
        print(f"Found {len(self.input_image_files)} input images.")
        if not self.input_image_files:
            print(f"ERROR: No .png files found in {input_dir}. Check your zip structure and the path.")
            print("Expected file names like 1.png, 2.png, etc.")

    def __len__(self):
        return len(self.input_image_files)

    def __getitem__(self, idx):
        input_img_path = self.input_image_files[idx]
        # Extract base name correctly (e.g., '1' from '1.png')
        match = re.search(r'(\d+)\.png$', os.path.basename(input_img_path))
        if not match:
            raise ValueError(f"Could not extract numeric base name from {input_img_path}")
        base_name = match.group(1)

        # Construct target image path (assuming .jpg extension)
        target_img_path_jpg = os.path.join(self.target_dir, f"{base_name}.jpg")
        target_img_path_png = os.path.join(self.target_dir, f"{base_name}.png") # Also check for .png target

        target_img_path = None
        if os.path.exists(target_img_path_jpg):
            target_img_path = target_img_path_jpg
        elif os.path.exists(target_img_path_png):
            target_img_path = target_img_path_png
        else:
            raise FileNotFoundError(f"Target sketch for input {input_img_path} (base name {base_name}) not found as {base_name}.jpg or {base_name}.png in {self.target_dir}")

        input_image = Image.open(input_img_path).convert("RGB")
        target_image = Image.open(target_img_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# Determine actual data paths (assuming subdirectories 'images' and 'sketch')
actual_input_dir_name = "images" # This should be the name of the folder inside your input_images.zip
actual_target_dir_name = "sketch" # This should be the name of the folder inside your target_sketches.zip

# Check if the subdirectories exist after unzipping
possible_input_subdirs = [d for d in os.listdir(f'./{input_data_dir}') if os.path.isdir(os.path.join(f'./{input_data_dir}', d))]
if actual_input_dir_name in possible_input_subdirs:
    actual_input_dir = os.path.join(f'./{input_data_dir}', actual_input_dir_name)
elif len(possible_input_subdirs) == 1: # If only one subdir, assume it's the correct one
    actual_input_dir = os.path.join(f'./{input_data_dir}', possible_input_subdirs[0])
    print(f"Warning: Expected input subdir '{actual_input_dir_name}' not found. Using '{possible_input_subdirs[0]}' instead.")
else: # If no subdir or multiple, assume files are at the root of the unzipped folder
    actual_input_dir = f'./{input_data_dir}'
    print(f"Warning: Input subdir '{actual_input_dir_name}' not found. Assuming images are directly in ./{input_data_dir}/")


possible_target_subdirs = [d for d in os.listdir(f'./{target_data_dir}') if os.path.isdir(os.path.join(f'./{target_data_dir}', d))]
if actual_target_dir_name in possible_target_subdirs:
    actual_target_dir = os.path.join(f'./{target_data_dir}', actual_target_dir_name)
elif len(possible_target_subdirs) == 1:
    actual_target_dir = os.path.join(f'./{target_data_dir}', possible_target_subdirs[0])
    print(f"Warning: Expected target subdir '{actual_target_dir_name}' not found. Using '{possible_target_subdirs[0]}' instead.")
else:
    actual_target_dir = f'./{target_data_dir}'
    print(f"Warning: Target subdir '{actual_target_dir_name}' not found. Assuming sketches are directly in ./{target_data_dir}/")

print(f"Final determined input directory: {actual_input_dir}")
print(f"Final determined target directory: {actual_target_dir}")

dataset = ImageSketchDataset(input_dir=actual_input_dir,
                             target_dir=actual_target_dir,
                             transform=transform)

if len(dataset) == 0:
    print("CRITICAL ERROR: Dataset is empty. Please check the paths and file names.")
    print(f"Expected input images in {actual_input_dir} (e.g., {actual_input_dir}/1.png)")
    print(f"Expected target sketches in {actual_target_dir} (e.g., {actual_target_dir}/1.jpg or {actual_target_dir}/1.png)")
else:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} image pairs.")
    try:
        sample_input, sample_target = next(iter(dataloader))
        print("Sample input shape:", sample_input.shape)
        print("Sample target shape:", sample_target.shape)
    except Exception as e:
        print(f"Error getting a sample from dataloader: {e}")

#@title 3. Define Pix2Pix Model (Generator U-Net, Discriminator PatchGAN)

# --- Generator (U-Net) ---
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            # InstanceNorm2d often has affine=True by default for Pix2Pix
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final_up(u7)

# --- Discriminator (PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# --- Initialize models and weights (CORRECTED) ---
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1: # Covers Conv2d and ConvTranspose2d
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Norm") != -1: # Covers BatchNorm2d and InstanceNorm2d
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)

# Explicitly set affine=True in InstanceNorm2d layers or ensure weights_init handles None
# The UNetDown/Up and Discriminator classes now explicitly use affine=True for InstanceNorm
# where normalization is applied, so weights_init_normal should work.
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

print("Generator and Discriminator models defined and initialized.")

#@title 4. Define Loss, Optimizers, and User Input for Epochs

criterion_GAN = nn.BCEWithLogitsLoss().to(device)
criterion_L1 = nn.L1Loss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

start_epoch = 1
total_epochs_to_run = 0

continue_training = input("Do you want to add additional epochs to an existing model? (y/n, default n): ").lower().strip() or "n"

if continue_training == 'y':
    print("Please upload your existing Generator model (.pth file):")
    uploaded_gen = files.upload()
    gen_model_path = None
    if uploaded_gen:
        gen_model_path = list(uploaded_gen.keys())[0]
        print(f"Uploaded Generator: {gen_model_path}")
    else:
        print("No Generator file uploaded. Starting fresh training.")
        continue_training = 'n'

    if continue_training == 'y': # Check again in case first upload failed
        print("Please upload your existing Discriminator model (.pth file):")
        uploaded_disc = files.upload()
        disc_model_path = None
        if uploaded_disc:
            disc_model_path = list(uploaded_disc.keys())[0]
            print(f"Uploaded Discriminator: {disc_model_path}")
        else:
            print("No Discriminator file uploaded. Starting fresh training.")
            continue_training = 'n'

    if continue_training == 'y': # Both files uploaded
        try:
            match = re.search(r'epoch_(\d+)\.pth$', gen_model_path)
            completed_epochs = 0
            if match:
                completed_epochs = int(match.group(1))
                print(f"Detected model completed {completed_epochs} epochs.")
            else:
                completed_epochs_input = input("Could not auto-detect. Enter number of ALREADY completed epochs (default 0): ")
                completed_epochs = int(completed_epochs_input) if completed_epochs_input else 0
            start_epoch = completed_epochs + 1

            generator.load_state_dict(torch.load(gen_model_path, map_location=device))
            discriminator.load_state_dict(torch.load(disc_model_path, map_location=device))
            # Consider loading optimizer states if saved previously for smoother continuation
            # e.g., optimizer_G.load_state_dict(torch.load(optimizer_G_path))
            print("Previously trained models loaded successfully.")

            additional_epochs_input = input(f"How many ADDITIONAL epochs to run? (Starts from epoch {start_epoch}, default 5): ")
            additional_epochs = int(additional_epochs_input) if additional_epochs_input else 5
            total_epochs_to_run = start_epoch + additional_epochs - 1

        except Exception as e:
            print(f"Error loading models or parsing epochs: {e}. Reverting to fresh training.")
            generator = GeneratorUNet().to(device) # Re-initialize
            discriminator = Discriminator().to(device) # Re-initialize
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)
            optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999)) # Re-init optimizers
            optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
            start_epoch = 1
            continue_training = 'n' # Ensure we fall into the 'n' block below

if continue_training == 'n':
    # This block is for fresh training or if 'y' path failed and reverted
    default_total_epochs = 5
    epochs_prompt = f"How many epochs do you want to train? (default {default_total_epochs}): "
    if start_epoch > 1: # Means 'y' path failed AFTER setting start_epoch but before total_epochs_to_run
        epochs_prompt = f"Model loading failed. How many epochs to run starting from epoch {start_epoch}? (default {default_total_epochs}): "

    total_epochs_input = input(epochs_prompt)
    num_epochs_for_this_run = int(total_epochs_input) if total_epochs_input else default_total_epochs
    total_epochs_to_run = start_epoch + num_epochs_for_this_run - 1


print(f"Training will run from epoch {start_epoch} to {total_epochs_to_run}.")

#@title 5. Training Loop
if 'dataloader' not in globals() or not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
    print("CRITICAL ERROR: Dataloader not initialized or empty. Skipping training. Please re-run Cell 2 and ensure dataset is loaded correctly.")
else:
    print(f"\n--- Starting Training from Epoch {start_epoch} to {total_epochs_to_run} ---")
    
    # Determine PatchGAN output size once before the loop
    patch_size = (0,0)
    try:
        example_input_A, _ = next(iter(dataloader))
        example_input_A = example_input_A.to(device)
        with torch.no_grad():
            discriminator.eval() # ensure D is in eval for this check if it has dropout/bn
            example_D_out = discriminator(example_input_A, example_input_A) # Use dummy input B
            discriminator.train() # Set back to train
            patch_size = example_D_out.shape[2:] # (H, W) of the patch
            print(f"Discriminator PatchGAN output size detected: {example_D_out.shape}")
            print(f"Ground truth tensor shape for GAN loss will be: (batch_size, 1, {patch_size[0]}, {patch_size[1]})")
    except Exception as e:
        print(f"Could not determine PatchGAN output size: {e}. Using default (16,16). This might cause issues.")
        patch_size = (16,16) # Fallback, adjust if your D is very different

    if total_epochs_to_run < start_epoch:
        print(f"Warning: total_epochs_to_run ({total_epochs_to_run}) is less than start_epoch ({start_epoch}). No training will occur.")

    for epoch in range(start_epoch, total_epochs_to_run + 1):
        generator.train()
        discriminator.train()

        total_loss_G = 0
        total_loss_D = 0
        
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            valid = torch.ones((real_A.size(0), 1, patch_size[0], patch_size[1]), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, patch_size[0], patch_size[1]), requires_grad=False).to(device)
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            fake_B = generator(real_A)
            
            pred_real = discriminator(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, valid)
            
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            total_loss_D += loss_D.item()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            # fake_B is already generated by G(real_A)
            pred_fake_for_G = discriminator(real_A, fake_B)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, valid)
            loss_G_L1 = criterion_L1(fake_B, real_B)
            loss_G = loss_G_GAN + LAMBDA_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            total_loss_G += loss_G.item()

            if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
                print(
                    f"[Epoch {epoch}/{total_epochs_to_run}] [Batch {i+1}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f} (GAN: {loss_G_GAN.item():.4f}, L1: {loss_G_L1.item():.4f})]"
                )
        
        avg_loss_G = total_loss_G / len(dataloader) if len(dataloader) > 0 else 0
        avg_loss_D = total_loss_D / len(dataloader) if len(dataloader) > 0 else 0
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Average D loss: {avg_loss_D:.4f}, Average G loss: {avg_loss_G:.4f}")

        gen_path = os.path.join(MODEL_SAVE_DIR, f"generator_epoch_{epoch}.pth")
        disc_path = os.path.join(MODEL_SAVE_DIR, f"discriminator_epoch_{epoch}.pth")
        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)
        print(f"Saved models for epoch {epoch} at: \n  {gen_path}\n  {disc_path}")

        if epoch % 1 == 0:
            generator.eval()
            with torch.no_grad():
                try:
                    vis_A, vis_B = next(iter(dataloader)) # Get a fresh batch
                    vis_A = vis_A.to(device)
                    vis_B = vis_B.to(device)
                    generated_vis_B = generator(vis_A)
                    
                    img_sample_input = (vis_A * 0.5 + 0.5).cpu()
                    img_sample_generated = (generated_vis_B * 0.5 + 0.5).cpu()
                    img_sample_real = (vis_B * 0.5 + 0.5).cpu()
                    
                    if img_sample_input.size(0) > 0: # Ensure batch is not empty
                        concatenated_img = torch.cat((img_sample_input[0], img_sample_generated[0], img_sample_real[0]), dim=2)
                        os.makedirs(f"{MODEL_SAVE_DIR}/sample_images", exist_ok=True)
                        sample_img_path = os.path.join(MODEL_SAVE_DIR, "sample_images", f"epoch_{epoch}_sample.png")
                        save_image(concatenated_img, sample_img_path, normalize=False)
                        print(f"Saved sample image for epoch {epoch} at: {sample_img_path}")
                except Exception as e:
                    print(f"Error generating or saving sample image: {e}")
            generator.train()

    print("--- Training Finished ---")
    print(f"Final models saved in ./{MODEL_SAVE_DIR}/")

#@title 6. (Optional) Generate a sketch from a new image
def generate_sketch_from_file():
    print("Please upload the Generator model (.pth) you want to use for inference:")
    uploaded_model = files.upload()
    if not uploaded_model:
        print("No model file uploaded. Aborting.")
        return
    
    model_path = list(uploaded_model.keys())[0]
    print(f"Using model: {model_path}")

    infer_generator = GeneratorUNet().to(device)
    try:
        infer_generator.load_state_dict(torch.load(model_path, map_location=device))
        infer_generator.eval()
        print("Generator model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nPlease upload the input photo to convert to sketch:")
    uploaded_image = files.upload()
    if not uploaded_image:
        print("No image file uploaded. Aborting.")
        return
    
    input_image_path = list(uploaded_image.keys())[0]
    print(f"Processing image: {input_image_path}")

    try:
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_sketch_tensor = infer_generator(img_tensor)
        
        generated_sketch_tensor = (generated_sketch_tensor * 0.5 + 0.5).cpu().squeeze(0)
        
        output_sketch_path = "generated_sketch.png"
        save_image(generated_sketch_tensor, output_sketch_path, normalize=False)
        print(f"Sketch saved as {output_sketch_path}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Input Photo")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        generated_sketch_display = generated_sketch_tensor.permute(1, 2, 0).numpy()
        plt.imshow(generated_sketch_display)
        plt.title("Generated Sketch")
        plt.axis('off')
        plt.show()
        
        print("\nDownload the generated sketch:")
        files.download(output_sketch_path)

    except Exception as e:
        print(f"An error occurred during sketch generation: {e}")

# To run inference after training or with a pre-trained model:
# 1. Ensure a trained generator model (.pth) is available.
# 2. Uncomment and run the line below.
# generate_sketch_from_file()
