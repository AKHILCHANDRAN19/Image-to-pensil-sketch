from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import os

def pencil_sketch(input_path, output_folder):
    # Load the image using Pillow
    img = Image.open(input_path).convert('L')
    img_array = np.array(img)

    # Invert the grayscale image
    inverted_img_array = 255 - img_array
    
    # Apply Gaussian blur to the inverted image
    blurred_img_array = gaussian_filter(inverted_img_array, sigma=10)
    
    # Convert image arrays to float to prevent overflow issues
    img_array = img_array.astype(float)
    blurred_img_array = blurred_img_array.astype(float)
    
    # Blend the original grayscale image with the blurred inverted image
    sketch_img_array = np.divide(img_array, 255 - blurred_img_array, out=np.zeros_like(img_array, dtype=float), where=(255 - blurred_img_array) != 0)

    # Normalize and convert to uint8 for realistic pencil sketch effect
    sketch_img_array = np.clip(sketch_img_array * 255, 0, 255).astype(np.uint8)

    # Convert the array back to an image
    sketch_img = Image.fromarray(sketch_img_array)

    # Extract the original image format or use the file extension as fallback
    img_format = img.format
    if img_format is None:
        img_format = os.path.splitext(input_path)[1][1:].lower()  # Extract file extension
    
    # Define the output file path
    base_name = os.path.basename(input_path)
    output_name = os.path.splitext(base_name)[0] + '_sketch.' + img_format
    output_path = os.path.join(output_folder, output_name)
    
    # Save the pencil sketch image
    sketch_img.save(output_path, format=img_format.upper())
    
    # Print confirmation message
    print(f"Image saved successfully: {output_path}")

# Paths for mobile internal storage
downloads_folder = '/storage/emulated/0/Download'
output_folder = '/storage/emulated/0/OUTPUT'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all images in the Downloads folder
for file_name in os.listdir(downloads_folder):
    input_path = os.path.join(downloads_folder, file_name)
    if os.path.isfile(input_path):
        try:
            pencil_sketch(input_path, output_folder)
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
