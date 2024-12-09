import os
from PIL import Image
import torch
from torchvision import transforms


def main():
    gt_image_list = []
    gt_image_folder = "/data/pwojcik/CelebAHQ/"

    # Define a transformation pipeline to load images as tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor (values in range [0, 1])
    ])

    # Check if the folder exists
    if not os.path.exists(gt_image_folder):
        print(f"Error: The folder '{gt_image_folder}' does not exist.")
        return

    # Loop over all JPG files in the folder
    for image_file in os.listdir(gt_image_folder):
        print(image_file)
        if image_file.endswith(".jpg"):
            try:
                # Load the image
                img_path = os.path.join(gt_image_folder, image_file)
                img = Image.open(img_path).convert("RGB")  # Ensure RGB format

                # Apply the transform
                img_tensor = transform(img)  # Shape: [3, H, W]

                # Reshape and convert to byte format
                reshaped_tensor = img_tensor.permute(1, 2, 0).reshape(-1, 3) * 255  # Shape: [H*W, 3]
                reshaped_tensor = reshaped_tensor.byte()  # Convert to byte

                # Append to the list
                gt_image_list.append(reshaped_tensor)
            except Exception as e:
                print(f"Error processing file '{image_file}': {e}")

    # Stack all tensors into a single tensor
    if gt_image_list:
        image_tensors = torch.stack(gt_image_list)
        print(f"Processed {len(gt_image_list)} images. Tensor shape: {image_tensors.shape}")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    main()
