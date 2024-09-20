import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import multiprocessing
import os

from torch.nn.parallel import DataParallel

from tqdm import tqdm

from inr.inr import INR
from utils import prepare_image, visualize_results


# Training function for a single image (modified for GPU parallelization)
def train_inr_single_image(args):
    image_path, image_size, hidden_dim, num_hidden_layers, num_epochs, learning_rate, device, save_dir, gpu_id = args

    torch.cuda.set_device(gpu_id)
    coords, pixels = prepare_image(image_path, image_size)
    model = INR(input_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, output_dim=3).to(device)
    model = DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    coords, pixels = coords.to(device), pixels.to(device)

    best_loss = float('inf')
    img_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f'inr_model_{img_name[:-4]}.pth')

    pbar = tqdm(range(num_epochs), desc=f"Training {img_name} on GPU {gpu_id}", position=gpu_id, leave=True)

    for epoch in pbar:
        optimizer.zero_grad()
        outputs = model(coords.unsqueeze(0))  # Add batch dimension
        loss = criterion(outputs.squeeze(0), pixels)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.module.state_dict(), save_path)  # Save the model without DataParallel wrapper

        if (epoch + 1) % 1000 == 0:  # Log every 1000 epochs
            pbar.set_description(
                f"{img_name} on GPU {gpu_id}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f'{img_name} on GPU {gpu_id}: Final Loss: {loss.item():.4f}')
    print(f'{img_name} on GPU {gpu_id}: Best model saved to {save_path}')

    # Visualize results
    viz_save_path = os.path.join(save_dir, f'inr_result_{img_name[:-4]}.png')
    visualize_results(model.module, image_size, device, viz_save_path)

    return img_name, best_loss


# Main execution
def main(args):
    # Use parsed arguments
    hidden_dim = args.hidden_dim
    num_hidden_layers = args.num_hidden_layers
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    image_size = args.image_size
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    # Get list of image paths
    image_paths = [os.path.join(dataset_dir, img_name) for img_name in os.listdir(dataset_dir) if
                   img_name.endswith('.jpg')]

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        device = torch.device("cpu")
        num_gpus = 1  # To allow the loop to run once for CPU
    else:
        print(f"Number of available GPUs: {num_gpus}")
        device = torch.device("cuda")

    # Prepare arguments for each image
    args_list = [(image_path, image_size, hidden_dim, num_hidden_layers, num_epochs, learning_rate, device, save_dir,
                  i % num_gpus)
                 for i, image_path in enumerate(image_paths)]

    # Use ProcessPoolExecutor for parallel processing
    num_processes = min(num_gpus, len(image_paths))  # Use at most one process per GPU
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(
            tqdm(executor.map(train_inr_single_image, args_list), total=len(args_list), desc="Overall Progress"))

    # Print summary of results
    for img_name, best_loss in results:
        print(f"{img_name}: Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Necessary for CUDA to work with multiprocessing

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train INR models on multiple images using GPUs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the INR model")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Number of hidden layers in the INR model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=64, help="Size to resize images to")
    parser.add_argument("--dataset_dir", type=str, default="./data/celeb",
                        help="Directory containing the image dataset")
    parser.add_argument("--save_dir", type=str, default="inr_models",
                        help="Directory to save models and visualizations")

    args = parser.parse_args()

    main(args)
