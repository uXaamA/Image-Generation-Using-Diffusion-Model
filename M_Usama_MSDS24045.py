# Importing Required Libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import zipfile

# Creating DataLoader
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # ImageFolder load image dataset and arranged in folder structure where each subfolder represent a class
        self.data = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Forward Diffusion Process
class Diffusion:
    # timesteps --> gradually adding noise on image in steps
    # beta_start --> Control how much noise is added each step
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps
        # linespace creates the 1D tensor  1D tensor of evenly spaced values between (start, end, steps)
        self.beta = torch.linspace(beta_start, beta_end, self.T)
        self.alpha = 1. - self.beta    #retained signals at that step
        # cumprod compute the comulative product along the given dimension
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t):
        # creating tensor of same shape as x0 and filled with random values drawn from std normal distribution
        noise = torch.randn_like(x0)
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

# Denoising Model 
class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)

# Loss Function
class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()

    def forward(self, pred_noise, true_noise):
        return F.mse_loss(pred_noise, true_noise)

# Training Loop

def train(model, dataloader, diffusion, epochs=10, lr=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiffusionLoss()
    model.train()

    os.makedirs('samples', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            # long will convert the datatype into int64
            t = torch.randint(0, diffusion.T, (images.size(0),), device=device).long()
            x_noisy, noise = diffusion.add_noise(images, t)
            pred = model(x_noisy, t)
            loss = criterion(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        save_image(x_noisy[:16], f'samples/noisy_epoch{epoch+1}.png')
        save_image(images[:16], f'samples/clean_epoch{epoch+1}.png')
        save_image(pred[:16], f'samples/pred_epoch{epoch+1}.png')

    torch.save(model.state_dict(), 'saved_models/denoise_model.pt')

    # Plot and save loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('samples/loss_graph.png')
    plt.close()

# Sampling Function (Reverse process)
def sample(model, diffusion, shape=(1, 3, 64, 64), device='cuda'):
    model.eval()
    img = torch.randn(shape).to(device)
    for t in reversed(range(diffusion.T)):
        t_batch = torch.tensor([t], device=device).long()
        noise_pred = model(img, t_batch)
        alpha = diffusion.alpha[t]
        alpha_hat = diffusion.alpha_hat[t]
        beta = diffusion.beta[t]

        if t > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * noise_pred) + beta.sqrt() * noise

    return img

# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset or zip file')
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    model = DenoiseModel().to(device)
    diffusion = Diffusion()

    if args.data_path.endswith('.zip'):
        extract_dir = args.data_path.replace('.zip', '')
        with zipfile.ZipFile(args.data_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        data_root = extract_dir
    else:
        data_root = args.data_path

    if args.mode == 'train':
        dataset = AnimalDataset(data_root, transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        train(model, dataloader, diffusion, device=device)
    elif args.mode == 'sample':
        model.load_state_dict(torch.load('saved_models/denoise_model.pt'))
        img = sample(model, diffusion, device=device)
        save_image(img, 'samples/generated_sample.png')
