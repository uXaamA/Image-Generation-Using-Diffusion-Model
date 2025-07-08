# Import necessary libraries
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from M_Usama_MSDS24045 import DenoiseModel, Diffusion, sample

# Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'saved_models/denoise_model.pt'

# Loading  model
model = DenoiseModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Create diffusion process instance
diffusion = Diffusion()

# Generate image from pure noise
print("Sampling a single image from noise...")
img = sample(model, diffusion, shape=(1, 3, 64, 64), device=device)
img = torch.clamp(img, 0., 1.)

# Save image
save_image(img, 'samples/generated_sample_demo.png')

# Show image
plt.figure(figsize=(4, 4))
plt.axis('off')
plt.title("Generated Image from Noise")
plt.imshow(make_grid(img.cpu(), nrow=1).permute(1, 2, 0))
plt.show()
