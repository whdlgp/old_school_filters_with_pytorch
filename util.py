import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

# Get execution time
def measure_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

# Load image and convert to Grayscale
def load_image(img_path):
    img = Image.open(img_path).convert("L")  # Grayscale
    img_tensor = TF.to_tensor(img)  # [C, H, W]
    return img_tensor.squeeze(0)  # [H, W]

# Show loaded image
def show_image(img):
    plt.imshow(img, cmap="gray")
    plt.show()

# Show 2 images for comparison
def show_comparison(img1, img2, title1="First", title2="Second"):
    # concat images
    combined_img = torch.cat((img1, img2), dim=1)

    # show
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_img, cmap="gray")
    
    # Title text
    img_width = img1.shape[1]
    plt.text(img_width // 2, -10, title1, fontsize=12, ha="center", color="red", fontweight="bold")
    plt.text(img_width + img_width // 2, -10, title2, fontsize=12, ha="center", color="blue", fontweight="bold")

    plt.axis("off")
    plt.show()