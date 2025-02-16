import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import cv2
import numpy as np

# Get execution time
def measure_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

# Load image and convert to Grayscale
# resize : (width, height)
def load_image(img_path, resize=None):
    img = Image.open(img_path).convert("L")  # Grayscale
    if resize:
        img = img.resize(resize, Image.BILINEAR)
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

# Displays all filter results in a single grid using OpenCV.
def save_comparison_grid(results, output_filename):
    combined_rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_colors = [(0, 0, 255), (255, 0, 0)]
    bg_color = (0, 0, 0)

    for images in results:
        image_tensors, titles = images[:-2], images[-2:]
        images_np = [(img.numpy() * 255).astype(np.uint8) for img in image_tensors]
        combined_row = np.hstack(images_np)

        if len(combined_row.shape) == 2:
            combined_row = cv2.cvtColor(combined_row, cv2.COLOR_GRAY2BGR)

        row_height, row_width = combined_row.shape[:2]
        text_y_position = max(30, row_height // 15)
        step = row_width // len(titles)

        for i, title in enumerate(titles):
            x_position = i * step + 10
            color = text_colors[i % 2]
            max_text_width = step - 20

            font_scale = max(0.7, min(2.5, row_height / 450))
            font_thickness = max(2, int(row_height / 250))
            while True:
                (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, font_thickness)
                if text_width <= max_text_width or font_scale < 0.5:
                    break
                font_scale -= 0.1

            box_coords = ((x_position - 5, text_y_position - text_height - 5),
                          (x_position + text_width + 5, text_y_position + 5))

            cv2.rectangle(combined_row, box_coords[0], box_coords[1], bg_color, -1)
            cv2.putText(combined_row, title, (x_position, text_y_position),
                        font, font_scale, color, font_thickness, cv2.LINE_AA)

        combined_rows.append(combined_row)

    final_output = np.vstack(combined_rows)
    cv2.imwrite(f"{output_filename}.jpg", final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])