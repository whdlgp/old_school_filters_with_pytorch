import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pathlib import Path
from util import load_image, show_comparison
from filter_imple import Sobel

class SimpleConv(nn.Module):
    """
    A simple convolutional model with a single 3x3 filter.
    The goal is to learn a filter that mimics the Sobel operator.
    """
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        G = self.conv(x) # Output shape: [batch, 2, H, W]
        Gx, Gy = G[:, 0, :, :], G[:, 1, :, :]  # Separate the two filters
        
        return torch.sqrt(torch.clamp(Gx**2 + Gy**2, min=1e-6)).unsqueeze(1)

class SobelTrainer:
    """
    Trainer for learning a convolutional filter to approximate the Sobel filter.
    Uses images from the 'test_images' directory.
    """
    def __init__(self, learning_rate, num_epochs):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.model = SimpleConv().to(self.device)
        # Target
        self.sobel = Sobel()
        # Training params
        self.input_size = (512, 512)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        # Input images
        self.image_dir = Path("test_images")
        # Loss history
        self.loss_history = torch.zeros(self.num_epochs, device="cpu")

    def load_dataset(self):
        """
        Loads all JPG images from input directory and generates
        corresponding Sobel-filtered images as training data.
        """
        input_images, target_images = [], []
        for img_path in self.image_dir.glob("*.jpg"):
            img = load_image(str(img_path), self.input_size)
            target = self.sobel.filter(img, method="conv2d") # conv2d method for applying filter

            input_images.append(img.unsqueeze(0))   # [1, H, W]
            target_images.append(target.unsqueeze(0)) # [1, H, W]

        return torch.stack(input_images).to(self.device), torch.stack(target_images).to(self.device)  # Shape: [N, 1, H, W]

    def train(self):
        """
        Trains the model to learn a Sobel-like filter.
        """
        input_tensor, target_tensor = self.load_dataset()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = self.loss_fn(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            # log loss
            self.loss_history[epoch] = loss.item()

            if (epoch+1) % 10 == 0:
                print(f"Epoch [{(epoch+1)}/{self.num_epochs}], Loss: {loss.item():.6f}")

    def evaluate(self):
        """
        Evaluates the trained model and compares its output to the Sobel filter.
        """
        input_tensor, target_tensor = self.load_dataset()

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        target_tensor = target_tensor.cpu()
        output_tensor = output_tensor.cpu()

        for i in range(len(input_tensor)):
            show_comparison(target_tensor[i].squeeze(0), output_tensor[i].squeeze(0), 
                            title1="Sobel Target", title2="Learned Conv2D")

        # Print the learned kernel
        learned_kernel = self.model.conv.weight.data.squeeze().cpu().numpy()
        print("Learned Kernel:\n", learned_kernel)

    def plot_loss(self):
        """
        Plots the loss curve and saves it as an image file.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.loss_history.cpu().numpy(), label="Training Loss", color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")
        plt.show()

if __name__ == "__main__":
    trainer = SobelTrainer(learning_rate=0.001, num_epochs=5000)
    trainer.train()
    trainer.plot_loss()
    trainer.evaluate()
