import torch
import torch.nn.functional as F

# Add padding to image
def pad_image(img, pad_size=1):
    H, W = img.shape
    padded = torch.zeros((H + 2*pad_size, W + 2*pad_size), dtype=img.dtype)
    padded[pad_size:-pad_size, pad_size:-pad_size] = img
    return padded

# Apply filter convolution with for loop
def apply_filter_manual(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    img_padded = pad_image(img, pad_size=kH//2)

    output = torch.zeros((H, W))

    for i in range(H):
        for j in range(W):
            region = img_padded[i:i+kH, j:j+kW]
            output[i, j] = torch.sum(region * kernel)
    return output

# Apply filter convolution with torchvision conv2d
def apply_filter_conv2d(img, kernel):
    padding = (kernel.shape[0] - 1) // 2
    img = img.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    output = F.conv2d(img, kernel, padding=padding)
    return output.squeeze(0).squeeze(0)

# Sobel Filter implementation
class Sobel:
    # Create Sobel filter
    def __init__(self):
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32)

        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=torch.float32)
        
    # Sobel detection
    def filter(self, img, method = "manual"):
        if method == "manual": # conv with for loop
            Gx = apply_filter_manual(img, self.sobel_x)
            Gy = apply_filter_manual(img, self.sobel_y)
        elif method == "conv2d": # conv with torchvision conv2d
            Gx = apply_filter_conv2d(img, self.sobel_x)
            Gy = apply_filter_conv2d(img, self.sobel_y)
        G = torch.sqrt(Gx**2 + Gy**2)
        return G
    
# Prewitt Filter implementation
class Prewitt:
    # Create Prewitt filter
    def __init__(self):
        self.prewitt_x = torch.tensor([[-1, 0, 1], 
                                       [-1, 0, 1], 
                                       [-1, 0, 1]], dtype=torch.float32)

        self.prewitt_y = torch.tensor([[-1, -1, -1], 
                                       [0, 0, 0], 
                                       [1, 1, 1]], dtype=torch.float32)
        
    # Prewitt detection
    def filter(self, img, method="manual"):
        if method == "manual":  # conv with for loop
            Gx = apply_filter_manual(img, self.prewitt_x)
            Gy = apply_filter_manual(img, self.prewitt_y)
        elif method == "conv2d":  # conv with torchvision conv2d
            Gx = apply_filter_conv2d(img, self.prewitt_x)
            Gy = apply_filter_conv2d(img, self.prewitt_y)
        G = torch.sqrt(Gx**2 + Gy**2)
        return G

# Laplacian Filter implementation
class Laplacian:
    # Create Laplacian filter
    def __init__(self):
        self.laplacian = torch.tensor([[0,  1,  0], 
                                       [1, -4,  1], 
                                       [0,  1,  0]], dtype=torch.float32)

    # Laplacian detection
    def filter(self, img, method="manual"):
        if method == "manual":
            return apply_filter_manual(img, self.laplacian)
        elif method == "conv2d":
            return apply_filter_conv2d(img, self.laplacian)
        
# Gaussian Blur Filter implementation
class GaussianBlur:
    # Create Gaussian filter
    def __init__(self):
        self.gaussian_blur = torch.tensor([[1, 2, 1], 
                                           [2, 4, 2], 
                                           [1, 2, 1]], dtype=torch.float32) / 16

    # Apply Gaussian Blur
    def filter(self, img, method="manual"):
        if method == "manual":
            return apply_filter_manual(img, self.gaussian_blur)
        elif method == "conv2d":
            return apply_filter_conv2d(img, self.gaussian_blur)

# Laplacian of Gaussian (LoG) Filter implementation
class LaplacianOfGaussian:
    def __init__(self):
        # LoG 필터 (5x5 커널)
        self.log_kernel = torch.tensor([
            [0,  0, -1,  0,  0],
            [0, -1, -2, -1,  0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1,  0],
            [0,  0, -1,  0,  0]
        ], dtype=torch.float32)

    # Apply LoG filter
    def filter(self, img, method="manual"):
        if method == "manual":
            return apply_filter_manual(img, self.log_kernel)
        elif method == "conv2d":
            return apply_filter_conv2d(img, self.log_kernel)