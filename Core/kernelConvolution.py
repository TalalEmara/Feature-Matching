import math

import numpy as np

from Core.imageMode import rgb_to_grayscale


def custom_convolution(image, kernel):
    if len(image.shape) == 3:
        image = rgb_to_grayscale(image)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='symmetric')
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output


def custom_normalize(image, out_min=0, out_max=255):
    """
    Normalize image values to a specific range without using OpenCV.
    """
    # Handle empty images
    if image is None or image.size == 0:
        return None

    # Get min and max values
    img_min = np.min(image)
    img_max = np.max(image)

    # Avoid division by zero
    if img_max == img_min:
        return np.ones_like(image) * out_min

    # Normalize to specified range
    normalized = (image - img_min) * (out_max - out_min) / (img_max - img_min) + out_min
    return normalized.astype(np.uint8)

def generate_sobel_kernels(kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    if kernel_size == 3:
        Gx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])

        Gy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    else:
        pascal_row = np.array([math.comb(kernel_size - 1, i) for i in range(kernel_size)])
        deriv_kernel = np.array([-i for i in range(-(kernel_size // 2), kernel_size // 2 + 1)])
        Gx = np.outer(deriv_kernel, pascal_row)
        Gy = np.outer(pascal_row, deriv_kernel)

    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))

    return Gx, Gy

def sobel(image, kernel_size=3):
    Gx, Gy = generate_sobel_kernels(kernel_size)
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = custom_normalize(gradient_magnitude, 0, 255)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel_size=5, sigma=1.5):
    kernel = gaussian_kernel(kernel_size, sigma)
    return custom_convolution(image, kernel).astype(np.uint8)
