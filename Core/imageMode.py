import numpy as np


def rgb_to_grayscale(image):
    """Convert an RGB image to grayscale using luminosity method."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b  # Standard grayscale conversion
    return grayscale.astype(np.uint8)

def rgb_to_yuv(image):
    """Convert an RGB image to YUV color space using NumPy."""
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]])
    return np.dot(image, matrix.T)

def yuv_to_rgb(yuv):
    """Convert a YUV image back to RGB using NumPy."""
    matrix = np.array([[1.0, 0.0, 1.13983],
                       [1.0, -0.39465, -0.58060],
                       [1.0, 2.03211, 0.0]])
    return np.dot(yuv, matrix.T)


def normalize_image(image):
    """Apply zero mean and unit variance normalization."""
    mean_val = np.mean(image)  # Compute mean intensity
    std_val = np.std(image)  # Compute standard deviation

    if std_val == 0:  # Avoid division by zero
        std_val = 1

    normalized_image = (image - mean_val) / std_val  # Zero mean normalization

    normalized_image = ((normalized_image - normalized_image.min()) /
                        (normalized_image.max() - normalized_image.min())) * 255

    return normalized_image.astype(np.uint8)

def prepare_for_display(image):
    """Scale a zero-mean image into [0,255] while keeping contrast."""
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
