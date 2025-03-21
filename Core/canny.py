# Canny filter implementation
from Core.kernelConvolution import gaussian_filter, sobel
import numpy as np

def canny(image, low_threshold=100, high_threshold=150):
    """
    image: rgb image
    low_threshold: low threshold for edge detection
    high_threshold: high threshold for edge detection
    return: edges detected using canny filter (2d array)
    """
    # image = gaussian_filter(image, 3, 2)
    image = gaussian_filter(image, 4, 1)
    _, _, magnitude, direction = sobel(image)
    suppressed = non_maximum(magnitude, direction)
    thresholded = double_threshold(suppressed, low_threshold, high_threshold)
    final_edges = hysteresis(thresholded)
    return final_edges.astype(np.uint8)

def non_maximum(magnitude, direction):
    """
    magnitude: magnitude of the gradient
    direction: direction of the gradient (radians)
    return: non-maximum suppressed image (2d array)
    """
    quantized_directions = quantization(direction)
    suppressed = np.zeros_like(magnitude)
    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            current_magnitude = magnitude[y, x]
            current_direction = quantized_directions[y, x]

            if current_direction == 0:  # Horizontal
                neighbors = [magnitude[y, x-1], magnitude[y, x+1]]
            elif current_direction == 45:  # Diagonal
                neighbors = [magnitude[y-1, x+1], magnitude[y+1, x-1]]
            elif current_direction == 90:  # Vertical
                neighbors = [magnitude[y-1, x], magnitude[y+1, x]]
            elif current_direction == 135:  # Anti-diagonal
                neighbors = [magnitude[y-1, x-1], magnitude[y+1, x+1]]
            if current_magnitude >= max(neighbors):
                suppressed[y, x] = current_magnitude
    return suppressed

def quantization(direction):
    """
    direction: direction of the gradient (radians)
    return: quantized directions (0, 45, 90, 135) degrees
    """
    direction = np.rad2deg(direction)
    quantized_directions = np.zeros_like(direction, dtype=int)
    quantized_directions[(direction >= 0) & (direction < 22.5)] = 0
    quantized_directions[(direction >= 22.5) & (direction < 67.5)] = 45
    quantized_directions[(direction >= 67.5) & (direction < 112.5)] = 90
    quantized_directions[(direction >= 112.5) & (direction < 157.5)] = 135
    quantized_directions[(direction >= 157.5) & (direction <= 180)] = 0
    return quantized_directions

def double_threshold(magnitude, low_threshold, high_threshold):
    """
    magnitude: magnitude of the gradient
    low_threshold: low threshold for edge detection
    high_threshold: high threshold for edge detection   
    detectes edges by Tl and Th
    return: edges after applying double thresholding 
    """
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    strong_edges = (magnitude >= high_threshold)
    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)
    edges[strong_edges] = 255
    edges[weak_edges] = 100
    return edges

def hysteresis(edges):
    """
    edges: edges after applying double thresholding
    iterates over the weak edges and connects them to strong edges
    return: edges after applying hysterisis thresholding
    """
    final_edges = np.zeros_like(edges)
    strong_edges = (edges == 255)
    weak_edges = (edges == 100)
    final_edges[strong_edges] = 255
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            if weak_edges[y, x]:
                for dy, dx in neighbors:
                    if final_edges[y + dy, x + dx] == 255:
                        final_edges[y, x] = 255
                        break
    return final_edges

# --------------Test the function---------------
def test_edge_filters():
    import cv2
    import matplotlib.pyplot as plt

    def cannyb(image, low_threshold=50, high_threshold=150):
        image = gaussian_filter(image, 3, 2)
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges.astype(np.uint8)

    image_path = "CV/Image-Editor-Computer-Vision/images/bobama.jpg"
    imageRGB = cv2.imread(image_path)
    imageRGB = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2RGB)

    cannyimg = canny(imageRGB)
    cannybimg = cannyb(imageRGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(imageRGB)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Canny")
    plt.imshow(cannyimg, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Canny (OpenCV)")
    plt.imshow(cannybimg, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# test_edge_filters()
