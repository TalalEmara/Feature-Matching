import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def compute_gradients(image):
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=float)

    Ix = convolve2d(image, Kx)
    Iy = convolve2d(image, Ky)
    return Ix, Iy


def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    return output


def non_maximum_suppression(λ_map, threshold, window_size=5):
    keypoints = []
    half_win = window_size // 2
    h, w = λ_map.shape

    for y in range(half_win, h - half_win):
        for x in range(half_win, w - half_win):
            current = λ_map[y, x]
            if current > threshold:
                local_patch = λ_map[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
                if current == np.max(local_patch):
                    keypoints.append((x, y))
    return keypoints


def lambda_detector(image, threshold=1e6):
    start_time = time.time()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(float)

    # Step 1: Gradients
    gray_blurred = gaussian_filter(gray, sigma=1)
    # Ix, Iy = compute_gradients(gray_blurred)

    Ix, Iy = compute_gradients(gray.astype(float))

    # Step 2: Structure tensor elements
    Ixx = gaussian_filter(Ix ** 2, sigma=1)
    Iyy = gaussian_filter(Iy ** 2, sigma=1)
    Ixy = gaussian_filter(Ix * Iy, sigma=1)

    # Step 3: Compute min eigenvalue λ_min
    height, width = gray.shape
    lambda_min = np.zeros_like(gray, dtype=float)

    for y in range(height):
        for x in range(width):
            M = np.array([[Ixx[y, x], Ixy[y, x]],
                          [Ixy[y, x], Iyy[y, x]]])
            eigenvalues = np.linalg.eigvalsh(M)
            lambda_min[y, x] = min(eigenvalues)

    # Apply Non-Maximum Suppression
    keypoints = non_maximum_suppression(lambda_min, threshold)

    end_time = time.time()
    print(f"Computation Time: {end_time - start_time:.4f} seconds")
    print(f"Total Keypoints Detected after NMS: {len(keypoints)}")

    return keypoints, lambda_min


def visualize_keypoints(image, keypoints):
    if keypoints:
        plt.imshow(image, cmap='gray')
        xs, ys = zip(*keypoints)
        plt.scatter(xs, ys, c='r', s=5)
        plt.title("λ-based Keypoints after NMS")
        plt.axis('off')
        plt.show()
    else:
        print("No keypoints to display.")


# Example usage
if __name__ == "__main__":
    image = cv2.imread("../images/chessboard.jpg")  # Read the image

    if image is None:
        print("Error: Image not found or cannot be read. Please check the file path.")
    else:
        keypoints, λ_map = lambda_detector(image, threshold=2e4)
        visualize_keypoints(image, keypoints)