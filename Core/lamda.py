import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


# def convolve2d(image, kernel):
#     k_h, k_w = kernel.shape
#     pad_h, pad_w = k_h // 2, k_w // 2
#     padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
#     output = np.zeros_like(image, dtype=float)
#
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             region = padded[i:i + k_h, j:j + k_w]
#             output[i, j] = np.sum(region * kernel)
#     return output

def convolve2d_fast(image, kernel):
    """Fast convolution using im2col strategy."""
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    H, W = image.shape
    output = np.zeros((H, W), dtype=float)

    for i in range(k_h):
        for j in range(k_w):
            output += kernel[i, j] * padded[i:i + H, j:j + W]

    return output



def gaussian_smooth(image, kernel_size=5, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d_fast(image, kernel)


def compute_gradients(image):
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=float)

    Ix = convolve2d_fast(image, Kx)
    Iy = convolve2d_fast(image, Ky)
    return Ix, Iy


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


def min_eigenvalue_2x2(A, B, C):
    trace = A + C
    det_sqrt = np.sqrt(((A - C) / 2) ** 2 + B ** 2)
    return (trace / 2) - det_sqrt


def lambda_detector(image, threshold_ratio=0.01, kernel_size=5, sigma=1.0):
    start_time = time.time()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(float)

    gray = gray.astype(float)
    gray_blurred = gaussian_smooth(gray, kernel_size, sigma)
    Ix, Iy = compute_gradients(gray_blurred)

    Ixx = gaussian_smooth(Ix ** 2, kernel_size, sigma)
    Iyy = gaussian_smooth(Iy ** 2, kernel_size, sigma)
    Ixy = gaussian_smooth(Ix * Iy, kernel_size, sigma)

    height, width = gray.shape
    lambda_min = np.zeros_like(gray, dtype=float)

    for y in range(height):
        for x in range(width):
            # M = np.array([[Ixx[y, x], Ixy[y, x]],
            #               [Ixy[y, x], Iyy[y, x]]])
            # eigenvalues = np.linalg.eigvalsh(M)
            # lambda_min[y, x] = min(eigenvalues)
            A = Ixx[y, x]
            B = Ixy[y, x]
            C = Iyy[y, x]
            lambda_min[y, x] = min_eigenvalue_2x2(A, B, C)

    max_val = np.max(lambda_min)
    threshold = threshold_ratio * max_val
    keypoints = non_maximum_suppression(lambda_min, threshold)
    output_img = draw_keypoints(image, keypoints)

    end_time = time.time()
    print(f"[λ Scratch] Time: {end_time - start_time:.4f}s | Keypoints: {len(keypoints)}")
    return keypoints, lambda_min ,output_img


def opencv_lambda_detector(image, threshold_ratio=0.01, block_size=3, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lambda_min = cv2.cornerMinEigenVal(gray, blockSize=block_size, ksize=ksize)
    max_val = np.max(lambda_min)
    threshold = threshold_ratio * max_val

    keypoints = non_maximum_suppression(lambda_min, threshold)
    print(f"[λ OpenCV] Keypoints: {len(keypoints)}")
    return keypoints, lambda_min


def visualize_keypoints_comparison(image, keypoints1, keypoints2, title1='Scratch λ', title2='OpenCV λ'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, keypoints, title in zip(axes, [keypoints1, keypoints2], [title1, title2]):
        ax.imshow(image, cmap='gray')
        if keypoints:
            xs, ys = zip(*keypoints)
            ax.scatter(xs, ys, c='red', s=5)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def draw_keypoints(image, keypoints, color=(0, 0, 255), radius=2, thickness=-1):
    """Returns a copy of the image with keypoints drawn."""
    output = image.copy()
    for x, y in keypoints:
        cv2.circle(output, (x, y), radius, color, thickness)
    return output


# Main
if __name__ == "__main__":
    image = cv2.imread("../images/Feature matching/Notre Dam 1.png")  # Change this to your image path

    if image is None:
        print("Image not found.")
    else:
        # Adjustable parameters
        kernel_size = 5   # Must be odd
        sigma = 1.0
        threshold_ratio = 0.01

        # keypoints_lambda, _ = lambda_detector(image, threshold_ratio=threshold_ratio,
        #                                       kernel_size=kernel_size, sigma=sigma)
        # keypoints_opencv, _ = opencv_lambda_detector(image, threshold_ratio=threshold_ratio)
        # visualize_keypoints_comparison(image, keypoints_lambda, keypoints_opencv)
        keypoints_lambda, _, lambda_output_img = lambda_detector(
                    image, threshold_ratio=threshold_ratio,
                    kernel_size=kernel_size, sigma=sigma
                )

        cv2.imshow("Lambda Scratch Detector", lambda_output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()