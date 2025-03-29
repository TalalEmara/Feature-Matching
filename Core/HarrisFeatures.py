import cv2
import numpy as np

from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from Core.kernelConvolution import sobel, gaussian_filter


def extractHarrisFeatures(img, k=0.04, threshold=0.005):
    image = img.copy()

    # edges = cv2.Canny(rgb_to_grayscale(image),150,200)

    # edges = gaussian_filter(image, 5, 1)

    gradienX, gradienY, _, _ = sobel(rgb_to_grayscale(image), 3)


    Ixx = gradienX ** 2
    Iyy = gradienY ** 2
    Ixy = gradienX * gradienY

    Ixx = gaussian_filter(Ixx, 5, .5)
    Iyy = gaussian_filter(Iyy, 5, .5)
    Ixy = gaussian_filter(Ixy, 5, .5)

    # Efficiently stack Ixx, Ixy, Iyy into the correct (H, W, 2, 2) format
    harrisMat = np.stack((np.stack((Ixx, Ixy), axis=-1),
                          np.stack((Ixy, Iyy), axis=-1)), axis=-2)


    # Compute determinant and trace in one go
    det_M = np.linalg.det(harrisMat)  # Shape (H, W)
    trace_M = np.trace(harrisMat, axis1=-2, axis2=-1)  # Shape (H, W)

    # Compute Harris response R
    R = det_M - k * (trace_M ** 2)  # Shape (H, W)

    # Normalize R
    R_min, R_max = np.min(R), np.max(R)
    if R_max != R_min:
        R_norm = ((R - R_min) / (R_max - R_min)) * 255
    else:
        R_norm = np.zeros_like(R)

    R_norm = R_norm.astype(np.uint8)

    # Compute threshold value
    threshold_value = np.percentile(R_norm[R_norm > 0], 98) if np.any(R_norm > 0) else 0
    # threshold_value = np.mean(R[R > 0]) + 2 * np.std(R[R > 0])

    # Apply thresholding first
    corners = (R_norm > threshold_value).astype(np.uint8) * 255

    # Apply non-max suppression AFTER thresholding
    # corners = non_max_suppression(corners, 5)
    corners = distance_based_nms_fast(corners, R_norm, 10)

    # Create a blue visualization map (R_norm mapped to blue channel)
    blue_map = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    blue_map[:, :, 0] = R_norm  # Map response to the blue channel correctly

    return corners, blue_map, image


def non_max_suppression(subject, window_size=3):

    H, W = subject.shape
    half_w = window_size // 2
    suppressed = np.zeros_like(subject)

    # Iterate through each pixel, avoiding edges
    for y in range(half_w, H - half_w):
        for x in range(half_w, W - half_w):
            window = subject[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]  # Extract local region
            max_value = np.max(window)  # Get max in window

            if subject[y, x] == max_value:  # Keep only local maxima
                suppressed[y, x] = subject[y, x]

    return suppressed

def distance_based_nms_fast(corners, response_map, dist_thresh=10):
        y, x = np.where(corners > 0)
        if len(x) == 0:
            return np.zeros_like(corners)

        responses = response_map[y, x]
        sorted_idx = np.argsort(-responses)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # Pre-compute squared distance threshold
        sq_dist_thresh = dist_thresh ** 2

        kept_x = []
        kept_y = []
        filtered = np.zeros_like(corners)

        for i in range(len(x_sorted)):
            cx = x_sorted[i]
            cy = y_sorted[i]

            # Vectorized distance check
            if len(kept_x) > 0:
                dists_sq = (np.array(kept_x) - cx) ** 2 + (np.array(kept_y) - cy) ** 2
                if np.any(dists_sq <= sq_dist_thresh):
                    continue

            kept_x.append(cx)
            kept_y.append(cy)
            filtered[cy, cx] = 255

        return filtered


if __name__ == "__main__":
    # Read and process the image
    img = cv2.imread("../images/Chess.png")
    if img is None:
        raise FileNotFoundError("Could not load image at path: ../images/Chess.png")

    corners, blue_map, image = extractHarrisFeatures(img)

    # Ensure all images are 3-channel (convert grayscale if needed)
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    if len(corners.shape) == 2:
        corners_color = cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)
    else:
        corners_color = corners.copy()

    # Blue_map should already be 3-channel from the extractHarrisFeatures function

    # Find the maximum dimensions
    max_height = max(image_color.shape[0], corners_color.shape[0], blue_map.shape[0])
    max_width = max(image_color.shape[1], corners_color.shape[1], blue_map.shape[1])


    # Resize all images to the same dimensions
    def resize_to_match(img, height, width):
        return cv2.resize(img, (width, height))


    image_resized = resize_to_match(image_color, max_height, max_width)
    corners_resized = resize_to_match(corners_color, max_height, max_width)
    blue_map_resized = resize_to_match(blue_map, max_height, max_width)

    # Arrange in 2x2 format
    top_row = cv2.vconcat([image_resized])
    bottom_row = cv2.vconcat([corners_resized, blue_map_resized])

    # Ensure both rows have the same width
    if top_row.shape[1] != bottom_row.shape[1]:
        min_width = min(top_row.shape[1], bottom_row.shape[1])
        top_row = cv2.resize(top_row, (min_width, top_row.shape[0]))
        bottom_row = cv2.resize(bottom_row, (min_width, bottom_row.shape[0]))

    # stacked_image = cv2.Hconcat([top_row, bottom_row])
    stacked_image = cv2.hconcat([image_resized, corners_resized, blue_map_resized])

    # Display the result
    cv2.imshow("Harris Features - Original | Corners | Response Map", stacked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()