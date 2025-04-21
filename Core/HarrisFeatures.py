import cv2
import numpy as np

from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from Core.kernelConvolution import sobel, gaussian_filter


def extractHarrisFeatures(img, k=0.04, window_size=7, dist_threshold=50):
    image = img.copy()
    gray = rgb_to_grayscale(image).astype(np.float32)

    # edges = cv2.Canny(rgb_to_grayscale(image),150,200)

    # edges = gaussian_filter(image, 5, 1)

    gradienX, gradienY, _, _ = sobel(gray, 5)


    Ixx = gradienX ** 2
    Iyy = gradienY ** 2
    Ixy = gradienX * gradienY

    # Ixx = gaussian_filter(Ixx, 5, .8)
    # Iyy = gaussian_filter(Iyy, 3, .8)
    # Ixy = gaussian_filter(Ixy, 3, 1)
    #
    # cv2.imshow("Ixx", Ixx)
    # cv2.imshow("Iyy", Iyy)
    # cv2.imshow("Ixy", Ixy)

    # # Efficiently stack Ixx, Ixy, Iyy into the correct (H, W, 2, 2) format
    # harrisMat = np.stack((np.stack((Ixx, Ixy), axis=-1),
    #                       np.stack((Ixy, Iyy), axis=-1)), axis=-2)
    #
    # # Compute determinant of the Harris matrix
    # det_M = (Ixx * Iyy) - (Ixy ** 2)  # Faster than np.linalg.det()
    #
    # # Compute trace of the Harris matrix
    # trace_M = Ixx + Iyy  # Faster than np.trace()
    #
    # # Compute Harris response R
    # R = det_M - k * (trace_M ** 2)
    # Pad all matrices to handle window borders
    # pad = window_size // 2
    # Ixx_padded = np.pad(Ixx, pad, mode='constant')
    # Iyy_padded = np.pad(Iyy, pad, mode='constant')
    # Ixy_padded = np.pad(Ixy, pad, mode='constant')

    height, width = gray.shape
    R = np.zeros((height, width))

    # Sliding window to compute local Harris response
    for i in range(height):
        for j in range(width):
            # Extract window for each component
            window_Ixx = Ixx[i:i + window_size, j:j + window_size]
            window_Iyy = Iyy[i:i + window_size, j:j + window_size]
            window_Ixy = Ixy[i:i + window_size, j:j + window_size]

            # Sum of values in the window
            Sxx = np.sum(window_Ixx)
            Syy = np.sum(window_Iyy)
            Sxy = np.sum(window_Ixy)

            # Harris matrix H
            det_H = (Sxx * Syy) - (Sxy ** 2)
            trace_H = Sxx + Syy

            # Harris response R
            R[i, j] = det_H - k * (trace_H ** 2)


    # Normalize R
    R_min, R_max = np.min(R), np.max(R)
    if R_max != R_min:
        R_norm = ((R - R_min) / (R_max - R_min)) * 255
    else:
        R_norm = np.zeros_like(R)


    # Compute threshold value
    threshold_value = np.percentile(R_norm[R_norm > 0], 100) if np.any(R_norm > 0) else 0
    # threshold_value = np.mean(R[R > 0]) + 2 * np.std(R[R > 0])

    # Apply thresholding first
    corners = (R > threshold_value).astype(np.uint8) * 255

    # Apply non-max suppression AFTER thresholding

    corners = distance_based_nms_fast(corners, R, dist_threshold)
    corners = non_max_suppression(corners, 10)


    corner_coords = np.where(corners > 0)
    corner_coords = list(zip(corner_coords[1], corner_coords[0]))  # (x, y) format

    # Create image with corners marked
    marked_image = image.copy()
    for (x, y) in corner_coords:
        cv2.circle(marked_image, (x, y), 2, (0, 0, 255), -1)  # Red circles


    # Create a blue visualization map (R_norm mapped to blue channel)
    blue_map = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    blue_map[:, :, 0] = R_norm  # Map response to the blue channel correctly

    blue_map_thresholded = np.zeros_like(blue_map)  # Initialize empty blue map
    blue_map_thresholded = R_norm * (corners > 0)

    BRmin, BRmax = np.min(blue_map_thresholded), np.max(blue_map_thresholded)
    if BRmax != BRmin:
        blue_map_thresholded = ((blue_map_thresholded - BRmin) / (BRmax - BRmin)) * 255
    else:
        blue_map_thresholded = np.zeros_like(R)

    blue_map_thresholded = blue_map_thresholded.astype(np.uint8)


    blue_map_thresholded_final = np.zeros_like(blue_map)
    blue_map_thresholded_final[:, :, 0] = blue_map_thresholded

    return corners, blue_map,blue_map_thresholded_final, marked_image


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

def distance_based_nms_fast(corners, response_map, dist_thresh=60):
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


def test_harris_features():
    import os
    import cv2

    # Load test image (adjust path as needed)
    img_path = '../images/Chess.png'  # Replace with your image path
    if not os.path.exists(img_path):
        print(f"Image not found at: {img_path}")
        return

    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))  # Resize for consistency

    # Run the Harris feature extraction
    corners, blue_map, blue_map_thresholded_final, marked_image = extractHarrisFeatures(
        img, k=0.04, window_size=3, threshold=0.005
    )

    # Show outputs using OpenCV windows
    cv2.imshow("Original Image", img)
    cv2.imshow("Corners Marked (Red)", marked_image)
    cv2.imshow("Harris Response Map (Blue)", blue_map)
    cv2.imshow("Thresholded Corners (Blue)", blue_map_thresholded_final)

    print(f"Detected {np.sum(corners > 0)} corners.")
    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Entry point for the script
if __name__ == "__main__":
    test_harris_features()
