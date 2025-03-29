import cv2
import numpy as np

from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from Core.kernelConvolution import sobel, gaussian_filter


def extractHarrisFeatures(img, k=0.02, threshold=0.005):
    image = img.copy()

    # edges = cv2.Canny(rgb_to_grayscale(image),150,200)

    # edges = gaussian_filter(image, 5, 1)

    gradienX, gradienY, _, _ = sobel(rgb_to_grayscale(image), 5)


    Ixx = gradienX ** 2
    Iyy = gradienY ** 2
    Ixy = gradienX * gradienY

    Ixx = gaussian_filter(Ixx, 5, .8)
    Iyy = gaussian_filter(Iyy, 3, .8)
    Ixy = gaussian_filter(Ixy, 3, 1)

    cv2.imshow("Ixx", Ixx)
    cv2.imshow("Iyy", Iyy)
    cv2.imshow("Ixy", Ixy)

    # Efficiently stack Ixx, Ixy, Iyy into the correct (H, W, 2, 2) format
    harrisMat = np.stack((np.stack((Ixx, Ixy), axis=-1),
                          np.stack((Ixy, Iyy), axis=-1)), axis=-2)

    # Compute determinant of the Harris matrix
    det_M = (Ixx * Iyy) - (Ixy ** 2)  # Faster than np.linalg.det()

    # Compute trace of the Harris matrix
    trace_M = Ixx + Iyy  # Faster than np.trace()

    # Compute Harris response R
    R = det_M - k * (trace_M ** 2)


    # Normalize R
    R_min, R_max = np.min(R), np.max(R)
    if R_max != R_min:
        R_norm = ((R - R_min) / (R_max - R_min)) * 255
    else:
        R_norm = np.zeros_like(R)

    R_norm = R_norm.astype(np.uint8)

    # Compute threshold value
    threshold_value = np.percentile(R_norm[R_norm > 0], 99.8) if np.any(R_norm > 0) else 0
    # threshold_value = np.mean(R[R > 0]) + 2 * np.std(R[R > 0])

    # Apply thresholding first
    corners = (R > threshold_value).astype(np.uint8) * 255

    # Apply non-max suppression AFTER thresholding
    # corners = non_max_suppression(corners, 5)

    corners = distance_based_nms_fast(corners, R, 30)


    corner_coords = np.where(corners > 0)
    corner_coords = list(zip(corner_coords[1], corner_coords[0]))  # (x, y) format

    # Create image with corners marked
    marked_image = image.copy()
    for (x, y) in corner_coords:
        cv2.circle(marked_image, (x, y), 1, (0, 0, 255), -1)  # Red circles


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

def distance_based_nms_fast(corners, response_map, dist_thresh=1000):
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

import cv2
import numpy as np

if __name__ == "__main__":
    # Read the image
    img = cv2.imread("../images/Notre Dam 1.png")
    if img is None:
        raise FileNotFoundError("Could not load image at path")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute Harris response using OpenCV (ensure input is float32)
    harris_response = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)

    # Normalize Harris response for visualization
    harris_normalized = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Overlay Harris corners on the original image (Red color)
    img_harris_corners = img.copy()
    threshold = 0.02 * harris_response.max()  # Adjust threshold for visibility
    img_harris_corners[harris_response > threshold] = [0, 0, 255]  # Mark corners in red

    # Extract your Harris Features
    corners, blue_map, thresholdBlue, image = extractHarrisFeatures(img)

    # 🔹 Ensure all images are uint8 and 3-channel (to fix cv2.vconcat issue)
    def prepare_image(img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:  # Convert grayscale to 3-channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # Convert all images
    image_color = prepare_image(image)
    corners_color = prepare_image(corners)
    blue_map_color = prepare_image(blue_map)
    thresholdBlue_color = prepare_image(thresholdBlue)
    harris_corners_color = prepare_image(img_harris_corners)

    # Resize all images to match dimensions
    max_height = max(image_color.shape[0], corners_color.shape[0], blue_map_color.shape[0], thresholdBlue_color.shape[0], harris_corners_color.shape[0])
    max_width = max(image_color.shape[1], corners_color.shape[1], blue_map_color.shape[1], thresholdBlue_color.shape[1], harris_corners_color.shape[1])

    def resize_to_match(img, height, width):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    image_resized = resize_to_match(image_color, max_height, max_width)
    corners_resized = resize_to_match(corners_color, max_height, max_width)
    blue_map_resized = resize_to_match(blue_map_color, max_height, max_width)
    thresholdBlue_resized = resize_to_match(thresholdBlue_color, max_height, max_width)
    harris_corners_resized = resize_to_match(harris_corners_color, max_height, max_width)

    # 🔹 Ensure all images have the same dtype and shape before concatenation
    print("Dtype Check:", image_resized.dtype, corners_resized.dtype, blue_map_resized.dtype, thresholdBlue_resized.dtype, harris_corners_resized.dtype)
    print("Shape Check:", image_resized.shape, corners_resized.shape, blue_map_resized.shape, thresholdBlue_resized.shape, harris_corners_resized.shape)

    # Arrange images in a grid
    topRow = cv2.hconcat([image_resized, corners_resized, harris_corners_resized])
    bottomRow = cv2.hconcat([blue_map_resized, thresholdBlue_resized])

    # Stack all images together

    # Display the result
    cv2.imshow("Harris Feaginal | Your Corners | CV2 Harris | Response Map | Thresholded Response", topRow)
    # cv2.imshow("Harris Features - Original | Your Corners | CV2 Harris | Response Map | Thresholded Response", bottomRow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
