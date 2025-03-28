import cv2
import numpy as np

def harris_corner_detector(image, k=0.04, threshold=0.01):
    """    
    Parameters:
        image: Grayscale input image of shape
        k: Harris detector free parameter
        threshold: Threshold to select strong corners
    
    Returns:
        numpy Array of detected keypoints of shape (N, 2) where N is the number of detected corners
    """
     
    gray = np.float32(image)
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=k)
    harris_response = cv2.dilate(harris_response, None)
    keypoints = np.argwhere(harris_response > threshold * harris_response.max())
    return keypoints[:, [1, 0]]

def sift_detector(image):
    """    
    Parameters:
        image: Grayscale input image of shape
    
    Returns:
        - numpy Array of keypoints of shape (N, 2) 
        - numpy Array of descriptors of shape (N, 128)
    """
     
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    kp_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return kp_array, descriptors

def match_features(des1, des2, method="ssd", top_n=50, ratio_threshold=0.75):
    """    
    Parameters:
        des1 (numpy.ndarray): Feature descriptors from the first image of shape (N1, 128)
        des2 (numpy.ndarray): Feature descriptors from the second image of shape (N2, 128)

    Returns:
        list: List of matched keypoint index pairs [(idx1, idx2)].
    """
    matches = []
    scores = []

    for i, d1 in enumerate(des1):
        best_match = None
        second_best_match = None
        best_score = float("inf") if method == "ssd" else -1
        second_best_score = float("inf") if method == "ssd" else -1

        for j, d2 in enumerate(des2):
            if method == "ssd":
                score = np.sum((d1 - d2) ** 2)  # SSD
                if score < best_score:
                    second_best_score = best_score
                    second_best_match = best_match
                    best_score = score
                    best_match = j
                elif score < second_best_score:
                    second_best_score = score
            elif method == "ncc":
                score = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))  # NCC
                if score > best_score:
                    second_best_score = best_score
                    second_best_match = best_match
                    best_score = score
                    best_match = j
                elif score > second_best_score:
                    second_best_score = score

        if best_match is not None and second_best_score != float("inf"):
            if method == "ssd" and best_score / second_best_score < ratio_threshold:
                matches.append((i, best_match))
                scores.append(best_score)
            elif method == "ncc" and best_score / second_best_score > ratio_threshold:
                matches.append((i, best_match))
                scores.append(best_score)

    if len(matches) > top_n:
        sorted_indices = np.argsort(scores)[:top_n]
        matches = [matches[i] for i in sorted_indices]

    return matches


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output_img[:h1, :w1] = img1
    output_img[:h2, w1:] = img2

    for idx1, idx2 in matches:
        pt1 = tuple(map(int, keypoints1[idx1]))
        pt2 = (int(keypoints2[idx2][0] + w1), int(keypoints2[idx2][1]))
        color = tuple(np.random.randint(0, 255, 3).tolist())  
        cv2.line(output_img, pt1, pt2, color, 2)
        cv2.circle(output_img, pt1, 4, color, -1)
        cv2.circle(output_img, pt2, 4, color, -1)
    return output_img


if __name__ == "__main__":
    img1 = cv2.imread("images/Notre Dam 1.png")
    img2 = cv2.imread("images/Notre Dam 2.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    print("Extracting features using SIFT")
    kp1, des1 = sift_detector(gray1)
    kp2, des2 = sift_detector(gray2)
    print(f"Number of keypoints in image 1: {len(kp1)}")
    print(f"Number of keypoints in image 2: {len(kp2)}")

    print("Performing feature matching")
    matches = match_features(des1, des2, method="ssd")
    print(f"Number of matches found: {len(matches)}")

    print("Drawing matched features")
    matched_img = draw_matches(img1, kp1, img2, kp2, matches)
    print("Feature matching visualization completed")
    print("Displaying the matched image")
    cv2.imshow("Feature Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
