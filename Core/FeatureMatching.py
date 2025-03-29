import cv2
import numpy as np
import time

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage


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

import numpy as np

def match_features(des1, des2, method="ssd", top_n=100, ratio_threshold=0.6):
    """
    Parameters:
        des1 : numpy array Feature descriptors from the first image of shape (N1, 128)
        des2 : numpy array Feature descriptors from the second image of shape (N2, 128)
        method : str, Matching method, either 'ssd' or 'ncc'
        top_n : Maximum number of matches to return

    Returns:
        list: List of matched keypoint index pairs [(idx1, idx2)].
    """
    matches = []
    scores = []
    
    if method == "ssd":
        for i, d1 in enumerate(des1):
            dists = np.sum((des2 - d1) ** 2, axis=1) # Squared Euclidean distance
            best_idx = np.argmin(dists)
            second_best_idx = np.partition(dists, 1)[1]
            
            if dists[best_idx] / second_best_idx < ratio_threshold:
                matches.append((i, best_idx))
                scores.append(dists[best_idx])
    
    elif method == "ncc": # Normalized Cross-Correlation
        des1_norm = des1 / (np.linalg.norm(des1, axis=1, keepdims=True) + 1e-10)
        des2_norm = des2 / (np.linalg.norm(des2, axis=1, keepdims=True) + 1e-10)
        
        for i, d1 in enumerate(des1_norm):
            scores_all = np.dot(des2_norm, d1)
            best_idx = np.argmax(scores_all)
            second_best_idx = np.partition(scores_all, -2)[-2]
            
            if scores_all[best_idx] > ratio_threshold and (second_best_idx < 0 or scores_all[best_idx] / second_best_idx > 1.1):
                matches.append((i, best_idx))
                scores.append(scores_all[best_idx])
    
    if len(matches) > top_n:
        sorted_indices = np.argsort(scores)[:top_n] if method == "ssd" else np.argsort(scores)[::-1][:top_n]
        matches = [matches[i] for i in sorted_indices]
    
    return matches


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output_img[:h1, :w1] = img1
    output_img[:h2, w1:] = img2

    np.random.seed(42)  
    colors = np.random.randint(0, 256, (len(matches), 3)).tolist()

    for i, (idx1, idx2) in enumerate(matches):
        pt1 = tuple(map(int, keypoints1[idx1]))
        pt2 = (int(keypoints2[idx2][0] + w1), int(keypoints2[idx2][1]))
        color = tuple(map(int, colors[i]))  

        cv2.line(output_img, pt1, pt2, color, 2) 
        cv2.circle(output_img, pt1, 4, color, -1)
        cv2.circle(output_img, pt2, 4, color, -1)

    return output_img

#IF you want to draw it directly with pyqt not return image
# def draw_matches(img1, keypoints1, img2, keypoints2, matches):
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#     output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     output_img[:h1, :w1] = img1
#     output_img[:h2, w1:] = img2

#     for idx1, idx2 in matches:
#         pt1 = tuple(map(int, keypoints1[idx1]))
#         pt2 = (int(keypoints2[idx2][0] + w1), int(keypoints2[idx2][1]))
#         color = tuple(np.random.randint(0, 255, 3).tolist())  
#         cv2.line(output_img, pt1, pt2, color, 2)
#         cv2.circle(output_img, pt1, 4, color, -1)
#         cv2.circle(output_img, pt2, 4, color, -1)

#     height, width, channel = output_img.shape
#     bytes_per_line = 3 * width
#     q_img = QImage(output_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

#     app = QApplication(sys.argv)
#     window = QWidget()
#     window.setWindowTitle("Feature Matching")

#     label = QLabel()
#     label.setPixmap(QPixmap.fromImage(q_img))

#     layout = QVBoxLayout()
#     layout.addWidget(label)
#     window.setLayout(layout)
#     window.resize(width, height)

#     window.show()
#     sys.exit(app.exec_())



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

    print("Performing feature matching using SSD")
    start_time = time.time()
    matches_ssd = match_features(des1, des2, method="ssd")
    ssd_time = time.time() - start_time
    print(f"SSD Matching Time: {ssd_time:.6f} seconds")
    print(f"Number of matches found (SSD): {len(matches_ssd)}")

    print("Performing feature matching using NCC")
    start_time = time.time()
    matches_ncc = match_features(des1, des2, method="ncc")
    ncc_time = time.time() - start_time
    print(f"NCC Matching Time: {ncc_time:.6f} seconds")
    print(f"Number of matches found (NCC): {len(matches_ncc)}")


    print("Drawing matched features using SSD")
    matched_img1 = draw_matches(img1, kp1, img2, kp2, matches_ssd)

    print("Drawing matched features using NCC")
    matched_img2 = draw_matches(img1, kp1, img2, kp2, matches_ncc)
    
    #directly using pyqt
    #draw_matches(img1, kp1, img2, kp2, matches)

    print("Feature matching visualization completed")
    print("Displaying the matched image")
    cv2.imshow("Feature Matches - SSD", matched_img1)
    cv2.imshow("Feature Matches - NCC", matched_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
