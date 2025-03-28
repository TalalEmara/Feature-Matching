import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage


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

def match_features(des1, des2, top_n=100, ratio_threshold=0.6):
    matches = []
    scores = []

    for i, d1 in enumerate(des1):
        best_match = None
        second_best_match = None
        best_score = float("inf")
        second_best_score = float("inf")

        for j, d2 in enumerate(des2):
            score = np.sum((d1 - d2) ** 2)  # SSD
            if score < best_score:
                second_best_score = best_score
                second_best_match = best_match
                best_score = score
                best_match = j
            elif score < second_best_score:
                second_best_score = score

        if best_match is not None and second_best_score != float("inf"):
            if best_score / second_best_score < ratio_threshold:
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

    print("Performing feature matching")
    matches = match_features(des1, des2)
    print(f"Number of matches found: {len(matches)}")

    print("Drawing matched features")
    matched_img = draw_matches(img1, kp1, img2, kp2, matches)
    
    #directly using pyqt
    #draw_matches(img1, kp1, img2, kp2, matches)

    print("Feature matching visualization completed")
    print("Displaying the matched image")
    cv2.imshow("Feature Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
