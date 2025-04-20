from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

from gaussian_filter import gaussian_filter
from gaussian_pyramid import generate_gaussian_pyramid
from DoG_pyramid import generate_DoG_pyramid
from keypoints import get_keypoints
from orientation import assign_orientation
from descriptors import get_local_descriptors

class SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussian_filter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr = generate_gaussian_pyramid(self.im, self.num_octave, self.s, self.sigma)
        DoG_pyr = generate_DoG_pyramid(gaussian_pyr)
        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats, kp_pyr
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_sift_implementation():
    # 1. Load test image
    image_path = 'CV/Feature-Matching/images/Feature matching/Notre Dam 2resized.png'  # Replace with your image path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    # 2. Create SIFT instance
    sift = SIFT(image)
    
    # 3. Extract features
    print("Extracting features...")
    features, keypoints_pyramid = sift.get_features()
    
    # 4. Visualize keypoints (from first octave)
    if len(keypoints_pyramid) > 0:
        # Convert your keypoints to OpenCV format for visualization
        kp_cv = []
        for octave_idx, octave_kps in enumerate(keypoints_pyramid):
            for kp in octave_kps:
                # Convert your keypoint format to OpenCV KeyPoint
                # Adjust these conversions based on your actual keypoint format
                x, y = kp[0], kp[1]  # Example - modify according to your keypoint structure
                size = kp[2] if len(kp) > 2 else 10  # Default size if not available
                angle = kp[3] if len(kp) > 3 else -1  # Default angle if not available
                cv_kp = cv2.KeyPoint(x, y, size, angle)
                kp_cv.append(cv_kp)
        
        # Draw keypoints
        img_kp = cv2.drawKeypoints(image, kp_cv, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Show results
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title('Custom SIFT Keypoints')
        plt.axis('off')
        plt.show()
        
        # Print feature statistics
        total_kps = sum(len(octave) for octave in keypoints_pyramid)
        print(f"Total keypoints detected: {total_kps}")
        for i, octave in enumerate(features):
            print(f"Octave {i}: {len(octave)} features, descriptor shape: {octave[0].shape if len(octave) > 0 else 'None'}")
    
    # 5. Optional: Compare with OpenCV's SIFT
    compare_with_opencv(image, kp_cv)

def compare_with_opencv(image, custom_kps):
    """Compare with OpenCV's SIFT implementation"""
    print("\nComparing with OpenCV SIFT...")
    
    # OpenCV SIFT
    sift_cv = cv2.SIFT_create()
    kp_cv, desc_cv = sift_cv.detectAndCompute(image, None)
    
    # Visual comparison
    img_cv = cv2.drawKeypoints(image, kp_cv, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV SIFT Keypoints')
    plt.axis('off')
    
    if len(custom_kps) > 0:
        img_custom = cv2.drawKeypoints(image, custom_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_custom, cv2.COLOR_BGR2RGB))
        plt.title('Custom SIFT Keypoints')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison stats
    print(f"OpenCV SIFT keypoints: {len(kp_cv)}")
    print(f"Custom SIFT keypoints: {len(custom_kps)}")

if __name__ == "__main__":
    test_sift_implementation()