# SIFT
import cv2 as cv
import numpy as np
from math import pi, exp, sqrt

def sift(image, num_octaves=4, num_scales=3, contrast_threshold=0.01, edge_threshold=12.0):
    """Complete SIFT implementation"""
    gaussian_pyramid, dog_pyramid = space_scale_construction(image, num_octaves, num_scales)
    keypoints = find_keypoints(dog_pyramid, contrast_threshold, edge_threshold)
    refined = refine_keypoints(gaussian_pyramid, dog_pyramid, keypoints, contrast_threshold)
    oriented = assign_orientations(gaussian_pyramid, refined)
    descriptors = compute_descriptors(gaussian_pyramid, oriented)
    
    # Convert custom keypoints to OpenCV format for visualization
    custom_kp = []
    for octave, scale, x, y, angle in oriented:
        # Adjust for octave scaling
        size = 1.6 * (2 ** octave)
        kp = cv.KeyPoint(y, x, size, angle)
        custom_kp.append(kp)
    return custom_kp, descriptors

def space_scale_construction(image, num_octaves=4, num_scales=3, sigma=1.6):
    """Build Gaussian and Difference of Gaussian pyramids"""
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    
    k = 2 ** (1.0 / num_scales)
    gaussian_pyramid = []
    dog_pyramid = []
    
    # Initial image doubling and blur
    base_image = cv.resize(image, (0,0), fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    base_image = cv.GaussianBlur(base_image, (0,0), sigmaX=sqrt(sigma**2 - 0.5**2))
    
    for octave in range(num_octaves):
        octave_images = [base_image]
        for s in range(1, num_scales + 3):
            sigma_total = sigma * (k ** s)
            octave_images.append(cv.GaussianBlur(base_image, (0,0), sigmaX=sigma_total))
        
        gaussian_pyramid.append(octave_images)
        
        # Build DoG for current octave
        dog_images = []
        for s in range(len(octave_images)-1):
            dog_images.append(octave_images[s+1] - octave_images[s])
        dog_pyramid.append(dog_images)
        
        # Prepare next octave
        base_image = cv.resize(octave_images[-3], (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    
    return gaussian_pyramid, dog_pyramid

def find_keypoints(dog_pyramid, contrast_threshold=0.03, edge_threshold=10.0):
    """Detect scale-space extrema"""
    keypoints = []
    edge_thresh = (edge_threshold + 1)**2 / edge_threshold
    
    for octave_idx, octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(octave)-1):
            dog = octave[scale_idx]
            h, w = dog.shape
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # 26-neighbor comparison
                    val = dog[i,j]
                    if abs(val) < contrast_threshold:
                        continue
                    
                    # Check if extremum
                    neighbors = [
                        octave[scale_idx-1][i-1:i+2, j-1:j+2],
                        octave[scale_idx][i-1:i+2, j-1:j+2],
                        octave[scale_idx+1][i-1:i+2, j-1:j+2]
                    ]
                    neighbors = np.concatenate([n.flatten() for n in neighbors])
                    neighbors = np.delete(neighbors, 13)  # Remove center
                    
                    if not (val > np.max(neighbors) or val < np.min(neighbors)):
                        continue
                    
                    # Edge response check
                    dxx = dog[i+1,j] + dog[i-1,j] - 2*val
                    dyy = dog[i,j+1] + dog[i,j-1] - 2*val
                    dxy = (dog[i+1,j+1] - dog[i+1,j-1] - dog[i-1,j+1] + dog[i-1,j-1])/4.0
                    det = dxx*dyy - dxy*dxy
                    
                    if det <= 0:
                        continue
                    
                    tr = dxx + dyy
                    if tr*tr/det >= edge_thresh:
                        continue
                    
                    keypoints.append((octave_idx, scale_idx, i, j))
    
    return keypoints

def refine_keypoints(gaussian_pyramid, dog_pyramid, keypoints, contrast_threshold=0.03, max_iter=5):
    refined = []

    for octave, scale, i, j in keypoints:
        # Ensure octave and scale are within valid range
        if octave >= len(dog_pyramid) or scale < 1 or scale >= len(dog_pyramid[octave]) - 1:
            continue

        dog = dog_pyramid[octave][scale]
        h, w = dog.shape

        # Skip keypoints too close to image borders
        if i <= 1 or j <= 1 or i >= h - 2 or j >= w - 2:
            continue

        for _ in range(max_iter):
            # Re-check bounds in case offset pushes out of bounds
            if i <= 1 or j <= 1 or i >= h - 2 or j >= w - 2 or scale < 1 or scale >= len(dog_pyramid[octave]) - 1:
                break

            # Compute gradient and Hessian
            dx = (dog[i+1,j] - dog[i-1,j])/2.0
            dy = (dog[i,j+1] - dog[i,j-1])/2.0
            ds = (dog_pyramid[octave][scale+1][i,j] - dog_pyramid[octave][scale-1][i,j])/2.0
            grad = np.array([dx, dy, ds])

            dxx = dog[i+1,j] + dog[i-1,j] - 2*dog[i,j]
            dyy = dog[i,j+1] + dog[i,j-1] - 2*dog[i,j]
            dss = dog_pyramid[octave][scale+1][i,j] + dog_pyramid[octave][scale-1][i,j] - 2*dog[i,j]
            dxy = (dog[i+1,j+1] - dog[i+1,j-1] - dog[i-1,j+1] + dog[i-1,j-1])/4.0
            dxs = (dog_pyramid[octave][scale+1][i+1,j] - dog_pyramid[octave][scale+1][i-1,j] -
                   dog_pyramid[octave][scale-1][i+1,j] + dog_pyramid[octave][scale-1][i-1,j])/4.0
            dys = (dog_pyramid[octave][scale+1][i,j+1] - dog_pyramid[octave][scale+1][i,j-1] -
                   dog_pyramid[octave][scale-1][i,j+1] + dog_pyramid[octave][scale-1][i,j-1])/4.0

            H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

            try:
                offset = -np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break

            if np.abs(offset).max() < 0.5:
                break

            i += int(round(offset[0]))
            j += int(round(offset[1]))
            scale += int(round(offset[2]))

        else:
            continue  # max_iter was reached, discard point

        # Final bounds check
        if i <= 1 or j <= 1 or i >= h - 2 or j >= w - 2 or scale < 1 or scale >= len(dog_pyramid[octave]) - 1:
            continue

        dog = dog_pyramid[octave][scale]
        contrast = dog[i,j] + 0.5 * np.dot(grad, offset)
        if abs(contrast) < contrast_threshold:
            continue

        refined.append((octave, scale, i, j))

    return refined


def assign_orientations(gaussian_pyramid, keypoints, num_bins=36):
    """Assign dominant orientations to keypoints"""
    oriented = []
    bin_width = 360 / num_bins
    
    for octave, scale, i, j in keypoints:
        # Skip invalid scales
        if scale < 0 or scale >= len(gaussian_pyramid[octave]):
            continue
            
        img = gaussian_pyramid[octave][scale]
        h, w = img.shape
        
        # Skip boundary points
        radius = int(round(3 * 1.5 * (2 ** octave)))
        if i < radius or j < radius or i >= h-radius or j >= w-radius:
            continue
            
        # Compute gradients
        dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx*dx + dy*dy)
        ori = (np.arctan2(dy, dx) * 180 / np.pi) % 360
        
        # Create orientation histogram
        hist = np.zeros(num_bins)
        
        # Weight by Gaussian window
        for x in range(i-radius, i+radius+1):
            for y in range(j-radius, j+radius+1):
                if 0 <= x < h and 0 <= y < w:  # Additional boundary check
                    weight = exp(-((x-i)**2 + (y-j)**2) / (2 * (1.5 * 2**octave)**2))
                    bin_idx = int(ori[x,y] / bin_width) % num_bins
                    hist[bin_idx] += mag[x,y] * weight
        
        # Find peaks in histogram
        max_val = np.max(hist)
        if max_val < 0.1:  # Skip weak keypoints
            continue
            
        # Find all peaks within 80% of max
        peak_threshold = 0.8 * max_val
        for bin_idx in range(num_bins):
            if hist[bin_idx] >= peak_threshold:
                # Parabolic interpolation for more accurate orientation
                left = (bin_idx - 1) % num_bins
                right = (bin_idx + 1) % num_bins
                
                # Quadratic interpolation
                if hist[left] < hist[bin_idx] and hist[right] < hist[bin_idx]:
                    offset = 0.5 * (hist[left] - hist[right]) / (hist[left] - 2*hist[bin_idx] + hist[right])
                    angle = (bin_idx + offset) * bin_width % 360
                    oriented.append((octave, scale, i, j, angle))
    
    return oriented

def compute_descriptors(gaussian_pyramid, keypoints):
    """Compute 128-dim SIFT descriptors"""
    descriptors = []
    
    for octave, scale, i, j, angle in keypoints:
        img = gaussian_pyramid[octave][scale]
        h, w = img.shape
        
        # Compute gradients relative to keypoint orientation
        dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx*dx + dy*dy)
        ori = (np.arctan2(dy, dx) * 180 / np.pi - angle) % 360
        
        # Rotate coordinates
        cos_angle = np.cos(angle * np.pi / 180)
        sin_angle = np.sin(angle * np.pi / 180)
        
        descriptor = np.zeros(128)
        hist_idx = 0
        
        for x in range(-8, 8, 4):
            for y in range(-8, 8, 4):
                hist = np.zeros(8)
                
                for sub_x in range(x, x+4):
                    for sub_y in range(y, y+4):
                        # Rotate sample location
                        rot_x = sub_x * cos_angle - sub_y * sin_angle
                        rot_y = sub_x * sin_angle + sub_y * cos_angle
                        
                        sample_x = int(round(i + rot_x))
                        sample_y = int(round(j + rot_y))
                        
                        if 0 <= sample_x < h and 0 <= sample_y < w:
                            bin = int(ori[sample_x, sample_y] / 45) % 8
                            weight = mag[sample_x, sample_y] * \
                                     exp(-(rot_x**2 + rot_y**2) / (2 * (16/3)**2))
                            hist[bin] += weight
                
                descriptor[hist_idx:hist_idx+8] = hist
                hist_idx += 8
        
        # Normalize descriptor
        descriptor /= np.linalg.norm(descriptor)
        descriptor = np.clip(descriptor, 0, 0.2)
        descriptor /= np.linalg.norm(descriptor)
        
        descriptors.append(descriptor)
    
    return np.array(descriptors)


##### TEST #####
def test():
    # Load test image
    image = cv.imread('CV/Feature-Matching/images/box.png')
    if image is None:
        print("Error: Image not found!")
        return

    print(f"[DEBUG] Loaded test image with shape {image.shape}")

    # start time calculations
    start_time = cv.getTickCount()
    print(f"[DEBUG] Starting SIFT computation...{start_time}")
    # Run custom SIFT implementation
    keypoints, descriptors = sift(image)
    # print(f"[DEBUG] keypoints: {keypoints}")
    end_time = cv.getTickCount()
    time_taken = (end_time - start_time) / cv.getTickFrequency()
    print(f"[DEBUG] SIFT computation completed in {time_taken:.4f} seconds")
    print(f"[DEBUG] Number of keypoints detected: {len(keypoints)}")
    print(f"[DEBUG] Descriptor shape: {descriptors.shape}")


    # Draw custom keypoints
    img_custom = image.copy()
    img_custom = cv.drawKeypoints(image, keypoints, img_custom,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Run OpenCV SIFT
    sift_built_in = cv.SIFT_create()
    kp_opencv, desc_opencv = sift_built_in.detectAndCompute(image, None)
    # print(f"[DEBUG] OpenCV keypoints: {kp_opencv}")

    # Draw OpenCV keypoints
    img_opencv = image.copy()
    img_opencv = cv.drawKeypoints(image, kp_opencv, img_opencv,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Print comparison stats
    print("\n=== Comparison Results ===")
    print(f"Custom SIFT keypoints: {len(keypoints)}")
    print(f"OpenCV SIFT keypoints: {len(kp_opencv)}")
    print(f"Keypoint ratio (custom/OpenCV): {len(keypoints) / len(kp_opencv):.2f}")

    if len(keypoints) > 0:
        print(f"Custom descriptor shape: {descriptors.shape}")
    print(f"OpenCV descriptor shape: {desc_opencv.shape}")

    # Display results side by side
    comparison = np.hstack((img_custom, img_opencv))
    cv.imshow("SIFT Comparison: Custom (Left) vs OpenCV (Right)", comparison)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save results
    cv.imwrite("custom_sift.jpg", img_custom)
    cv.imwrite("opencv_sift.jpg", img_opencv)
    cv.imwrite("sift_comparison.jpg", comparison)

test()