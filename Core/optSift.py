import cv2 as cv
import numpy as np
from math import pi, exp, sqrt
from numba import njit, prange

def sift(image, num_octaves=4, num_scales=3, contrast_threshold=0.01, edge_threshold=10.0):
    """Optimized SIFT implementation with Numba compatibility"""
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Build pyramids
    gaussian_pyramid, dog_pyramid = build_pyramids(image, num_octaves, num_scales)
    
    # Find and refine keypoints (without Numba for pyramid processing)
    keypoints = find_keypoints(dog_pyramid, contrast_threshold, edge_threshold)
    refined = refine_keypoints(gaussian_pyramid, dog_pyramid, keypoints, contrast_threshold)
    
    # Assign orientations and compute descriptors
    oriented = assign_orientations(gaussian_pyramid, refined)
    descriptors = compute_descriptors(gaussian_pyramid, oriented)
    
    # Convert to OpenCV keypoints format
    custom_kp = []
    for octave, scale, x, y, angle in oriented:
        size = 1.6 * (2 ** octave)
        kp = cv.KeyPoint(y, x, size, angle)
        custom_kp.append(kp)
    
    return custom_kp, descriptors

def convert_pyramid_to_3darray(dog_pyramid):
    """Convert the pyramid structure to a 3D numpy array for Numba"""
    num_octaves = len(dog_pyramid)
    num_scales = len(dog_pyramid[0])
    h, w = dog_pyramid[0][0].shape
    
    # Create a 3D array with dimensions (octaves, scales, height*width)
    # We'll reshape it in the Numba function
    dog_array = np.zeros((num_octaves, num_scales, h, w), dtype=np.float32)
    
    for o in range(num_octaves):
        for s in range(num_scales):
            dog_array[o, s] = dog_pyramid[o][s]
    
    return dog_array

def build_pyramids(image, num_octaves=4, num_scales=3, sigma=1.6):
    """Build Gaussian and Difference of Gaussian pyramids"""
    image = image.astype(np.float32)
    k = 2 ** (1.0 / num_scales)
    
    gaussian_pyramid = []
    dog_pyramid = []
    
    # Initial image processing
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
    """Keypoint detection without Numba for variable-sized octaves"""
    keypoints = []
    edge_thresh = (edge_threshold + 1)**2 / edge_threshold
    
    for octave_idx, octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(octave)-1):
            dog = octave[scale_idx]
            h, w = dog.shape
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    val = dog[i,j]
                    if abs(val) < contrast_threshold:
                        continue
                    
                    # Check 26-neighborhood
                    is_max = True
                    is_min = True
                    
                    # Current scale
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            neighbor = dog[i+di,j+dj]
                            if val <= neighbor:
                                is_max = False
                            if val >= neighbor:
                                is_min = False
                    
                    # Scale above
                    if is_max or is_min:
                        scale_above = octave[scale_idx+1]
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                neighbor = scale_above[i+di,j+dj]
                                if val <= neighbor:
                                    is_max = False
                                if val >= neighbor:
                                    is_min = False
                    
                    # Scale below
                    if is_max or is_min:
                        scale_below = octave[scale_idx-1]
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                neighbor = scale_below[i+di,j+dj]
                                if val <= neighbor:
                                    is_max = False
                                if val >= neighbor:
                                    is_min = False
                    
                    if not (is_max or is_min):
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
    """Optimized orientation assignment with vectorized operations"""
    oriented = []
    bin_width = 360 / num_bins
    
    for octave, scale, i, j in keypoints:
        if scale < 0 or scale >= len(gaussian_pyramid[octave]):
            continue
            
        img = gaussian_pyramid[octave][scale]
        h, w = img.shape
        radius = int(round(3 * 1.5 * (2 ** octave)))
        
        if i < radius or j < radius or i >= h-radius or j >= w-radius:
            continue
            
        # Compute gradients in the region of interest
        roi = img[i-radius:i+radius+1, j-radius:j+radius+1]
        dx = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
        dy = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx*dx + dy*dy)
        ori = (np.arctan2(dy, dx) * 180 / np.pi) % 360
        
        # Create orientation histogram
        hist = np.zeros(num_bins)
        
        # Weight by Gaussian window
        y_coords, x_coords = np.indices(ori.shape)
        y_coords = y_coords - radius  # Center at keypoint
        x_coords = x_coords - radius
        gauss_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * (1.5 * 2**octave)**2))
        
        bin_indices = (ori / bin_width).astype(int) % num_bins
        for bin_idx in range(num_bins):
            mask = (bin_indices == bin_idx)
            hist[bin_idx] = np.sum(mag[mask] * gauss_weights[mask])
        
        # Find peaks
        max_val = np.max(hist)
        if max_val < 0.1:
            continue
            
        peak_threshold = 0.8 * max_val
        for bin_idx in range(num_bins):
            if hist[bin_idx] >= peak_threshold:
                # Find local maxima
                left = (bin_idx - 1) % num_bins
                right = (bin_idx + 1) % num_bins
                
                if hist[left] < hist[bin_idx] and hist[right] < hist[bin_idx]:
                    offset = 0.5 * (hist[left] - hist[right]) / (hist[left] - 2*hist[bin_idx] + hist[right])
                    angle = (bin_idx + offset) * bin_width % 360
                    oriented.append((octave, scale, i, j, angle))
    
    return oriented

def compute_descriptors(gaussian_pyramid, keypoints):
    """Optimized descriptor computation with vectorized operations"""
    descriptors = []
    
    for octave, scale, i, j, angle in keypoints:
        img = gaussian_pyramid[octave][scale]
        h, w = img.shape
        
        # Pre-compute gradients
        dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx*dx + dy*dy)
        ori = (np.arctan2(dy, dx) * 180 / np.pi - angle) % 360
        
        # Pre-compute rotation matrix
        angle_rad = angle * np.pi / 180
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        descriptor = np.zeros(128)
        
        # Process 4x4 blocks
        for block_y in range(-8, 8, 4):
            for block_x in range(-8, 8, 4):
                hist = np.zeros(8)
                
                # Process each pixel in the 4x4 block
                for sub_y in range(block_y, block_y+4):
                    for sub_x in range(block_x, block_x+4):
                        # Rotate sample location
                        rot_x = sub_x * cos_angle - sub_y * sin_angle
                        rot_y = sub_x * sin_angle + sub_y * cos_angle
                        
                        sample_x = int(round(i + rot_x))
                        sample_y = int(round(j + rot_y))
                        
                        if 0 <= sample_x < h and 0 <= sample_y < w:
                            bin_idx = int(ori[sample_x, sample_y] / 45) % 8
                            weight = mag[sample_x, sample_y] * exp(-(rot_x**2 + rot_y**2) / (2 * (16/3)**2))
                            hist[bin_idx] += weight
                
                # Add to descriptor
                descriptor[(block_y+8)//4*32 + (block_x+8)//4*8 : 
                          (block_y+8)//4*32 + (block_x+8)//4*8 + 8] = hist
        
        # Normalize descriptor
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
            descriptor = np.clip(descriptor, 0, 0.2)
            descriptor /= np.linalg.norm(descriptor)
        
        descriptors.append(descriptor)
    
    return np.array(descriptors)

def test_optimized():
    """Optimized test function with timing and comparison"""
    image = cv.imread('CV/Feature-Matching/images/colored2.jpg')
    if image is None:
        print("Error: Image not found!")
        return
    
    # Test custom implementation
    start_time = cv.getTickCount()
    keypoints, descriptors = sift(image)
    custom_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    
    # Test OpenCV implementation
    start_time = cv.getTickCount()
    sift_built_in = cv.SIFT_create()
    kp_opencv, desc_opencv = sift_built_in.detectAndCompute(image, None)
    opencv_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    
    # Print results
    print("\n=== Optimized SIFT Results ===")
    print(f"Custom SIFT time: {custom_time:.4f} seconds")
    print(f"OpenCV SIFT time: {opencv_time:.4f} seconds")
    print(f"Speedup ratio: {opencv_time/max(0.001, custom_time):.2f}x")
    print(f"Custom keypoints: {len(keypoints)}")
    print(f"OpenCV keypoints: {len(kp_opencv)}")
    
    # Visualization
    img_custom = cv.drawKeypoints(image, keypoints, None, 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_opencv = cv.drawKeypoints(image, kp_opencv, None,
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    comparison = np.hstack((img_custom, img_opencv))
    cv.imshow("Optimized SIFT Comparison", comparison)
    cv.waitKey(0)
    cv.destroyAllWindows()

# test_optimized()