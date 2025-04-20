import cv2 as cv
import numpy as np
from math import exp, sqrt
from numba import njit, prange
import time

def sift(image, num_octaves=4, num_scales=3, contrast_threshold=0.03, edge_threshold=10.0, sigma=1.6):
    """Optimized SIFT implementation without Numba for better compatibility"""
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Pre-process image with initial blur
    image = cv.GaussianBlur(image.astype(np.float32), (0,0), sigmaX=0.5)
    
    # Build pyramids with optimized parameters
    gaussian_pyramid, dog_pyramid = build_pyramids(image, num_octaves, num_scales, sigma)
    
    # Find and refine keypoints
    keypoints = find_keypoints(dog_pyramid, contrast_threshold, edge_threshold)
    refined = refine_keypoints(gaussian_pyramid, dog_pyramid, keypoints, contrast_threshold)
    
    # Assign orientations and compute descriptors
    oriented = assign_orientations(gaussian_pyramid, refined)
    descriptors = compute_descriptors(gaussian_pyramid, oriented)
    
    # Convert to OpenCV keypoints format with accurate size calculation
    custom_kp = []
    for octave, scale, x, y, angle in oriented:
        size = 1.6 * sigma * (2 ** (octave + scale/num_scales))
        kp = cv.KeyPoint(y, x, size, angle)
        custom_kp.append(kp)
    
    return custom_kp, descriptors

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

def build_pyramids(image, num_octaves=4, num_scales=3, sigma=1.6):
    """Optimized pyramid construction with better memory handling"""
    k = 2 ** (1.0 / num_scales)
    gaussian_pyramid = []
    dog_pyramid = []
    
    # Initial image processing with better interpolation
    base_image = cv.resize(image, (0,0), fx=2, fy=2, interpolation=cv.INTER_LANCZOS4)
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
        
        # Prepare next octave with better downsampling
        if octave < num_octaves - 1:
            base_image = cv.resize(octave_images[-3], 
                                 (octave_images[-3].shape[1]//2, octave_images[-3].shape[0]//2),
                                 interpolation=cv.INTER_NEAREST)
    
    return gaussian_pyramid, dog_pyramid

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

@njit(parallel=True)
def find_keypoints_numba(dog_array, contrast_threshold=0.03, edge_threshold=10.0):
    """Numba-optimized keypoint detection using 3D array"""
    keypoints = []
    edge_thresh = (edge_threshold + 1)**2 / edge_threshold
    num_octaves, num_scales, h, w = dog_array.shape
    
    for octave_idx in prange(num_octaves):
        for scale_idx in range(1, num_scales-1):
            for i in range(1, h-1):
                for j in range(1, w-1):
                    val = dog_array[octave_idx, scale_idx, i, j]
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
                            neighbor = dog_array[octave_idx, scale_idx, i+di, j+dj]
                            if val <= neighbor:
                                is_max = False
                            if val >= neighbor:
                                is_min = False
                    
                    # Scale above
                    if is_max or is_min:
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                neighbor = dog_array[octave_idx, scale_idx+1, i+di, j+dj]
                                if val <= neighbor:
                                    is_max = False
                                if val >= neighbor:
                                    is_min = False
                    
                    # Scale below
                    if is_max or is_min:
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                neighbor = dog_array[octave_idx, scale_idx-1, i+di, j+dj]
                                if val <= neighbor:
                                    is_max = False
                                if val >= neighbor:
                                    is_min = False
                    
                    if not (is_max or is_min):
                        continue
                    
                    # Edge response check
                    dxx = dog_array[octave_idx, scale_idx, i+1,j] + dog_array[octave_idx, scale_idx, i-1,j] - 2*val
                    dyy = dog_array[octave_idx, scale_idx, i,j+1] + dog_array[octave_idx, scale_idx, i,j-1] - 2*val
                    dxy = (dog_array[octave_idx, scale_idx, i+1,j+1] - dog_array[octave_idx, scale_idx, i+1,j-1] - 
                           dog_array[octave_idx, scale_idx, i-1,j+1] + dog_array[octave_idx, scale_idx, i-1,j-1]) / 4.0
                    det = dxx*dyy - dxy*dxy
                    
                    if det <= 0:
                        continue
                    
                    tr = dxx + dyy
                    if tr*tr/det >= edge_thresh:
                        continue
                    
                    keypoints.append((octave_idx, scale_idx, i, j))
    
    return keypoints

def find_keypoints(dog_pyramid, contrast_threshold=0.03, edge_threshold=10.0):
    """Keypoint detection without Numba"""
    keypoints = []
    edge_thresh = (edge_threshold + 1)**2 / edge_threshold
    
    for octave_idx, octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(octave)-1):
            dog = octave[scale_idx]
            h, w = dog.shape
            
            # Vectorized neighborhood checking
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
    """Improved keypoint refinement with better convergence checking"""
    refined = []
    
    for octave, scale, i, j in keypoints:
        if octave >= len(dog_pyramid) or scale < 1 or scale >= len(dog_pyramid[octave]) - 1:
            continue
        
        dog = dog_pyramid[octave][scale]
        h, w = dog.shape
        
        if i <= 1 or j <= 1 or i >= h-2 or j >= w-2:
            continue
        
        for _ in range(max_iter):
            # Compute gradient and Hessian
            dx = 0.5 * (dog[i+1,j] - dog[i-1,j])
            dy = 0.5 * (dog[i,j+1] - dog[i,j-1])
            ds = 0.5 * (dog_pyramid[octave][scale+1][i,j] - dog_pyramid[octave][scale-1][i,j])
            grad = np.array([dx, dy, ds])
            
            dxx = dog[i+1,j] + dog[i-1,j] - 2*dog[i,j]
            dyy = dog[i,j+1] + dog[i,j-1] - 2*dog[i,j]
            dss = dog_pyramid[octave][scale+1][i,j] + dog_pyramid[octave][scale-1][i,j] - 2*dog[i,j]
            dxy = 0.25 * (dog[i+1,j+1] - dog[i+1,j-1] - dog[i-1,j+1] + dog[i-1,j-1])
            dxs = 0.25 * (dog_pyramid[octave][scale+1][i+1,j] - dog_pyramid[octave][scale+1][i-1,j] - 
                  dog_pyramid[octave][scale-1][i+1,j] + dog_pyramid[octave][scale-1][i-1,j])
            dys = 0.25 * (dog_pyramid[octave][scale+1][i,j+1] - dog_pyramid[octave][scale+1][i,j-1] - 
                  dog_pyramid[octave][scale-1][i,j+1] + dog_pyramid[octave][scale-1][i,j-1])
            
            H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
            
            try:
                offset = -np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            
            if np.all(np.abs(offset) < 0.5):
                break
                
            i += int(round(offset[0]))
            j += int(round(offset[1]))
            scale += int(round(offset[2]))
            
            # Check new position is valid
            if (scale < 1 or scale >= len(dog_pyramid[octave])-1 or
                i <= 1 or j <= 1 or i >= dog_pyramid[octave][scale].shape[0]-2 or 
                j >= dog_pyramid[octave][scale].shape[1]-2):
                break
        
        else:  # No break occurred, max_iter reached
            continue
            
        # Final contrast check
        contrast = dog[i,j] + 0.5 * np.dot(grad, offset)
        if abs(contrast) < contrast_threshold:
            continue
            
        refined.append((octave, scale, i, j))
    
    return refined

def assign_orientations(gaussian_pyramid, keypoints, num_bins=36):
    """Optimized orientation assignment with better peak detection"""
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
        y_coords = y_coords - radius
        x_coords = x_coords - radius
        gauss_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * (1.5 * 2**octave)**2))
        
        bin_indices = (ori / bin_width).astype(int) % num_bins
        for bin_idx in range(num_bins):
            mask = (bin_indices == bin_idx)
            hist[bin_idx] = np.sum(mag[mask] * gauss_weights[mask])
        
        # Find peaks with better interpolation
        max_val = np.max(hist)
        if max_val < 0.1:
            continue
            
        peak_threshold = 0.8 * max_val
        for bin_idx in range(num_bins):
            if hist[bin_idx] >= peak_threshold:
                left = (bin_idx - 1) % num_bins
                right = (bin_idx + 1) % num_bins
                
                if hist[left] < hist[bin_idx] and hist[right] < hist[bin_idx]:
                    # Better parabolic interpolation
                    offset = 0.5 * (hist[left] - hist[right]) / max(1e-6, (hist[left] - 2*hist[bin_idx] + hist[right]))
                    angle = (bin_idx + offset) * bin_width % 360
                    oriented.append((octave, scale, i, j, angle))
    
    return oriented

def compute_descriptors(gaussian_pyramid, keypoints):
    """Improved descriptor computation with better normalization"""
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
        
        # Improved normalization
        norm = np.linalg.norm(descriptor)
        if norm > 1e-7:
            descriptor = descriptor / norm
            descriptor = np.clip(descriptor, 0, 0.2)
            descriptor = descriptor / max(1e-7, np.linalg.norm(descriptor))
        
        descriptors.append(descriptor)
    
    return np.array(descriptors)


##### TEST #####
def test():
    """Optimized test function with timing and comparison"""
    image = cv.imread('CV/Feature-Matching/images/Feature matching/Notre Dam 1resized.png')
    if image is None:
        print("Error: Image not found!")
        return
    
    # Test custom implementation
    start_time = cv.getTickCount()
    keypoints, descriptors = sift(image, num_octaves=3, num_scales=3, contrast_threshold=0.03, edge_threshold=4.0)
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

test()