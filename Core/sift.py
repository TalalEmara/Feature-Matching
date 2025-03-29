# SIFT
import cv2 as cv
import numpy as np

# Step 1: Scale space construction
def space_scale_construction(image, num_octaves, num_scales=2, sigma=1.6):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = image_gray.astype(np.float32)
    k = 2 ** (1 / num_scales)
    gaussian_pyramid = []
    dog_pyramid = []

    for o in range(num_octaves):
        gaussian_images = []
        dog_images = []
        for s in range(num_scales + 3):
            sigma_actual = sigma * (k ** s)
            blurred = cv.GaussianBlur(image_gray, (0, 0), sigma_actual)
            gaussian_images.append(blurred)
            if s > 0:
                dog = cv.subtract(gaussian_images[s], gaussian_images[s - 1])
                dog_images.append(dog)
        
        gaussian_pyramid.append(gaussian_images)
        dog_pyramid.append(dog_images)
        image_gray = cv.resize(image_gray, (image_gray.shape[1] // 2, image_gray.shape[0] // 2), interpolation=cv.INTER_NEAREST)
    
    return gaussian_pyramid, dog_pyramid

# Step 2: Scale-space extrema detection
def space_scale_extrema(dog_pyramid, threshold=0.03, r=10):
    key_points = []
    edge_threshold = (r + 1) ** 2 / r ** 2

    for i in range(len(dog_pyramid)):
        for s in range(1, len(dog_pyramid[i]) - 1):
            dog = dog_pyramid[i][s]
            prev_dog = dog_pyramid[i][s - 1]
            next_dog = dog_pyramid[i][s + 1]

            h, w = dog.shape
            for x in range(1, h - 1):
                for y in range(1, w - 1):
                    pixel = dog[x, y]
                    neighbors = np.concatenate([
                        prev_dog[x-1:x+2, y-1:y+2].flatten(),
                        dog[x-1:x+2, y-1:y+2].flatten(),
                        next_dog[x-1:x+2, y-1:y+2].flatten()
                    ])
                    neighbors = np.delete(neighbors, 13)

                    if pixel > np.max(neighbors) or pixel < np.min(neighbors):
                        if abs(pixel) > threshold:
                            dxx = dog[x+1, y] + dog[x-1, y] - 2 * pixel
                            dyy = dog[x, y+1] + dog[x, y-1] - 2 * pixel
                            dxy = (dog[x+1, y+1] - dog[x+1, y-1] - dog[x-1, y+1] + dog[x-1, y-1]) / 4

                            det_h = (dxx * dyy) - (dxy ** 2)
                            trace_h = dxx + dyy
                            if det_h > 0:
                                R = (trace_h ** 2) / det_h
                                if R < edge_threshold:
                                    key_points.append((i, s, x, y))
    
    return key_points

def refine_keypoints(dog_pyramid, keypoints, contrast_threshold=0.03):
    refined_keypoints = []
    
    for i, s, x, y in keypoints:
        dog = dog_pyramid[i][s]
        h, w = dog.shape
        if x <= 1 or x >= h-1 or y <= 1 or y >= w-1:
            continue

        Dx = (dog[x+1, y] - dog[x-1, y]) / 2
        Dy = (dog[x, y+1] - dog[x, y-1]) / 2
        Ds = (dog_pyramid[i][s+1][x, y] - dog_pyramid[i][s-1][x, y]) / 2
        gradient = np.array([Dx, Dy, Ds])

        Dxx = dog[x+1, y] + dog[x-1, y] - 2 * dog[x, y]
        Dyy = dog[x, y+1] + dog[x, y-1] - 2 * dog[x, y]
        Dss = dog_pyramid[i][s+1][x, y] + dog_pyramid[i][s-1][x, y] - 2 * dog[x, y]
        Dxy = (dog[x+1, y+1] - dog[x+1, y-1] - dog[x-1, y+1] + dog[x-1, y-1]) / 4
        Dxs = (dog_pyramid[i][s+1][x+1, y] - dog_pyramid[i][s+1][x-1, y] - 
               dog_pyramid[i][s-1][x+1, y] + dog_pyramid[i][s-1][x-1, y]) / 4
        Dys = (dog_pyramid[i][s+1][x, y+1] - dog_pyramid[i][s+1][x, y-1] - 
               dog_pyramid[i][s-1][x, y+1] + dog_pyramid[i][s-1][x, y-1]) / 4

        Hessian = np.array([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
        try:
            offset = -np.linalg.inv(Hessian) @ gradient
        except np.linalg.LinAlgError:
            continue

        if np.abs(offset).max() > 0.5:
            continue
        
        DoG_value = dog[x, y] + 0.5 * gradient @ offset
        if abs(DoG_value) < contrast_threshold:
            continue 
    
        refined_keypoints.append((i, s, x + int(round(offset[0])), y + int(round(offset[1]))))

    return refined_keypoints


# Step 3: Orientation assignment
def assign_orientation():
    pass

# Step 4: Keypoint descriptor
def keypoint_descriptor():
    pass