import cv2
import numpy as np
import matplotlib.pyplot as plt
from Core.kernelConvolution import gaussian_filter, sobel, custom_convolution
from Core.imageMode import normalize_image, rgb_to_grayscale
from Core.canny import canny

start_point = None
end_point = None
cropping = False

def select_initial_region(image):
    """Allow the user to select an initial region by drawing a rectangle."""
    global start_point, end_point, cropping

    def mouse_callback(event, x, y, flags, param):
        global start_point, end_point, cropping

        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
            start_point = (x, y)
            cropping = True

        elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
            if cropping:
                temp_img = image.copy()
                cv2.rectangle(temp_img, start_point, (x, y), (255, 0, 0), 2)
                cv2.imshow("Select Initial Region", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button release
            end_point = (x, y)
            cropping = False
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Select Initial Region", image)

    cv2.imshow("Select Initial Region", image)
    cv2.setMouseCallback("Select Initial Region", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if start_point and end_point:
        return start_point, end_point
    else:
        raise ValueError("No region selected! Please select an initial region.")

def detect_edges(img):
    if img is None:
        raise ValueError("Input image is None")
    gray = rgb_to_grayscale(img)
    # blurred = gaussian_filter(gray, 5, 1)
    edges1 = canny(gray, 50, 100)
    edges2 = canny(gray, 50, 150)
    edges3 = canny(gray, 80, 200)
    combined_edges = np.maximum.reduce([edges1, edges2, edges3])
    return gray, combined_edges, img

def compute_energy_maps(image):
    gray, combined_edges, original_img = detect_edges(image)
    blurred = gaussian_filter(gray, 5, 1)
    _, _, gradient_magnitude, _ = sobel(blurred, 3)

    # Compute energy map as a weighted sum of gradient magnitude and edges
    # E = 255 - (0.6 * |∇I| + 0.4 * E_edges)
    energy_map = 255 - (0.6 * gradient_magnitude + 0.4 * combined_edges)

    edge_distance = distance_transform(255 - combined_edges)
    edge_distance = normalize_image(edge_distance)

    # Combined energy function , weighted average
    # E_combined = 0.5 * E_gradient + 0.5 * E_distance
    combined_energy = 0.5 * energy_map + 0.5 * edge_distance
    combined_energy = combined_energy.astype(np.float32)

    # Smooth the energy map to reduce noise
    energy_map_smooth = gaussian_filter(combined_energy, 5, 1)
    return energy_map_smooth, original_img

def compute_gradient_directions(energy_map_smooth):
    sobelx = custom_convolution(energy_map_smooth, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobely = custom_convolution(energy_map_smooth, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    
    # Compute gradient magnitude: |∇E| = sqrt((dE/dx)^2 + (dE/dy)^2)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Compute gradient direction: θ = atan2(dE/dy, dE/dx)
    gradient_direction = np.arctan2(sobely, sobelx)
    return gradient_magnitude, gradient_direction

def initialize_contour(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point

    # Create points along the rectangle edges
    top = np.linspace([x1, y1], [x2, y1], 30)
    right = np.linspace([x2, y1], [x2, y2], 30)
    bottom = np.linspace([x2, y2], [x1, y2], 30)
    left = np.linspace([x1, y2], [x1, y1], 30)

    # Stack all points to form a closed contour
    contour = np.vstack([top, right, bottom, left])

    return contour

def compute_external_forces(contour, energy_map_smooth, center_x, center_y):
    h, w = energy_map_smooth.shape
    forces = np.zeros_like(contour)
    for i, point in enumerate(contour):
        dx = point[0] - center_x   # Compute x displacement from center
        dy = point[1] - center_y   # Compute y displacement from center
        angle = np.arctan2(dy, dx )# Compute angle of point relative to center
        current_radius = np.sqrt(dx**2 + dy**2) # Compute current radius from center

        # Find best energy position along radial direction
        search_points = 50
        best_energy = float('inf')
        best_radius = current_radius
        for r in np.linspace(current_radius - 15, current_radius + 15, search_points):
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            if 0 <= int(y) < h and 0 <= int(x) < w:
                x_floor, y_floor = int(x), int(y)
                x_ceil, y_ceil = min(x_floor + 1, w-1), min(y_floor + 1, h-1)
                wx = x - x_floor
                wy = y - y_floor
                
                # Bilinear interpolation for energy value
                energy = (1-wx)*(1-wy)*energy_map_smooth[y_floor, x_floor] + wx*(1-wy)*energy_map_smooth[y_floor, x_ceil] + (1-wx)*wy*energy_map_smooth[y_ceil, x_floor] + wx*wy*energy_map_smooth[y_ceil, x_ceil]
                if energy < best_energy:
                    best_energy = energy
                    best_radius = r
        radius_change = best_radius - current_radius
        max_radius_change = 2.0 * (1.0 / (1.0 + np.exp(-best_energy/50)))
        radius_change = np.clip(radius_change, -max_radius_change, max_radius_change)
        forces[i, 0] = radius_change * np.cos(angle)
        forces[i, 1] = radius_change * np.sin(angle)
    return forces

def compute_internal_forces(contour, gradient_magnitude, gradient_direction):
    h, w = gradient_magnitude.shape
    gradient_forces = np.zeros_like(contour)
    for i, point in enumerate(contour):
        x, y = int(np.clip(point[0], 0, w-1)), int(np.clip(point[1], 0, h-1))
        gdir = gradient_direction[y, x]
        gmag = gradient_magnitude[y, x]
        scale_factor = min(5.0, gmag / 30.0)
        gradient_forces[i, 0] = scale_factor * np.cos(gdir)
        gradient_forces[i, 1] = scale_factor * np.sin(gdir)
    return gradient_forces

def smooth_contour(contour, smooth_weight, rigidity_weight):
    num_points = len(contour)
    smoothed = contour.copy()
    for _ in range(3):
        for i in range(num_points):
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            smoothed[i] = (1 - smooth_weight) * contour[i] + smooth_weight * 0.5 * (contour[prev_idx] + contour[next_idx])
            if i > 0 and i < num_points - 1:
                smoothed[i] += rigidity_weight * (contour[i - 1] - 2 * contour[i] + contour[i + 1])
    return smoothed

def greedy_algorithm(contour, energy_map_smooth, gradient_magnitude, gradient_direction, center_x, center_y, max_iterations, alpha, beta, gamma, adaptive_weights):
    h, w = energy_map_smooth.shape
    for iteration in range(max_iterations):
        prev_contour = contour.copy()
        if adaptive_weights:
            progress = iteration / max_iterations
            ext_weight = gamma * (1.0 - 0.5 * progress)
            smooth_weight = alpha * (1.0 + 0.5 * progress)
            rigidity_weight = beta * (1.0 + 0.5 * progress)
        else:
            ext_weight = gamma
            smooth_weight = alpha
            rigidity_weight = beta

        # Compute external and internal forces
        forces = compute_external_forces(contour, energy_map_smooth, center_x, center_y)
        gradient_forces = compute_internal_forces(contour, gradient_magnitude, gradient_direction)
        combined_forces = 0.5 * forces + 0.5 * gradient_forces
        
        # Update contour position
        contour += ext_weight * combined_forces

        if iteration % 3 == 0:
            contour = smooth_contour(contour, smooth_weight, rigidity_weight)

        contour[:, 0] = np.clip(contour[:, 0], 0, w-1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h-1)

        if iteration % 15 == 0 and iteration > 0:
            contour = resample_contour(contour, len(contour))

        movement = np.mean(np.sqrt(np.sum((contour - prev_contour)**2, axis=1)))
        print(f"Iteration: {iteration}, Movement: {movement}")
        if movement < 0.05:
            print(f"Converged after {iteration + 1} iterations")
            break

    return contour

def snake_active_contour(image, start_point, end_point, alpha=0.1, beta=0.1, gamma=1.0, max_iterations=300, adaptive_weights=True, circularity_weight=0.0):
    """
    Active contour model (Snake) that minimizes energy function:
    
    E = α * E_internal + β * E_rigidity + γ * E_external

    where:
     E_internal controls smoothness of the contour
     E_rigidity keeps contour shape stable
     E_external attracts contour to edges
    """

    energy_map_smooth, original_img = compute_energy_maps(image)
    contour = initialize_contour(start_point, end_point)
    initial_contour = contour.copy()
    center_x, center_y = (start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2
    contour = resample_contour(contour, num_points=120)
    gradient_magnitude, gradient_direction = compute_gradient_directions(energy_map_smooth)
    contour = greedy_algorithm(contour, energy_map_smooth, gradient_magnitude, gradient_direction, center_x, center_y, max_iterations, alpha, beta, gamma, adaptive_weights)

    for _ in range(3):
        num_points = len(contour)
        smoothed = contour.copy()
        for i in range(num_points):
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            smoothed[i] = (1 - alpha) * contour[i] + alpha * 0.5 * (contour[prev_idx] + contour[next_idx])
            if i > 0 and i < num_points - 1:
                smoothed[i] += beta * (contour[i - 1] - 2 * contour[i] + contour[i + 1])
        contour = smoothed

    contour_int = np.round(contour).astype(np.int32)
    for i in range(len(contour_int) - 1):
        draw_line(original_img, tuple(contour_int[i]), tuple(contour_int[i + 1]), (0, 255, 0), 2)
    draw_line(original_img, tuple(contour_int[-1]), tuple(contour_int[0]), (0, 255, 0), 2)

    return original_img, initial_contour, contour

def distance_transform(image):
    dist = np.full(image.shape, np.inf, dtype=np.float32)
    h, w = image.shape

    # First pass: top-left to bottom-right
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0:
                dist[y, x] = 0
            else:
                if x > 0:
                    dist[y, x] = min(dist[y, x], dist[y, x - 1] + 1)
                if y > 0:
                    dist[y, x] = min(dist[y, x], dist[y - 1, x] + 1)

    # Second pass: bottom-right to top-left
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if x < w - 1:
                dist[y, x] = min(dist[y, x], dist[y, x + 1] + 1)
            if y < h - 1:
                dist[y, x] = min(dist[y, x], dist[y + 1, x] + 1)

    return dist

def draw_line(image, pt1, pt2, color, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    for _ in range(thickness):
        x, y = x1, y1
        while True:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = color
            if (x == x2) and (y == y2):
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

def resample_contour(contour, num_points=100):
    contour_tuple = contour.reshape(-1, 2)
    closed_contour = np.vstack([contour_tuple, contour_tuple[0]])
    dist = np.zeros(len(closed_contour))
    for i in range(1, len(closed_contour)):
        dist[i] = dist[i-1] + np.sqrt(np.sum((closed_contour[i] - closed_contour[i-1])**2))
    new_dist = np.linspace(0, dist[-1], num_points)
    new_contour = np.zeros((num_points, 2))
    for i in range(2):
        new_contour[:, i] = np.interp(new_dist, dist, closed_contour[:, i])
    return new_contour

# Explanation of the energy equation:
# The energy equation used in the active contour model is a combination of internal and external energies.
# Internal energy (E_int) is responsible for the smoothness and rigidity of the contour and is computed using gradient-based forces, smoothing, and rigidity.
# External energy (E_ext) is responsible for attracting the contour to the edges and is computed using radial forces based on the energy map.
# The total energy (E) is given by:
# E = E_int + E_ext
# where E_int = alpha * (gradient-based forces) + beta * (rigidity forces) and E_ext = gamma * (radial forces).

# Explanation of the greedy algorithm:
# The active contour model is a greedy algorithm because it iteratively updates the contour to minimize the energy function.
# In each iteration, the algorithm makes a locally optimal choice by adjusting the contour points based on the computed forces.
# This process continues until the contour converges to a stable shape, representing a locally optimal solution.

# Parameter ranges:
# alpha: Controls the elasticity of the contour (range: 0.0 to 1.0)
# beta: Controls the rigidity of the contour (range: 0.0 to 1.0)
# gamma: Controls the attraction to the edges (range: 0.0 to 1.0)

if __name__ == "__main__":
    img_path = "../images/apple.png"
    
    original_img = cv2.imread(img_path)
    image= original_img.copy()
    start_point, end_point = select_initial_region(original_img)

    final_img, initial_contour, final_contour = snake_active_contour(image, start_point, end_point, alpha=0.1, beta=0.1, gamma=1.0)
    
    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.plot(initial_contour[:, 0], initial_contour[:, 1], 'r--', label='Initial Contour')  # Plot initial contour
    plt.plot(final_contour[:, 0], final_contour[:, 1], 'g-', label='Final Contour')  # Plot final contour
    plt.title('Final Image with Contour')
    plt.legend()
    plt.show()