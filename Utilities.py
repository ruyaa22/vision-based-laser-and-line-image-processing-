import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path




















# ==============================
# 5. VIDEO PROCESSING UTILITIES
# ==============================


def extract_frames(video_path, target_fps=5, save_dir=None):
    """
    Extracts frames from a video at a specified FPS.

    Parameters:
        video_path (str): Path to the input video file.
        target_fps (int): Number of frames per second to extract.
        save_dir (str or None): If specified, saves frames to this directory.

    Returns:
        List[np.ndarray]: List of extracted frames (as images in memory).
    """
    frames = []

    # Create directory if saving is requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video file.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))

    print(f"Original FPS: {original_fps:.2f}")
    print(f"Target FPS: {target_fps}, extracting every {frame_interval} frame(s)")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep only every Nth frame
        if frame_count % frame_interval == 0:
            frames.append(frame)

            if save_dir:
                filename = os.path.join(save_dir, f"frame_{saved_count:04d}.png")
                cv2.imwrite(filename, frame)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Extracted {len(frames)} frame(s).")

    return frames

# ==============================
# 1. VISUALIZATION UTILITIES
# ==============================

def show_image_heatmap(image):
    """
    Displays a heatmap visualization of pixel intensity values for an image.

    - For grayscale images: Displays a single heatmap using a 'gray' colormap.
    - For RGB/BGR images: Displays 3 subplots, one for each channel (R, G, B),
      using distinct color-specific colormaps ('Reds', 'Greens', 'Blues').

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
            - Shape (H, W) → interpreted as grayscale.
            - Shape (H, W, 3) → interpreted as color (RGB or BGR).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if image.ndim == 2:
        # Grayscale image: Single-channel heatmap
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.title("Grayscale Heatmap")
        plt.show()

    elif image.ndim == 3 and image.shape[2] == 3:
        # Color image: show R, G, B heatmaps separately
        rgb_image = image[..., ::-1]  # Convert BGR (OpenCV default) to RGB
        colormaps = ['Reds', 'Greens', 'Blues']
        titles = ['Red Channel', 'Green Channel', 'Blue Channel']

        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(rgb_image[:, :, i], cmap=colormaps[i])
            plt.colorbar()
            plt.title(titles[i])
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Unsupported image format. Use 2D grayscale or 3D RGB images.")

def draw_line_on_image(image, line_params):
    """
    Draws a line defined by y = mx + b across the full width of an image.

    Automatically converts grayscale or binary images to BGR so that colored lines (e.g., yellow) are visible.

    Parameters:
        image (np.ndarray): Input image (2D or 3D). 1-channel images will be converted to BGR.
        line_params (Tuple[float, float]): (slope, intercept) of the line y = mx + b.
        color (Tuple[int, int, int]): BGR color of the line (default: yellow).
        thickness (int): Thickness of the drawn line.

    Returns:
        np.ndarray: A BGR image with the line overlaid.
    """
    if line_params is None:
        raise ValueError("line_params cannot be None. Make sure you passed a valid line.")

    m, b = line_params
    h, w = image.shape[:2]

    # Convert 1-channel image to BGR if needed
    if len(image.shape) == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Calculate line endpoints across image width
    x1, x2 = 0, w - 1
    y1 = int(m * x1 + b)
    y2 = int(m * x2 + b)

    image_with_line = image.copy()
    cv2.line(image_with_line, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return image_with_line

def draw_linelaser_on_image(image, line_params, center):
    """
    Draws a line defined by y = mx + b across the full width of an image.

    Automatically converts grayscale or binary images to BGR so that colored lines (e.g., yellow) are visible.

    Parameters:
        image (np.ndarray): Input image (2D or 3D). 1-channel images will be converted to BGR.
        line_params (Tuple[float, float]): (slope, intercept) of the line y = mx + b.
        color (Tuple[int, int, int]): BGR color of the line (default: yellow).
        thickness (int): Thickness of the drawn line.

    Returns:
        np.ndarray: A BGR image with the line overlaid.
    """
    if line_params is None:
        raise ValueError("line_params cannot be None. Make sure you passed a valid line.")

    m, b = line_params
    h, w = image.shape[:2]

    # Convert 1-channel image to BGR if needed
    if len(image.shape) == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Calculate line endpoints across image width
    x1, x2 = 0, w - 1
    y1 = int(m * x1 + b)
    y2 = int(m * x2 + b)

    center_x, center_y = center
    center_x = int(center_x)
    center_y = int(center_y)
    image_with_line = image.copy()
    cv2.line(image_with_line, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(image_with_line, (center_y,center_x), 3, (0, 0, 255), -1)

    return image_with_line

# ==============================
# 3. FEATURE EXTRACTION
# ==============================

def rgb_to_grayscale(image):
    """
    Converts an RGB image to a grayscale image using OpenCV.

    Parameters:
        image (np.ndarray): RGB image with shape (H, W, 3)

    Returns:
        np.ndarray: Grayscale image with shape (H, W), 1 channel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a color image with 3 channels.")
    
    # Convert using OpenCV (weights: R=0.2989, G=0.5870, B=0.1140)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

# ==============================
# 5. MASK GENERATION
# ==============================

def make_binary_mask(gray_img, threshold=50):
    """
    Converts a grayscale image to a binary mask using a fixed threshold.

    Parameters:
        gray_img (np.ndarray): 2D grayscale image.
        threshold (int): Pixel intensity threshold. Pixels > threshold become 255.

    Returns:
        np.ndarray: Binary image (0 or 255 values).
    """
    if gray_img.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")
    
    _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def enhance_image(img):
    """
    Enhances the brightness/contrast of an image using histogram equalization
    on the V (value) channel in HSV space. Returns both enhanced BGR image
    and its HSV version for further masking.

    Parameters:
        img (np.ndarray): BGR image as read by OpenCV.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (enhanced_BGR_image, HSV_version_of_it)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # Equalize brightness
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_img, cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

def extract_laser(img, save_path=None):
    """
    Isolates the red laser dot from a BGR image using HSV thresholding and
    returns the masked result. Optionally saves the output image.

    Parameters:
        img (np.ndarray): Input BGR image.
        save_path (str): Optional file path to save the extracted result.

    Returns:
        np.ndarray: Image containing only the red laser dot region.
    """
    enhanced_img, hsv_enhanced = enhance_image(img)

    # Red spans across the HSV hue boundary (0-180), so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    red_mask1 = cv2.inRange(hsv_enhanced, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_enhanced, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_result = cv2.bitwise_and(enhanced_img, enhanced_img, mask=red_mask)

    if save_path:
        cv2.imwrite(save_path, red_result)
        print(f"Laser saved to: {save_path}")

    return red_result


def extract_line(img, save_path=None):
    """
    Isolates the green line from a BGR image using HSV thresholding and
    returns the masked result. Optionally saves the output image.

    Parameters:
        img (np.ndarray): Input BGR image.
        save_path (str): Optional file path to save the extracted result.

    Returns:
        np.ndarray: Image containing only the green trajectory line.
    """
    enhanced_img, hsv_enhanced = enhance_image(img)

    # HSV threshold range for green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_enhanced, lower_green, upper_green)

    green_result = cv2.bitwise_and(enhanced_img, enhanced_img, mask=green_mask)

    if save_path:
        cv2.imwrite(save_path, green_result)
        print(f"Line saved to: {save_path}")

    return green_result

# ==============================
# 4. GEOMETRIC COMPUTATION
# ==============================

def fit_line_from_binary_mask(binary_mask):
    """
    Fits a straight line (y = mx + b) to the non-zero region in a grayscale or binary image.
    Intended for cases like detecting a green line or edge in a preprocessed mask.

    Parameters:
        gray_mask (np.ndarray): 2D grayscale or binary mask (e.g., thresholded image).
        threshold (int): Pixel intensity cutoff for considering a pixel as part of the line.

    Returns:
        Tuple[float, float] or None: (slope, intercept) of the best-fit line in the form y = mx + b,
                                     or None if insufficient points are found.
    """
    if binary_mask.ndim != 2:
        raise ValueError("Input must be a 2D grayscale or binary image.")

    # Step 2: Get coordinates of white (line) pixels
    points = np.column_stack(np.where(binary_mask > 0))  # (row, col)

    # Step 3: Reformat to (x, y) = (col, row)
    xy_points = np.fliplr(points)

    # Step 4: Fit line using linear regression
    if xy_points.shape[0] >= 2:
        x = xy_points[:, 0]
        y = xy_points[:, 1]
        slope, intercept = np.polyfit(x, y, deg=1)
        return slope, intercept
    else:
        return None  # Not enough data points to fit a line
    

def compute_laser_center(gray):
    """
    Computes the center (centroid) of the non-zero region in a grayscale image.
    Typically used to find the center of a detected laser dot.

    Parameters:
        gray (np.ndarray): Grayscale image where laser region is non-zero.

    Returns:
        Tuple[float, float] or None: (row_center, col_center)
            - Returns None if no non-zero pixels are found.
    """
    non_zero_indices = np.argwhere(gray != 0)

    if non_zero_indices.size == 0:
        return None  # No laser dot detected

    row_center = np.mean(non_zero_indices[:, 0])
    col_center = np.mean(non_zero_indices[:, 1])
    return row_center, col_center

def signed_perpendicular_distance(point, line_params):
    """
    Computes the signed perpendicular distance from a point (x0, y0) to a line y = mx + b.

    Positive distance means the point is above the line (greater y), and negative means below.

    Parameters:
        point (Tuple[float, float]): The point (x0, y0) — e.g., the laser center.
        line_params (Tuple[float, float]): The line parameters (slope m, intercept b).

    Returns:
        float: Signed perpendicular distance from the point to the line.
    """
    y0, x0 = point
    m, b = line_params

    # Compute the numerator and denominator of the perpendicular distance formula
    numerator = m * x0 - y0 + b
    denominator = np.sqrt(m**2 + 1)

    return numerator / denominator

def get_line_angle_and_length(x1, y1, x2, y2):
    """
    Returns the angle (in degrees, normalized to [0, 180)) and length of a line.
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 180
    length = np.hypot(dx, dy)
    return angle, length

def log_result_to_csv(csv_path, frame_name, laser_center, slope, intercept, distance):
    os.makedirs(Path(csv_path).parent, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['frame_name', 'laser_center_row', 'laser_center_col', 'slope', 'intercept', 'distance'])
        if laser_center is None:
            writer.writerow([frame_name, 'None', 'None', slope, intercept, distance])
        else:
            writer.writerow([
                frame_name,
                f"{laser_center[1]:.2f}",
                f"{laser_center[0]:.2f}",
                slope,
                intercept,
                f"{distance:.2f}" if distance is not None else "None"
            ])