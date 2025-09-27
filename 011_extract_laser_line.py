import cv2
import numpy as np
import os
from pathlib import Path
from Utilities import *



# === PATHS ===
base_dir = Path(r'C:\Users\roaan\OneDrive - KFUPM\Quanser Thesis\trimmed_videos_22_06\video_6_Y')
input_dir = base_dir
output_segmentation_dir = base_dir / 'segmentation'
line_dir = output_segmentation_dir / 'line'
laser_dir = output_segmentation_dir / 'laser'
contour_dir = output_segmentation_dir / 'contour'

os.makedirs(line_dir, exist_ok=True)
os.makedirs(laser_dir, exist_ok=True)
os.makedirs(contour_dir, exist_ok=True)

frames = sorted(input_dir.glob("frame_*.jpg"))
if not frames:
    print(f"⚠️ No frames found in {input_dir}")

for idx, frame_path in enumerate(frames):
    if idx == 450:
        break;
    
    frame_name = frame_path.stem
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"❌ Could not read {frame_path}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

    # === GREEN LINE ===
       # === GREEN LINE DETECTION ===
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_enhanced, lower_green, upper_green)

    # Morphological cleanup to remove noise and bridge gaps
    kernel = np.ones((9, 9), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)   # Remove small specks
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)           # Thicken faint areas

    # Filter contours by area and aspect ratio
    cleaned_green = np.zeros_like(green_mask)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = img.copy()                # Overlay on original

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        inv_aspect_ratio = h / w if w != 0 else 0

        if area > 400 and (aspect_ratio > 4 or inv_aspect_ratio > 4):
            cv2.drawContours(cleaned_green, [cnt], -1, 255, -1)
            cv2.drawContours(result_image, [cnt], -1, (255, 255, 0), 2)

    green_result = cv2.bitwise_and(enhanced_img, enhanced_img, mask=cleaned_green)
    

    # === RED LASER DOT ===

    # Convert to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Brightness thresholding to isolate bright white laser dot
    lower_rgb = np.array([240, 240, 240])
    upper_rgb = np.array([255, 255, 255])
    mask_white = cv2.inRange(image_rgb, lower_rgb, upper_rgb)

    # Step 2: Find contours
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Find the largest circular contour
    best_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.7 and area > max_area:
            best_contour = cnt
            max_area = area

    # If not confident, fallback to HSV red detection
    if best_contour is None or max_area < 20:
        print("⚠️  Fallback to HSV red detection")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_contour = max(contours, key=cv2.contourArea)

    
    # Step 4: Create output images
    filtered_mask = np.zeros_like(mask_white)  # Binary mask
    red_result = np.zeros_like(img)        # Red-on-black segmented version

    if best_contour is not None:
        # Binary mask
        cv2.drawContours(filtered_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
        
        # Overlay on original image
        cv2.drawContours(result_image, [best_contour], -1, (255, 0, 0), 2)
        
        # Red filled contour on black
        cv2.drawContours(red_result, [best_contour], -1, (0, 0, 255), thickness=cv2.FILLED)


     # === Save Outputs ===
    cv2.imwrite(str(line_dir / f"{frame_name}_line.png"), green_result)
    cv2.imwrite(str(laser_dir / f"{frame_name}_laser.png"), red_result)
    cv2.imwrite(str(contour_dir / f"{frame_name}_contour.png"), result_image)
    print(f"✅ Saved: {frame_name}_line.png and {frame_name}_laser.png")


    #print("✅ All frames processed and saved.")

