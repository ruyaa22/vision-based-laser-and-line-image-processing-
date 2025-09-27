import cv2
import numpy as np
import os
from pathlib import Path
from Utilities import *

# === Video Identifier ===
video_name = 'video'

# === INPUT PATHS ===
base_dir = Path(r'C:\Users\roaan\OneDrive - KFUPM\Quanser Thesis\trimmed_videos_22_06\video_7_Y')
segmentation_dir = base_dir / 'segmentation'
line_dir = segmentation_dir / 'line'
laser_dir = segmentation_dir / 'laser'

# === OUTPUT PATHS ===
output_dir = base_dir / 'error'
fitting_dir = output_dir / 'line'     # Line fit visualizations
processed_dir = output_dir / 'laser'  # Laser-line overlay visualizations
csv_output_path = output_dir / 'results.csv'

# === CREATE OUTPUT FOLDERS ===
os.makedirs(fitting_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
print(f"✅ Output folders created:")
print(f"  → Line fit images: {fitting_dir}")
print(f"  → Laser overlay images: {processed_dir}")

# === Process Each Frame Pair ===
line_frames = sorted(line_dir.glob("frame_*_line.png"))
if not line_frames:
    print(f"❌ No line frames found in: {line_dir}")
    exit()

for line_path in line_frames:
    # try:
        frame_base = line_path.stem.replace("_line", "")
        laser_path = laser_dir / f"{frame_base}_laser.png"

        if not laser_path.exists():
            print(f"⚠️ Missing laser frame for: {frame_base}")
            continue

        print(f"🔄 Processing: {frame_base}")

        # === Load Images ===
        line_img = cv2.imread(str(line_path))
        laser_img = cv2.imread(str(laser_path))
        if line_img is None or laser_img is None:
            print(f"❌ Failed to read image for {frame_base}")
            continue

        # === Line Mask & Filter ===
        gray_line = rgb_to_grayscale(line_img)
        _, binary_mask_line = cv2.threshold(gray_line, 60, 255, cv2.THRESH_BINARY)
        filtered_mask = np.zeros_like(binary_mask_line)

        lines = cv2.HoughLinesP(binary_mask_line, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle, length = get_line_angle_and_length(x1, y1, x2, y2)
                if length > 200:
                    cv2.line(filtered_mask, (x1, y1), (x2, y2), 255, thickness=2)
        else:
            print(f"⚠️ No lines detected for {frame_base}, using fallback.")
            filtered_mask = binary_mask_line.copy()

        # === Fit Line ===
        line_params = fit_line_from_binary_mask(filtered_mask)
        if not line_params:
            print(f"✅ Skipping: {frame_base}")
            continue
        
        slope, intercept = line_params

        # === Save Fitting Image ===
        fitting_img = draw_line_on_image(filtered_mask, line_params)
        cv2.imwrite(str(fitting_dir / f"{frame_base}_fitting.png"), fitting_img)

        # === Laser Detection ===
        gray_laser = rgb_to_grayscale(laser_img)
        laser_center = compute_laser_center(gray_laser)
        if not laser_center:
            print(f"✅ Skipping: {frame_base}")
            continue

        distance = signed_perpendicular_distance(laser_center, line_params) if laser_center else None

        # === Save Laser+Line Overlay ===
        overlay_img = draw_linelaser_on_image(np.zeros_like(gray_laser), line_params, laser_center)
        cv2.imwrite(str(processed_dir / f"{frame_base}_processed.png"), overlay_img)

        # === Log CSV ===
        log_result_to_csv(csv_output_path, frame_base, laser_center, slope, intercept, distance)

        print(f"✅ Done: {frame_base}")

    # except Exception as e:
    #     print(f"❌ Error in frame {frame_base}: {e}")
    #     continue

#print(f"\n✅ All frames processed.\nOutputs saved to:\n📁 {fitting_dir}\n📁 {processed_dir}\n📄 {csv_output_path}")
