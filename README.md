# Laser-Guided UAV Image Processing Pipeline

This repository contains the **image processing pipeline** developed as part of the project  
*“Preliminary Design and Testing of Laser-Guided Control of Multicopters.”*  

The pipeline extracts a **geometric error metric** from onboard camera recordings by detecting a projected laser dot and a floor reference line. These metrics were used to evaluate UAV trajectory tracking performance against OptiTrack motion capture ground truth.

---

## 📌 Features


- **Line Segmentation & Fitting**
   - HSV color space conversion  
   - Contrast-Limited Adaptive Histogram Equalization (CLAHE)  
  - Detection of floor reference line  
  - Linear regression fit (slope & intercept stats available)  

- **Laser Dot Detection**  
  - Color thresholding in HSV  
  - Centroid extraction for each frame  

- **Error Metric**  
  - Signed perpendicular distance between dot and line (px → mm via scaling)  
  - Error statistics (RMSE, variance, clustering, etc.)

    
## 📌 Data 
https://drive.google.com/drive/folders/1P5tx9iqFu7t_IttuaA8hcS5E4xo7--91?usp=sharing 
