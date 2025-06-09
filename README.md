
<h1 align="center">
    <img alt="Feature Detection and Matching Demo" src="Readme/demo.gif" />
</h1>

<h1 align="center">Feature Detection and Matching</h1>
<div align="center">
<!--   <img src="https://github.com/user-attachments/assets/ae9b3bb0-2d64-4f06-9c63-3a02bded5e16" > -->
  
  <img src="https://github.com/user-attachments/assets/69f4fc84-8a93-4dab-94f3-118a0305e117">


</div>

<h4 align="center"> 
	Status: ✅ Completed
</h4>

<p align="center">
 <a href="#about">About</a> •
 <a href="#features">Features</a> •
 <a href="#tech-stack">Tech Stack</a> •  
 <a href="#developers">Developers</a>
</p>

---

## 🧠 About

The **Feature Detection and Matching** project implements advanced computer vision techniques for identifying key points in images and matching them between different views. It includes implementations of Harris corner detection, λ- corner detection, and SIFT feature extraction with matching capabilities.

This tool serves as a foundational platform for tasks in:
- Object recognition
- Image stitching
- 3D reconstruction
- Visual tracking

---

## ✨ Features

### 🔍 Corner Detection
- **Harris Corner Detection**
  - Gradient computation using Sobel operator
  - Second-moment matrix assembly
  - Corner response calculation with adjustable sensitivity
  - Non-maxima suppression (local and distance-based)
 
  <div align="center">
  <img src="https://github.com/user-attachments/assets/29d3fab0-98bb-46fa-a25b-473f4198f4d0">
  </div>
- **λ- Corner Detection**
  - Gaussian smoothing for noise reduction
  - Structure tensor computation
  - Minimum eigenvalue calculation
  - Comparison with OpenCV implementation


<div align="center">
<!-- ![image](https://github.com/user-attachments/assets/ebbbc6aa-999f-4511-a04a-20e2524ebae3) -->

  <img src="https://github.com/user-attachments/assets/ebbbc6aa-999f-4511-a04a-20e2524ebae3">
</div>

### 🗝️ Feature Extraction (SIFT)
- Scale-space construction with Gaussian pyramid
- Keypoint detection in Difference of Gaussian (DoG) space
- Keypoint refinement with subpixel accuracy
- Orientation assignment
- 128-dimensional descriptor computation

**SIFT Pipeline:**
<div align="center">
  <img src="https://github.com/user-attachments/assets/38ef121b-9d70-4bf1-a9a2-ec50344f79ba"  height="30%">
</div>

### 🤝 Feature Matching
- **Sum of Squared Differences (SSD)**
  - Fast Euclidean distance calculation
  - Lowe's ratio test for match validation
  
- **Normalized Cross-Correlation (NCC)**
  - Illumination-invariant matching
  - More accurate but computationally intensive

<div align="center">
  <img src="https://github.com/user-attachments/assets/ec1c9369-79d0-414f-a84b-c2c5b68397f2" >
  <img src="https://github.com/user-attachments/assets/b22b2ff7-69d3-43b9-8c3d-7d9f2e2d92ab" >
</div>

### 🔍 Lowe's Ratio Test
- Improved match reliability
- Adjustable threshold for precision/recall tradeoff
- Reduces false positives

---

## ⚙️ Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **SciPy**

---

## Developers

| [**Talal Emara**](https://github.com/TalalEmara) | [**Meram Mahmoud**](https://github.com/Meram-Mahmoud) | [**Maya Mohammed**](https://github.com/Mayamohamed207) | [**Nouran Hani**](https://github.com/Nouran-Hani) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|
---
---

## 📎 Learn More

* [Harris Corner Detection](https://en.wikipedia.org/wiki/Harris_Corner_Detector)
* [Shi-Tomasi Corner Detector](https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html)
* [SIFT Algorithm](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
* [Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
