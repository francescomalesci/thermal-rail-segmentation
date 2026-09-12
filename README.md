# Thermal Rail Segmentation & Object Detection Pipeline

Computer Vision pipeline designed for automated infrastructure segmentation and anomaly detection on thermal/infrared video streams. 
This project integrates Deep Learning models to extract track masks and identify obstacles in low-visibility environments.

Developed as a Bachelor's Thesis project in Computer Engineering.

## 🎯 Architecture & Tech Stack
* **Goal:** High-accuracy segmentation of railway tracks and bounding box localization of potential hazards.
* **Core Models:** 
  * Fine-tuned **Ultralytics YOLO** for fast, real-time object detection.
  * **SAM 2 (Segment Anything Model 2)** with a custom adapter, fine-tuned for thermal domain prompt-based mask generation.
* **Stack:** Python, PyTorch, OpenCV, NumPy, CUDA.

---

## 📸 Demo & Results

### 1. Spatial AND Logic 
The pipeline successfully merges YOLO bounding boxes with SAM 2 segmentation masks. Objects detected *inside* the track mask are flagged as **CRITICAL**, while objects *outside* are flagged as **Ignored**.

![Static Inference Demo](assets/demo_spatial.jpg)

### 2. Zero-Shot Real-World Inference (WIP)
Testing the pipeline on out-of-distribution (OOD) field video streams.

![Field Video Demo](assets/demo_video.gif)

### YOLO Training Metrics
![YOLO Training Results](assets/yolo_results.png)

#### 🔍 Engineering Analysis & Known Limitations
The model demonstrates high precision on the validation set, but encounters specific edge cases during zero-shot real-world video inference:
* **Domain Shift:** The field video contains environmental variations and thermal noise not present in the training distribution.
* **Temporal Consistency:** Inference is currently performed frame-by-frame, causing minor flickering in the segmentation masks.
* **Next Steps:** Enhancing track segmentation accuracy and improving anomaly detection precision in complex scenarios.

---

## 📁 Repository Structure
```text
thermal-rail-segmentation/
├── assets/                   # Demo images and GIFs for documentation
├── configs/                  # YAML configuration files for training/inference
├── data/sample/              # Sample thermal images and ground truth masks
├── src/                      # Source code (Custom SAM2 Adapter & YOLO scripts)
│   ├── train.py              # Custom training loop with RandomErasing & MultiStepLR
│   ├── batch_inference.py    # E2E inference script (Detection + Segmentation)
│   ├── datasets.py           # Dataloader logic
│   └── wrappers.py           # Dataset wrappers with custom augmentations
└── README.md
```
---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Testa97/thermal-rail-segmentation.git
cd thermal-rail-segmentation
pip install -r requirements.txt
```
*Note: This project requires the official [SAM 2 repository](https://github.com/facebookresearch/segment-anything-2) for base weights and core architecture.*

### 2. Run Batch Inference
Test the end-to-end pipeline on the provided sample data:
```bash
python src/batch_inference.py \
  --image_dir data/sample/images \
  --mask_dir data/sample/labels \
  --weights models/best.pt \
  --output_dir results_batch/
```

### 3. Custom Training
To run the SAM2 adapter fine-tuning loop with custom augmentations:
```bash
python src/train.py --config configs/thermal-rail-sam2.yaml --name my_custom_training
```

---

## ⚙️ Key Technical Highlights
* **VRAM Optimization:** Forced `.cpu()` offloading during validation loops to prevent CUDA Out Of Memory errors.
* **Backbone Freezing:** Frozen SAM2 image encoder to retain zero-shot generalization while training only the custom adapter.
* **Custom Augmentations:** Implemented robust preprocessing and `RandomErasing` to prevent overfitting on small domain-specific datasets.
