# Borr AI: Autonomous Checkout System (Formerly Haurus AI)

A production-grade computer vision and sensor fusion pipeline for checkout-free retail environments. This system tracks customers in 3D, detects product interactions, and identifies items using deep learning and weight sensors.

## 🚀 Core Features

- **Stereoscopic 3D Tracking**: Dual-camera person tracking using YOLOv8-pose and triangulation.
- **Autonomous Interaction Detection**: Automated "proximity event" detection when hands enter shelf zones.
- **Product Identification**: High-accuracy visual recognition using Swin Transformer V2 and Milvus vector database.
- **Sensor Fusion**: Correlation of visual events with wireless weight sensor data for transaction validation.
- **Hand Segmentation**: Deep learning-based hand isolation to improve product embedding quality.
- **Person Re-ID**: Persistent identity tracking across camera views using Locally-Aware Transformers (LATransformer).

## 🛠 Technical Architecture

### Computer Vision Pipeline
1. **Detection**: YOLOv8-pose for 17-point human skeleton extraction.
2. **Association**: Binary Integer Programming (BIP) solver for optimal cross-camera detection clustering.
3. **Segmentation**: U-Net based hand segmentation for clean product extraction.
4. **Embeddings**: Swin-V2 Large (384x384) feature extraction for product matching.

### Hardware Integration
- **Stereo Setup**: Calibrated pinhole camera pairs with full intrinsic/extrinsic modeling.
- **Smart Shelves**: Integration with wireless weight sensors via TCP/Serial.
- **Inference**: Optimized for CUDA-enabled GPUs with multiprocessing stream handling.

### Data Management
- **Milvus**: Cloud-native vector database for high-speed product SKU retrieval.
- **DVC**: Data Version Control for managing model artifacts and training datasets.
- **Baking**: Pre-computation system for rapid development on recorded datasets.

## 📁 Project Structure

- `3D-tracking/`: Main application logic including detection, tracking, and calibration.
- `embeddings/`: Product embedding generation and Milvus integration.
- `LATransformer/`: Locally-Aware Transformer models for person re-identification.
- `transforms/`: Image transformation and hand segmentation modules.
- `calib_data/`: Camera calibration matrices and distortion coefficients.
- `landing-page/`: React-based visualization and marketing interface.

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU
- Milvus instance (URI and Token required)

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Update `3D-tracking/params.yaml` with your model paths.
2. Set environment variables in `.env`:
   ```env
   MILVUS_URI=your_uri
   MILVUS_TOKEN=your_token
   PRODUCTS_COLLECTION=products
   ```

### Running the System
To start the tracking and detection pipeline:
```bash
python 3D-tracking/detection_and_tracking.py
```

## 📊 Visualization
Use the built-in 3D visualizer to inspect skeleton tracking:
```bash
python 3D-tracking/visualiser.py --points ./poses_3d.json
```

---
*Note: This project is part of a retail automation suite.*
