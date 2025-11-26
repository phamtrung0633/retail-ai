## Autonomous shopping system - The code of this repo is under victor_branch

# Haurus AI 

Haurus AI builds high-precision computer vision systems for frictionless retail. Our MVP combines multi-camera visual understanding with low-cost radio-frequency signals to track items, reduce queue friction, and enable cashierless checkout while maintaining auditability for retailers.

**We were awarded the UQ Ventures Validate $1K prize for our early technical progress and validation work.**

### Why this matters

Buying friction: Queues and checkout friction cost retailers billions each year.
Shrink/theft: Existing self-checkout solutions suffer from mis-scans and shrink.
Labour shortages: Retailers face chronic staffing gaps; automation must be precise and trustworthy.
Our approach focuses on measurable accuracy, low-latency inference, and retailer-grade reliability.

### Technical Overview

#### System Architecture

At a high level, the system combines:

Multiple cameras providing visual observations of products and interactions.
Radio-frequency signals providing corroborative item presence/movement cues.
These signals are fused to detect product removal and returns with high accuracy.

#### ML Stack

Framework: PyTorch.
Vision Models: Object detection and interaction reasoning to identify pick/place events.
Multi-camera support to improve robustness in real environments.
Training Pipeline

Datasets: Curated retail scenarios containing item pick/place events, occlusions, and crowd motion. Synthetic augmentation (domain randomization, lighting, pose).
Annotation: Bounding boxes, keypoints/hand-object interactions, and event timestamps; semi-automatic labeling tooling to accelerate iteration.
Training: PyTorch Lightning-style loops for reproducibility; early stopping and automated hyperparameter sweeps.
Validation: Scenario-based evaluation (crowding, occlusions, reflective packaging) plus real-world pilot footage.
Inference Pipeline (Edge)

Multi-camera frame ingest with time sync.
Batched detector inference (PyTorch) with per-stream pre-processing.
Short-term association (IoU/appearance embedding) across frames.
Cross-camera stitching using geometry + re-id embeddings.
RF corroboration to disambiguate visually ambiguous events.
Event emission: pick/place, basket add/remove, anomaly flags.
Hardware & Deployment

Cameras: Multiple overhead/shelf cameras.
RF: Low-cost passive infrastructure.
Engineering Highlights

Hardware + Embedded Systems: Designed and implemented embedded architectures for autonomous checkout and delivery systems. Integrated real-time weight sensing hardware with our perception stack to corroborate visual events.
Sensor Fusion with Predictive ML: Collected and streamed weight telemetry in sync with video; fused signals with PyTorch models to improve precision on product removal and return events.
Low-Latency Infrastructure: Built high-speed, wireless-first edge-to-server pipelines to move data and model outputs reliably with minimal latency.
High-Accuracy Event Detection: Achieved strong accuracy on pick/place detection, enabling trustworthy cashierless workflows with auditable trails.
Patentability & Advantage: Architected a system with defensible advantages for certain retailer formats; components are being prepared for patent protection.
Bootstrapped Efficiency: Delivered a functional MVP and pilot readiness on a $1,000 budget (UQ Ventures Validate prize), emphasizing pragmatic engineering and rapid iteration.
