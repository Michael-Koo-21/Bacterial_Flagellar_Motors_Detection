# Bacterial Flagellar Motors Detection

## Project Overview

This repository contains a solution for the BYU Locating Bacterial Flagellar Motors 2025 Kaggle competition. The goal is to develop an algorithm that automatically identifies the presence and 3D location of flagellar motors in cryogenic electron tomography (cryo-ET) reconstructions of bacteria.

### The Challenge

Flagellar motors are molecular machines that enable bacterial motility. While cryo-ET allows visualization of these structures in near-native conditions, identifying them in tomograms is challenging due to:

- Poor signal-to-noise ratio (negative SNR values)
- Variable motor orientations
- Crowded intracellular environments
- Variable tomogram sizes
- Motors appearing as darker regions than surroundings

### Our Approach

We implement an ensemble approach that combines:

1. **2D YOLOv8 Detection**: Processes 2D slices to identify potential motor candidates
2. **3D CNN Validation**: Analyzes 3D subvolumes around candidates to confirm detections and refine positions
3. **Ensemble Decision Making**: Combines confidence scores from both models for final prediction

## Full Training Pipeline

```
training-notebook-mk-byu-flagellar-motor-detection.ipynb
```

## Submission Notebook

```
submission-mk-byu-flagellar-motor-detection.md           
```

## Installation and Requirements

### Dependencies

```bash
# Core dependencies
pip install numpy pandas matplotlib opencv-python Pillow tqdm
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics optuna

# For Kaggle environment (offline installation)
# Pre-downloaded dependencies can be loaded with:
kagglehub.dataset_download('rachiteagles/yolo-pkg')
kagglehub.dataset_download('michaelkoo21/flagellar-motor-model-2')
pip install --no-index --no-deps /kaggle/input/yolo-pkg/yolo/ultralytics-8.3.112-py3-none-any.whl
```

### Hardware Requirements

- GPU with at least 8GB VRAM is recommended for training
- The inference pipeline is optimized to run within Kaggle's 12-hour time limit

## Data Preprocessing Pipeline

Our preprocessing workflow includes:

1. **Slice Extraction**: Extract 2D slices containing motors with context (±5 slices)
2. **Adaptive Normalization**: Combine percentile-based contrast enhancement and adaptive histogram equalization
3. **Annotation Creation**: Generate YOLO-compatible bounding box annotations
4. **Train/Validation Split**: Split at the tomogram level (not at the slice level) to prevent data leakage

```python
# Key parameters
CONTEXT_RANGE = 5     # Number of slices above/below motors to include
BOX_SIZE = 24         # Size of bounding box for annotations
TEST_SPLIT = 0.2      # Proportion of tomograms for validation
NORM_METHOD = 'adaptive'  # Normalization method
```

## Model Training

### YOLOv8 Training

The YOLOv8 model is trained on the 2D slices with the following optimizations:

- Transfer learning from pre-trained weights
- Hyperparameter optimization using Optuna
- Early stopping based on DFL loss
- Data augmentation (rotations, flips, contrast adjustments)

```python
# Hyperparameters optimized via Optuna
PRETRAINED_MODEL = "yolov8n.pt"  # or "yolov8s.pt", "yolov8m.pt"
BATCH_SIZE = 16                  # Range: 8-32
IMG_SIZE = 640                   # Range: 640-960
OPTIMIZER = "AdamW"              # Options: "AdamW", "SGD"
LEARNING_RATE = 1e-4             # Range: 1e-5 to 1e-3
BOX_GAIN = 7.5                   # Range: 5.0-10.0
CLS_GAIN = 0.5                   # Range: 0.3-1.0
DFL_GAIN = 1.5                   # Range: 1.0-2.0
```

### 3D CNN Training

The 3D CNN model incorporates an attention mechanism to focus on motor-specific features:

- Input: 3D subvolumes (64×64×64) centered around potential motors
- Architecture: 3D convolutions with attention block and regression head
- Output: Binary classification (motor presence) and location refinement

```python
# 3D CNN parameters
SUBVOLUME_SIZE = 64      # Size of cubic subvolumes
DROPOUT_RATE = 0.3       # Dropout rate for training
EPOCHS = 20              # Number of training epochs
LEARNING_RATE = 1e-4     # Initial learning rate
```

## Inference Pipeline

The inference pipeline processes test tomograms with these steps:

1. Process each tomogram slice-by-slice with YOLO to identify potential motor locations
2. Perform 3D non-maximum suppression to merge nearby detections
3. Extract 3D subvolumes around candidate locations
4. Validate candidates using the 3D CNN model
5. Generate the final submission with motor coordinates or (-1, -1, -1) for tomograms without motors

```python
# Inference parameters
YOLO_CONFIDENCE_THRESHOLD = 0.30  # Initial filtering threshold
CNN_CONFIDENCE_THRESHOLD = 0.45   # 3D CNN validation threshold  
NMS_IOU_THRESHOLD = 0.2           # For 3D non-maximum suppression
BATCH_SIZE = 8                    # Dynamically adjusted based on GPU memory
```

## Performance Optimization

The solution incorporates several optimizations to ensure efficiency:

- **GPU Memory Management**: Dynamic batch sizing based on available GPU memory
- **Parallel Processing**: CUDA streams for concurrent slice processing
- **Preloading**: Batch preloading while processing current batch
- **Half-Precision Inference**: FP16 computation on compatible GPUs
- **3D Clustering**: Efficient non-maximum suppression for merging detections

## Usage Instructions

### Training Pipeline

To run the complete training pipeline:

```bash
# Clone the repository
git clone https://github.com/yourusername/flagellar-motor-detection.git
cd flagellar-motor-detection

# Run the training notebook
python training-notebook-mk-byu-flagellar-motor-detection.py
```

### Submission Generation

To generate a submission file:

```bash
# Run the submission notebook
python submission-mk-byu-flagellar-motor-detection.py
```

## Results Visualization

The solution provides visualization tools for:

- Tomogram slices with motor annotations
- Comparison of normalization methods
- Model predictions on random validation samples
- Detection results on test data

## Evaluation Metric

Submissions are evaluated using the F₂-score with a 1000Å distance threshold:

- **True Positive (TP)**: Prediction within 1000Å of ground truth
- **False Negative (FN)**: Motor exists but prediction beyond threshold or missed
- **False Positive (FP)**: Prediction exists but no motor present
- **F₂ = (1+2²)·TP/((1+2²)·TP+2²·FN+FP)**

## Acknowledgments

- BYU Locating Bacterial Flagellar Motors 2025 Kaggle competition organizers
- Ultralytics for YOLOv8 implementation
- Kaggle for computing resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.