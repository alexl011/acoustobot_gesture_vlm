# COMP0234: MSc Systems Engineering for the Internet of Things Project
## A Gesture-Based Multi-Agent Visual Learning Model for Acoustophoretic Interactions using Swarm of AcoustoBots

### Overview

This project implements a real-time gesture recognition system using OpenCLIP (Contrastive Language-Image Pre-training) for controlling acoustophoretic interactions with AcoustoBots. The system combines computer vision and machine learning to enable intuitive gesture-based control. <a href="https://alexl011.github.io/acoustobot_gesture_vlm/UCL_SEIoT_MSc_Project_Gesture_Based_Multi_Agent.pdf">mypdf</a>

The project leverages a pre-trained Vision Transformer (ViT-B-32) from OpenCLIP and fine-tunes it with a linear probe classifier to recognize three distinct hand gestures: **thumbs up**, **fist**, and **palm**.



##  Installation & Setup

### Prerequisites

- Python 3.8+
- Webcam for data collection and real-time inference

### Environment Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd acoustobot_gesture_vlm
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv clip_env
   source clip_env/bin/activate  # On Windows: clip_env\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install torch torchvision torchaudio
   pip install open-clip-torch
   pip install opencv-python
   pip install matplotlib
   pip install pillow
   pip install onnx onnxruntime
   ```

---

## Dataset

### Current Dataset Statistics
- **Total Images**: 790
- **Gestures**:
  - **Fist**: 310 images
  - **Palm**: 275 images  
  - **Thumbs Up**: 205 images
- **Training Split**: 80% (632 images)
- **Validation Split**: 20% (158 images)

### Data Collection

To expand the dataset with your own gesture images:

```bash
cd openclip_gesture_vlm
python collect_data.py
```

**Instructions**:
- Position your hand in the camera frame
- Press `SPACEBAR` to capture an image
- Select the gesture type (0: Thumbs Up, 1: Fist, 2: Palm)
- Press `q` to quit

After collecting new data, regenerate annotations:
```bash
python create_annotations.py
```

---

## Training

### Quick Start Training

```bash
cd openclip_gesture_vlm
python train_gestures.py
```


### Visualizing Training Results

Generate training plots for a specific run:
```bash
python plot_metrics.py 2025-07-09_14-54-24
```

Compare multiple training runs:
```bash
python compare_runs.py
```

---

## Deployment

### 1. Single Image Classification

Classify a single gesture image:
```bash
python classify_gesture.py path/to/your/image.jpg
```


### 2. Real-time Webcam Classification

Launch real-time gesture recognition:
```bash
python realtime_classify.py
```


### 3. ONNX Model Export

Export the trained model to ONNX format for deployment:
```bash
python export_to_onnx.py
```

This creates `gesture_model.onnx` optimized for inference.

---

## Customization

### Adding New Gestures

1. **Collect Data**: Use `collect_data.py` with new gesture labels
2. **Update Annotations**: Modify `create_annotations.py` for new classes
3. **Retrain Model**: Run `train_gestures.py` with updated dataset
4. **Update Inference**: Modify class mappings in inference scripts


## References

- [OpenCLIP Repository](https://github.com/mlfoundations/open_clip)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
