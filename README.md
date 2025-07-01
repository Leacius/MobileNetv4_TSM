# MobileNetv4_TSM

This project provides a **lightweight, modular video classification framework** that integrates the **Temporal Shift Module (TSM)** into a customized **MobileNetV4** backbone, an efficient architecture adapted from the official PyTorch MobileNetV3 to match the design of [MobileNetV4 (arXiv:2404.10518)](https://arxiv.org/abs/2404.10518).

- **MobileNetV4-style backbone**: Adapted from the PyTorch MobileNetV3 implementation and modified to follow the basic structure of MobileNetV4.
- **Temporal Shift Module (TSM)**: Added simple TSM logic to introduce temporal modeling without changing parameter count.
- **Lightweight by design**: This setup is intended for small-scale video classification experiments, especially for short clips or resource-constrained scenarios.

## Environment Setup

We recommend using `conda` to manage dependencies:

```bash
conda create -n MobileNetv4_TSM python=3.11
conda activate MobileNetv4_TSM
pip install -r requirements.txt
```

## Data Format
To train the `MobileNetv4_TSM` model, your dataset should follow this structure:

```
dataset_root/
├── video1.mp4
├── video2.mp4
└── ...
```
And a label file in the format:

```
video1 123 0
video2 146 1
...
```
Each line contains:
video_name (without .mp4) total_frame_count class_label

### Acknowledgements

This project is a side project that builds on the work of two research contributions:

- **Temporal Shift Module (TSM)**:  
  [GitHub](https://github.com/mit-han-lab/temporal-shift-module) | [Paper](https://arxiv.org/abs/1811.08383)

- **MobileNetV4**:  
  [Paper: "MobileNetV4: Universal CNNs for Mobile Vision"](https://arxiv.org/abs/2404.10518)
