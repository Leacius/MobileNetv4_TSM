# MobileNetv4_TSM

This project provides a **lightweight, modular video classification framework** that integrates the **Temporal Shift Module (TSM)** into a customized **MobileNetV4** backbone, an efficient architecture adapted from the official PyTorch MobileNetV3 to match the design of [MobileNetV4 (arXiv:2404.10518)](https://arxiv.org/abs/2404.10518).

## What’s Included

- **MobileNetV4-style backbone**: Adapted from the PyTorch MobileNetV3 implementation and modified to follow the basic structure of MobileNetV4.
- **Temporal Shift Module (TSM)**: Added simple TSM logic to introduce temporal modeling without changing parameter count.
- **Lightweight by design**: This setup is intended for small-scale video classification experiments, especially for short clips or resource-constrained scenarios.

## Environment Setup

We recommend using `conda` to manage dependencies:

```bash
conda create -n MobileNetv4_TSM python=3.11
conda activate MobileNetv4_TSM
pip install -r requirements.txt
