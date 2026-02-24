# AGENTS.md - AI Agent Interaction Guide

This document provides guidelines for AI agents (like GitHub Copilot, Claude, or other LLMs) to effectively assist with development on this project.

## Project Overview

**Project Name:** reid-soft-attributes  
**Language:** Python 3.12  
**Framework:** PyTorch 2.10.0 + torchvision 0.25.0  
**Package Manager:** uv  

### Technology Stack
- **Core ML:** PyTorch, torchvision
- **Data:** numpy, scipy, OpenCV, PIL
- **Visualization:** matplotlib, seaborn, tensorboard
- **Code Quality:** ruff (linting and formatting)

---

## Guidelines for AI Agents

### Code Style and Quality

1. **Python Version:** Ensure all code is compatible with Python 3.12+
2. **Type Hints:** Use type annotations for function parameters and returns
3. **Code Formatting:** All Python code must pass `ruff` checks
4. **Comments:** Use comments only when really needed, don't overflow the code with useless comments.
5. **Quotes:** Always use doubles quotes `"` for strings instead of single quotes `'`.

**Quality Assurance Command:**
```bash
uv run ruff check --fix
```

After each code modification, this command MUST be executed to ensure:
- Code formatting compliance
- No unused imports
- No style violations

### PyTorch and Deep Learning

1. **Device Handling:** Always consider GPU/CPU compatibility
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Data Loading:** Use PyTorch's `DataLoader` with appropriate batch sizes
3. **Model Architecture:** Follow the multi-branch architecture pattern (APN, APAN, fusion)
4. **Loss Functions:** Document the loss computation clearly

### Import Organization

Organize imports in the following order:
1. Standard library (os, sys, etc.)
2. Third-party libraries (torch, numpy, etc.)
3. Local project imports (from src.*)

Example:
```python
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from src.Model.model import GlobalReIDNetwork
from src.data_treatment.dataset import ReIDDataset
```

### Testing and Validation

1. **Experiment Scripts:** Use `experiments/` folder for validation and ablation studies
2. **Checkpoints:** Store intermediate checkpoints with epoch numbers for reproducibility
3. **Metrics Logging:** Use tensorboard and structured logging in `runs/` directory

---

## Business Logic and Domain Knowledge

### Project Purpose and Principles

This project implements a hybrid deep learning architecture for Person Re-Identification (Re-ID), merging task-based quality assessment with multi-scale semantic feature fusion. The following research papers form the core of this implementation.

#### Task-Based Image Quality Assessment (IDA)

Reference: H. Chen, E. J. Delp and A. R. Reibman, "Estimating Image Quality for Person Re-Identification," 2021 IEEE 23rd International Workshop on Multimedia Signal Processing (MMSP), Tampere, Finland, 2021, pp. 1-6, doi: 10.1109/MMSP53017.2021.9733688.

##### Purpose and Goal

The primary objective of this paper is to move beyond generic image quality metrics (like PSNR or SSIM) toward a task-specific metric. The authors introduce the concept of "Identifiability" (IDA): an unsupervised measure that quantifies how useful a specific query image is for a Re-ID system. It identifies whether an image is "challenging" or "reliable" before it enters the matching pipeline.

##### Implementation Details

The IDA measure is built on the principle of Feature Consistency.
- Unsupervised Approach: No ground truth labels are required for quality estimation.
- Perturbation Analysis: The system applies synthetic perturbations (e.g., noise, blur, or compression) to a query image.
- Consistency Score (Inverted Scale): By comparing the feature vector of the original image with the vectors of its perturbed versions, the model calculates a stability score.
    - Low IDA Score: High feature consistency, indicating a high-quality, reliable image for identification.
    - High IDA Score: High variance/instability under perturbations, indicating a low-quality, challenging image.

##### Core Techniques to Retain

- Task-Specific Quality: Quality is defined by the stability of the feature extractor, not just pixels.
- Robustness Ranking: Use IDA to rank query images; images with high scores (low quality) can be flagged for manual review or assigned lower weights in a fusion system.
- Compression Resilience: The metric effectively predicts how much additional compression an image can withstand before the Re-ID performance drops significantly.

#### Multi-Scale Pyramid Attention (MSPA)

Reference: S. U. Khan et al., "Visual Appearance and Soft Biometrics Fusion for Person Re-Identification Using Deep Learning," in IEEE Journal of Selected Topics in Signal Processing, vol. 17, no. 3, pp. 575-586, May 2023, doi: 10.1109/JSTSP.2023.3260627.

##### Purpose and Goal

This paper addresses the limitations of standard Re-ID methods that rely solely on identity labels. It proposes the Multi-Scale Pyramid Attention (MSPA) model, which jointly learns Visual Appearance (global identity) and Soft Biometrics (semantic attributes like gender, clothing type, or accessories) to create a more descriptive and discriminative pedestrian representation.

##### Implementation Details

The architecture utilizes a dual-stream approach based on a ResNet-50 backbone:
- Attribute Pyramid Attention Network (APAN): Uses an Attribute Pyramid Attention Network combined with a ConvLSTM. This allows the model to capture the spatial and semantic relationships between different body parts and their attributes.
- Appearance Pyramid Network (APN): Focuses on extracting global contextual features.
    - Multi-Feature Pyramid Module (MFPM): Employs dilated convolutions at different rates to capture multi-scale context, ensuring that both fine-grained details and global structures are preserved.

##### Core Techniques to Retain

- Dual-Stream Fusion: Combining low-level identity features with high-level semantic attributes (Soft Biometrics).
- ConvLSTM for Spatial Context: Using Recurrent Neural Networks on feature maps to maintain a "memory" of how attributes relate to each other spatially.
- Local Attention Module (LAM): A mechanism to focus on specific regions of interest, reducing the impact of background noise or occlusions.
- Pyramid Structures: Utilizing multiple scales to handle the varied resolutions and distances of pedestrians in surveillance footage.

#### Our Contribution: Quality-Aware Dynamic Fusion

Project Title: Quality-Aware Person Re-Identification for Improved Fusion of Visual and Soft-Biometric Cues, Dataset: Market-1501

Our implementation and evaluation are centered on the Market-1501 dataset, one of the most widely used benchmarks for Person Re-ID. It contains 32,668 images of 1,501 identities captured by six different cameras. This dataset provides the necessary complexity (varied viewpoints, poses, and backgrounds) to test the robustness of our quality-aware fusion strategy.

##### What Makes This Contribution Interesting?

The main innovation of this project lies in the active arbitration between visual appearance and semantic attributes. While traditional Re-ID systems often perform static fusion (concatenation), our architecture acknowledges that visual data quality is highly variable in real-world surveillance.

By introducing a third branch dedicated to Image Quality Assessment (IQA), our model can "decide" in real-time which feature stream to trust more. This creates a "Quality-Aware" system that maintains high performance even when visual data is degraded by blur, occlusion, or poor lighting.

##### Concrete Implementation Features

1. Triple-Branch Architecture:
- Visual Stream (APN): Captures the global appearance.
- Attribute Stream (APAN): Extracts soft biometrics (clothing, gender, etc.).
- IQA Branch: Computes the Identifiability Score using the unsupervised perturbation method.

2. Dynamic Weighting Mechanism:
- We implemented a lightweight MLP (Multi-Layer Perceptron) that takes the Identifiability score as input.
- This MLP outputs a fusion weight ($\alpha$) which dynamically balances the contributions of the APN and APAN features during the final inference.

3. Adaptive Robustness:
- When the IQA branch detects a high score (low quality/high instability), it increases the weight of the Attribute branch, as semantic descriptors (e.g., "red shirt", "backpack") often remain more reliable than pixel-level visual signatures in low-resolution or blurred scenarios.
- Evaluation on Market-1501 demonstrates improved mAP and Rank-1 metrics compared to standard fusion baselines.

##### Summary of the Integrated Pipeline

Our contribution transforms the passive MSPA model into an active, self-correcting system where the Identifiability score acts as a confidence signal, ensuring that the fusion of visual and soft-biometric cues is always optimized for the specific conditions of the query image within the Market-1501 environment.

---

## Development Workflow

### Documentation Requirements

When modifying or creating code, ensure:
- **Module-level docstrings:** Explain what the module does
- **Function signatures:** Include type hints and return type
- **Complex logic:** Add inline comments explaining the approach
- **References:** Cite papers or techniques used

---

## Environment and Dependencies

### Managing Dependencies with uv

Add new dependencies via `pyproject.toml` and initialize:
```bash
uv sync
```

Update dependencies:
```bash
uv update
```

### Key Versions to Respect

- Python: `>=3.12`
- PyTorch: `2.10.0`
- torchvision: `0.25.0`
- ruff: `>=0.15.0`

Ensure all changes remain compatible with these versions.

---

## Error Handling and Debugging

### Common Issues and Solutions

1. **Import Errors:** Check that paths in `from src.*` imports match actual file structure
2. **Tensor Shape Mismatches:** Review forward pass dimensions in model.py
3. **Device Errors:** Ensure `.to(device)` is called on all models and tensors

### Logging and Output

- Use `print()` for debugging during development with `flush=True` parameter for better debugging
- Use `tensorboard` for metric visualization
- Save experiment results to `experiments/visualizations/`

---

## AI Agent Response Protocol

When assisting with this project, AI agents should:

1. **Verify Dependencies:** Check if new functionality requires new imports
2. **Maintain Consistency:** Follow existing code patterns and naming conventions
3. **Quality Assurance:** Always end with `uv run ruff check --fix` verification
4. **Document Changes:** Add or update docstrings explaining modifications
5. **Check Compatibility:** Ensure code works with Python 3.12 and specified package versions
6. **Consider the Architecture:** Understand how components integrate before suggesting changes

---

## Resources and References

- **PyTorch Documentation:** https://pytorch.org/docs/stable/
- **torchvision Documentation:** https://pytorch.org/vision/stable/
- **ReID Datasets:** Market-1501
- **Ruff Documentation:** https://docs.astral.sh/ruff/

---

## Notes for Continuous Development

- Keep `AGENTS.md` updated as project evolves
- Document new architectural patterns as they emerge
- Update "Business Logic" section as methodology evolves
- Maintain consistency with Python 3.12 and specified dependencies
