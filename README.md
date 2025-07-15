## Project Overview

Neuro‐degenerative changes linked to **Alzheimer’s disease (AD)** begin years before clinical diagnosis.  
This repository explores whether a **single baseline T1-weighted MRI** already contains enough information to:

1. distinguish cognitively-normal (CN) controls from prodromal AD / late-MCI, and  
2. provide an interpretable biomarker pipeline that can be reproduced on commodity hardware.

Key points
- **Dataset** ADNI (300 + subjects) → BIDS → bias-field corrected, skull-stripped, MNI-aligned.  
- **Models** (1) light 3-layer 3-D CNN baseline; (2) **ResNet18-3D** trained on CPU;  
  (3) fine-tuned 2-class ResNet18-3D on Google Colab GPU (AMP mixed precision).  
- **Results** Best model reaches **74 % test accuracy / 0.85 F1**, but still misses CN cases  
  ⇢ ongoing work on class balancing, focal-loss and multimodal fusion.  

> *Goal: push fully automated AD screening closer to routine clinical use, while keeping the codebase  
> transparent, lightweight and reproducible.*
