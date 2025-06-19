# Simplicial Attention Neural Network with Applications to Trajectory Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/Damas200/sann-experiments)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Code Structure](#code-structure)
- [Results](#results)
- [Visualization](#visualization)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview
This repository contains the implementation and documentation for the **Simplicial Attention Neural Network (SANN)**, a novel deep learning architecture introduced in the thesis *Simplicial Attention Neural Network with Applications to Trajectory Prediction* by Damas Niyonkuru at the African Institute for Mathematical Sciences (AIMS) Rwanda. SANN leverages attention mechanisms and simplicial complexes to model higher-order interactions in complex systems, outperforming traditional graph neural networks in tasks involving group dynamics. The project evaluates SANN on four datasets: co-authorship networks (2-simplex and 3-simplex), ocean drifter trajectories, and synthetic flows, achieving test AUCs/accuracies of 97.8%, 97.1%, 98.4%, and 96.2%, respectively. The codebase includes Python scripts for data preprocessing, model training, and visualization, alongside a LaTeX chapter detailing the experiments. This work is dedicated to Damas' parents, Dr. Olakunle S. Abawonse, and Aime Barema for their support and guidance.

Key features:
- **2-Simplex Experiments**: Classifies three-author collaborations with a test AUC of 0.978.
- **3-Simplex Experiments**: Classifies four-author collaborations with a test AUC of 0.971.
- **Ocean Drifters Experiments**: Classifies ocean current edges (2011–2018) with a test accuracy of 0.9844.
- **Synthetic Flow Experiments**: Classifies synthetic flow edges with a test accuracy of 0.962.
- Visualizations of Ocean Drifters and Synthetic Flow simplicial complexes, highlighting topological patterns.

## Installation
To run the code and reproduce the experiments, follow these steps:

### Prerequisites
- Python 3.8+
- LaTeX distribution (e.g., TeX Live) for compiling the documentation
- Git for cloning the repository

### Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

Example `requirements.txt` (included in the repository):
```
torch>=1.10.0
torch-geometric>=2.0.0
pytorch-lightning>=1.5.0
optuna>=2.10.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

### Clone the Repository
```bash
git clone https://github.com/Damas200/sann-experiments.git
cd sann-experiments
```

## Usage
The repository includes scripts for preprocessing, training SANN models, and generating visualizations. Below are instructions for key tasks.

### Running Visualizations
To generate the Ocean Drifters visualization (Figure 4.1 in the thesis):
```bash
python scripts/visualize_ocean_drifters.py
```
- **Input**: `data/coords.pkl` (node coordinates), `data/incidence.pkl` (incidence matrices).
- **Output**: `figures/ocean_drifters.png`, showing 133 nodes, 320 edges, triangles (blue, alpha=0.4), edges 10–25 (thick black with arrowheads), and node 70 (blue circle).
- **Requirements**: `matplotlib`, `numpy`, `scipy`.

To generate the Synthetic Flow visualization (Figure 4.2):
```bash
python scripts/visualize_synthetic_flow.py
```
- **Output**: `figures/synthetic_flow.png`, showing ~100–200 nodes, 527 edges, triangles, and highlighted edges (thick blue with arrowheads) near topological holes.
- **Requirements**: Same as above.

### Training SANN Models
To train SANN on the Ocean Drifters dataset:
```bash
python scripts/train_san.py --dataset ocean_drifters --seed 42
```
- **Arguments**:
  - `--dataset`: `ocean_drifters`, `synthetic_flow`, `2_simplex`, or `3_simplex`.
  - `--seed`: Random seed (e.g., 42 for Ocean Drifters/Synthetic Flow, 12091996 for 2/3-simplex).
- **Output**: Checkpoints and logs in `outputs/`.

To train on the 2-simplex dataset:
```bash
python scripts/mainscc.py --dataset 2_simplex --seed 12091996
```
- **Output**: Checkpoints and logs in `outputs/`.

### Compiling the LaTeX Documentation
To compile the experiments chapter:
```bash
cd docs
latexmk -pdf thesis_chapter4.tex
```
- **Output**: `thesis_chapter4.pdf` in `docs/`.
- **Requirements**: LaTeX with `amsmath`, `booktabs`, `graphicx`, `listings`, `xcolor`.

## Dataset Details
The datasets are stored in `data/` and described below:
- **2-Simplex (Co-authorship)**: 352 nodes, 1474 edges, 3285 triangles, binary labels ($Y=1$ if collaboration strength $\leq 7$). Files: `data/s2_3_collaboration_complex/{150250}/{5}_boundaries.npy`, `{5}_cochains.npy`.
- **3-Simplex (Co-authorship)**: Extends to 502 tetrahedra. Same file structure, with `B_3` (shape: [3285, 502]).
- **Ocean Drifters**: 133 nodes, 320 edges, triangles, with 314 negative and 6 positive edge labels. Files: `data/data.pkl`, `data/incidence.pkl`, `data/coords.pkl`.
- **Synthetic Flow**: ~100–200 nodes, 527 edges, with 520 negative and 7 positive edge labels. Same file structure as Ocean Drifters.

Due to size or proprietary constraints, raw datasets may require preprocessing using provided scripts (e.g., `scripts/train_san.py`, `scripts/mainscc.py`). Contact the repository owner for access to raw data.

## Code Structure
```
sann-experiments/
├── data/                   # Datasets (e.g., data.pkl, incidence.pkl, coords.pkl)
│   └── s2_3_collaboration_complex/  # Co-authorship data
├── scripts/                # Python scripts
│   ├── train_san.py        # Training for Ocean Drifters/Synthetic Flow
│   ├── mainscc.py          # Training for 2/3-simplex
│   ├── net.py              # SANN model architecture
│   ├── sdataset_scc_copy.py # Dataset class
│   ├── triangoli.py        # Triangle/tetrahedron indices
│   ├── visualize_ocean_drifters.py  # Ocean Drifters visualization
│   └── visualize_synthetic_flow.py  # Synthetic Flow visualization
├── docs/                   # LaTeX documentation
│   └── thesis_chapter4.tex  # Experiments chapter
├── figures/                # Output visualizations (e.g., ocean_drifters.png)
├── outputs/                # Model checkpoints and logs
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Results
SANN demonstrates robust performance across tasks:
- **2-Simplex**: Test AUC 0.978, training AUC 0.995, validation AUC 0.985 (Table 4.1).
- **3-Simplex**: Test AUC 0.971, training AUC 0.986, validation AUC 0.987 (Table 4.2).
- **Ocean Drifters**: Test accuracy 0.9844, training AUC 0.867, validation AUC 1.000 (Table 4.3).
- **Synthetic Flow**: Test accuracy 0.962, training AUC 0.864, validation AUC 1.000 (Table 4.4).

Limitations include class imbalance ([314, 6] for Ocean Drifters, [520, 7] for Synthetic Flow), potential overfitting (zero training losses in 2/3-simplex), and CPU-only training. See `docs/thesis_chapter4.pdf` for detailed analysis.

## Visualization
- **Ocean Drifters (Figure 4.1)**: Visualizes 133 nodes, 320 edges, and triangles (semi-transparent blue, alpha=0.4). Edges 10–25 are thick black with arrowheads, and node 70 is a blue circle, highlighting topological flow patterns. Generated by `scripts/visualize_ocean_drifters.py`.
- **Synthetic Flow (Figure 4.2)**: Shows ~100–200 nodes, 527 edges, and triangles in a Delaunay triangulation. Highlighted edges (thick blue with arrowheads) trace trajectories near topological holes, reflecting high attention weights. Generated by `scripts/visualize_synthetic_flow.py`.

These visualizations underscore SANN’s focus on critical topological features, aligning with high test accuracies.



## Contact
For questions or dataset access, contact Damas Niyonkuru at [damas.niyonkuru@aims.ac.rw](mailto:damas.niyonkuru@aims.ac.rw).

## Acknowledgments
- **Dr. Olakunle S. Abawonse** (supervisor) for guidance in simplicial complexes and neural networks.
- **Aime Barema** (tutor) for significant support.
- AIMS Rwanda faculty and staff for research support.
- Damas’ family, especially parents, for encouragement.
- Peers at AIMS Rwanda for fellowship and mathematical discussions.
- Open-source libraries: PyTorch, PyTorch Geometric, Optuna, Matplotlib, scikit-learn.
