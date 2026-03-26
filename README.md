# Machine Learning Interatomic Potential with 3-Body Interactions

> Extending pairwise potentials to include angular (3-body) contributions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project demonstrates **3-body (angular) interactions** in machine learning interatomic potentials, extending beyond simple pairwise (2-body) models.

### Why 3-body?

| Potential Type | Description | Captures |
|----------------|-------------|----------|
| **2-body** (LJ, Morse) | Depends only on pairwise distances r_ij | Repulsion, dispersion |
| **3-body** (Stillinger-Weber, Tersoff) | Includes angular terms θ_ijk | Bond bending, directionality |
| **Many-body** (EAM, MEAM) | Local environment dependence | Coordination effects |

**Real materials often require 3-body terms for accuracy:**
- Silicon (tetrahedral bonding)
- Carbon (graphene, diamond)
- Water (hydrogen bonding)

---

## 🔬 Toy Model

We construct a simplified 3-body potential:

  - $$V_{\text{total}} = \sum_{i<j} V_{\text{LJ}}(r_{ij}) + \sum_{ijk} V_{\text{angle}}(\theta_{ijk})$$

where:

- **2-body term**: Standard Lennard-Jones potential
  $$V_{\text{LJ}}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

- **3-body term**: Angular (bond-bending) potential
  $$V_{\text{angle}}(\theta) = K (\cos\theta - \cos\theta_0)^2$$

**Parameters:**
- K = 0.5 ε (angular stiffness)
- θ₀ = 60° (equilibrium angle)

---

## 📊 Results

### Dataset

| Property | Value |
|----------|-------|
| Configurations | 3,000 |
| Atoms per config | 10 |
| 2-body energy range | [-4.95, 8.01] ε |
| 3-body energy range | [0.00, 16.57] ε |
| Force range | [-134, 134] ε/σ |

### Model Performance

| Metric | Value |
|--------|-------|
| Energy MAE | 1.22 ε |
| Energy R² | 0.61 |

**Note:** This is a minimal demonstration using only pairwise distances. Full 3-body force matching requires:

- Behler-Parrinello symmetry functions (radial + angular descriptors)
- Message-passing neural networks (SchNet, DimeNet)
- Equivariant architectures (NequIP, MACE)

---

## 📁 Project Structure

```
mlip-lennard-jones-3body/
├── README.md
├── data/
│   ├── generate_data_3body.py    # Generate 2-body + 3-body dataset
│   └── 3body_dataset.npz          # 3000 configurations (not in git)
│
├── model/
│   └── nn_potential_3body.py      # 3-body neural network architecture
│
├── training/
│   ├── train_3body_v2.py          # Full training with autograd
│   └── train_3body_minimal.py     # Minimal demo (energy-only)
│
└── results/
    ├── 3body_data_dist.png        # Data distribution
    └── 3body_minimal_results.png  # Training results
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Vead-YI/mlip-lennard-jones-3body.git
cd mlip-lennard-jones-3body
pip install numpy torch matplotlib
```

### 2. Generate Data

```bash
cd data
python generate_data_3body.py
```

### 3. Train Model

```bash
cd ../training
python train_3body_minimal.py
```

---

## 🔮 Future Work

For production-grade 3-body potentials:

| Direction | Tool/Framework | Difficulty |
|-----------|----------------|------------|
| **Symmetry Functions** | Behler-Parrinello descriptors | ⭐⭐ |
| **Message Passing** | SchNet, DimeNet, PaiNN | ⭐⭐⭐ |
| **Equivariant Models** | NequIP, MACE, Allegro | ⭐⭐⭐⭐ |
| **DFT Integration** | VASP, Quantum ESPRESSO data | ⭐⭐⭐ |

### Recommended Next Steps

1. **Implement angular descriptors** — Add Behler-Parrinello angular symmetry functions
2. **Force matching** — Train on forces via autograd through descriptors
3. **Real materials** — Apply to Si (Stillinger-Weber) or H₂O (TIP4P + ML correction)

---

## 📚 References

1. **Behler, J. & Parrinello, M.** (2007). Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. *Phys. Rev. Lett.*, 98, 146401.

2. **Stillinger, F. H. & Weber, T. A.** (1985). Computer simulation of local order in condensed phases of silicon. *Phys. Rev. B*, 31, 5262.

3. **Schütt, K. et al.** (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *NeurIPS*.

4. **Batzner, S. et al.** (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 4373.

---

## 📝 Related Projects

- [mlip-lennard-jones](https://github.com/Vead-YI/mlip-lennard-jones) — 2-body MLIP project

---

*This is an educational project demonstrating the transition from 2-body to 3-body potentials. For production MLIP research, consider established frameworks like AMPTORCH, NequIP, or MACE.*
