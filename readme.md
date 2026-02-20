# Merge Kernel for Bayesian Optimization on Permutation Spaces (ICLR 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-b31b1b.svg)](https://openreview.net/forum?id=7QtKdabBP9)

Official implementation of the ICLR 2026 paper: **"From Sorting Algorithms to Scalable Kernels: Bayesian Optimization in High-Dimensional Permutation Spaces"**.

This repository provides the code for **MergeBO**, a novel framework that leverages the divide-and-conquer structure of Merge Sort to produce a compact, $\Theta(n \log n)$ feature representation for Bayesian Optimization on permutation spaces, breaking the computational bottleneck of the $\Omega(n^2)$ Mallows kernel.

---

## 📋 Table of Contents
- [Dependencies](#-dependencies)
- [Repository Structure](#-repository-structure)
- [Usage & Reproducing Benchmarks](#-usage--reproducing-benchmarks)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🛠 Dependencies

> **⚠️ Important Notes on Dependencies:** 
>**Core Packages:** The core implementation relies on `torch`, `gpytorch`, and `botorch`. For GPU acceleration, ensure your PyTorch installation matches your system's CUDA version.
> * **High-Dimensional Benchmarks (TTP):** The Traveling Thief Problem (TTP) benchmark is implemented in Java. To run the TTP experiments, you **must have Java (JRE/JDK) installed on your system**, along with the `JPype1` package in your Python environment to enable the Python-Java bridge.


```bash
# Clone the repository
git clone [https://github.com/YourUsername/MergeBO.git](https://github.com/YourUsername/MergeBO.git)
cd MergeBO

# Create and activate the conda environment
conda create -n mergebo python=3.11
conda activate mergebo

# Install requirements
pip install -r requirements.txt

```
---

## 📂 Repository Structure

```text
MergeBO/
├── cell_placement/            # Cell Placement (n=30) benchmark
├── floorplanning/             # Floor Planning (n=30) benchmark
│   ├── floorplan_mallows.py   # Mallows kernel implementation on FP
│   ├── floorplan_mallows.sh   # Sbatch submit for batch running on FP (Mallows kernel)
│   └── ...                    # Other kernel implementations, including: Merge, random_selection, spearsman, turbo
├── qap_synthetic/             # QAP (n=15) benchmark
├── tsp_synthetic/             # TSP (n=15) benchmark
├── MergeKernelCPP-master/     # Traveling Thief Problem (TTP, n=280). JAVA required!
│   └── main.py                # Running relaxed TTP experiments with: Merge kernel, Mallows kernel, turbo and random selection
└── requirements.txt           # Dependencies

```


---

## 🚀 Usage & Reproducing Benchmarks

The experiments are modularized into specific folders based on the problem dimensionality and domain. To run an experiment or reproduce the paper's results, navigate to the corresponding directory and execute the runner script.

### Low-Dimensional Benchmarks (e.g., TSP, QAP, Floor Planning)

Navigate to the `low_dim` directory to evaluate the kernels on standard combinatorial tasks (corresponding to Section 4.1.1 in the paper):

```bash
cd tsp_synthetic  # or qap_synthetic, etc.

# Example: Run MergeBO on the TSP (n=15) instance for 20 trials
python tsp_merge.py  # or tsp_mallows.py, etc.
```

### High-Dimensional Benchmarks (Traveling Thief Problem)

For the large-scale TTP benchmarks (n=280) detailed in Section 4.1.2, navigate to the `high_dim` directory:

```bash
cd MergeKernelCPP-master
python main.py
```

---

## 📝 Citation

If you find our work or this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{xie2026merge,
  title={From Sorting Algorithms to Scalable Kernels: Bayesian Optimization in High-Dimensional Permutation Spaces},
  author={Xie, Zikai and Chen, Linjiang},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
---

## 🙏 Acknowledgements


**Funding & Resources:** The computational resources for this work were provided by the robotic AI-Scientist platform of the Chinese Academy of Sciences (CAS). We also acknowledge the State Key Laboratory of Precision and Intelligent Chemistry at the University of Science and Technology of China (USTC).

**Codebase & Benchmarks:** Our implementation is built upon and heavily relies on the following excellent open-source works. We sincerely thank the authors for making their code and datasets available:

* **BOPS-H (Core Baseline):** The core Bayesian Optimization pipeline, local search strategy, and the baseline Mallows kernel are built upon the official repository of [BOPS](https://github.com/aryandeshwal/BOPS/), from the AAAI 2022 paper *"Bayesian Optimization over Permutation Spaces"* by Deshwal et al.
* **TTP Environments:** The Java implementation and dataset instances for the Traveling Thief Problem are adapted from the official distribution of the GECCO 2014 paper *"A Comprehensive Benchmark Set and Heuristics for the Traveling Thief Problem"* by Polyakovskiy et al. ([Source code & data zip](https://cs.adelaide.edu.au/~optlog/research/ttp/TTP-JavaForDistribution.zip)).

