
# Learning-Augmented Lazy Search Trees

This repository contains a implementation of a **Lazy Search Tree (LST)**, augmented using a **Machine Learning Oracle**. The project explores the intersection of deferred and learning-augmented data structures. Our goal is to compare this new structure against traditional BSTs, in a variety of benchmarking scenarios  

## Overview

The **Lazy Search Tree** is a data structure that minimizes insertion costs by delaying sorting until query time. This implementation extends the original concept by using a **Learning-Augmented Treap** as the top-level gap structure. 

### Key features:
- **Lazy Sorting:** Only organizes data when queried, breaking the $O(\log n)$ insertion barrier.
- **ML-Augmented Priorities:** Uses a Neural Network (Oracle) to predict the Cumulative Distribution Function of query workloads.
- **C++/Python Hybrid:** High-performance computational core in C++ with an intuitive Python orchestration layer via `pybind11`.
- **Robustness Mechanism:** Implements a heuristic threshold to prevent performance collapse under adversarial workloads.


## Installation & Build

### Prerequisites

* C++17 Compiler (GCC/Clang/MSVC)
* Python 3.11
* PyTorch (for Oracle training)
* `pybind11`

### Build the C++ Modules

Execute the following command in the root directory to compile the C++ structures into Python-accessible modules:

```bash
python setup.py build_ext --inplace
```

## Benchmarking

The evaluation framework supports both synthetic and real-world datasets.

### Running Experiments

1. **Train the Oracle:** Generate weights for a specific workload.
```bash
python src/learned/cdf_oracle.py
```


2. **Run Benchmarks:** Compare the classic LST and the Learning-Augmented LST against standard Trees (Splay, Treap, B-Tree).

```bash
python run_real_tests.py
python run_synthetic_tests.py
```



## Key Findings

- **Memory Efficiency:** The LST architecture achieves up to **75x reduction** in memory footprint compared to `std::set` due to aggressive interval merging.
- **Throughput:** Outperforms traditional Trees in scenarios where insertions dominate queries.
- **Resilience:** The threshold mechanism ensures the structure maintains $O(\log n)$ worst-case performance even with an adversarial Oracle.

## References

* Sandlund, B., & Wild, S. (2020). Lazy Search Trees. arXiv. https://doi.org/10.48550/arXiv.2010.08840
* Chen, J., Cao, X., Stepin, A., & Chen, L. (2025). On the Power of Learning-Augmented Search Trees. arXiv. https://doi.org/10.48550/arXiv.2211.09251
* Lin, H., Luo, T., & Woodruff, D. P. (2022). Learning Augmented Binary Search Trees. arXiv. https://doi.org/10.48550/arXiv.2206.12110
* Rysgaard, C. M., & Wild, S. (2025). Towards Lazy B-Trees. LIPIcs, Volume 345, MFCS 2025, 345, 87:1-87:19. https://doi.org/10.4230/LIPIcs.MFCS.2025.87

---
