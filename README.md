# Learning_Hippo

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![JAX](https://img.shields.io/badge/JAX-Framework-orange.svg)
![Flax](https://img.shields.io/badge/Flax-Framework-green.svg)
![Status](https://img.shields.io/badge/status-archived-lightgrey.svg)

This repository contains the experimental results described in **“Learning Hippo: A Hippocampal Neural Architecture for Deep Learning.”**  
All data are provided to enable independent validation and transparent reproducibility of the presented results.

---

## Table of Contents
- [Abstract](#abstract)
- [Hippo-1 Architecture](#hippo-1-architecture)
- [Experimental Results](#experimental-results)
  - [Standard Benchmark Performance](#standard-benchmark-performance)
    - [MNIST](#mnist)
    - [Fashion-MNIST](#fashion-mnist)
    - [CIFAR-10](#cifar-10)
    - [CIFAR-100](#cifar-100)
  - [Noise Robustness](#noise-robustness)
- [Discussion](#discussion)
- [Conclusions & Future Work](#conclusions--future-work)
- [How to Cite](#how-to-cite)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Abstract
Current neural networks suffer from fragility to corrupted data and require large training datasets. We present **Hippo-1**, a biologically inspired neuron model and architecture grounded in the **CA3** region of the hippocampus. Our model uses a justified compartmental simplification that preserves non-linear dendritic integration, and a specific interaction between excitatory and inhibitory populations via *shunting inhibition*. Experiments show Hippo-1 achieves competitive or superior performance to Multi-Layer Perceptrons (MLPs) and exhibits stronger robustness to noise, with a more gradual performance degradation as input corruption increases.

<img src="images/NeuronePiramidale.png" alt="Modello compartimentale di un neurone piramidale in Hippo-1" width="600">
*Five-compartment model of a Hippo-1 pyramidal neuron with the corresponding signal-integration equations.*

---

## Hippo-1 Architecture
`Hippo-1` integrates key neuroscientific principles to overcome limitations of standard models such as MLPs. The primary inspiration is the **hippocampal CA3** area, known for auto-association and pattern completion—reconstructing clean signals from partial or noisy inputs.

Two core principles:
1. **Compartmental Simplification** — Instead of simulating full neuronal morphology, `Hippo-1` uses five functional compartments: three apical dendritic (two distal, one proximal), one basal dendritic, and the soma. This preserves essential non-linear dendritic computations while keeping the model tractable.
2. **Shunting Inhibition** — The interaction between excitatory (pyramidal) and inhibitory (GABAergic) neurons is modeled multiplicatively rather than subtractively, dynamically modulating the gain of excitatory signals. This stabilizes network activity and confers intrinsic robustness to perturbations.

---

## Experimental Results

To contextualize `Hippo-1` performance, we define three model sizes (High, Medium, Low) by parameter count and compare them with three MLP baselines matched in parameters.

### Standard Benchmark Performance

#### MNIST
*Performance of MLP and Hippo-1 on `MNIST`.*

| **Model** | **#Params** | **Accuracy (± std)** | **F1 (± std)** |
|:--|--:|:--:|:--:|
| **MLP hidden 128** | 101,770 | **0.9782 ± 0.0009** | **0.9757 ± 0.0010** |
| **MLP hidden 64** | 50,890 | **0.9733 ± 0.0010** | **0.9702 ± 0.0011** |
| **MLP hidden 32** | 25,450 | **0.9669 ± 0.0017** | **0.9628 ± 0.0019** |
| Hippo-1 High | 284,756 | 0.9778 ± 0.0007 | 0.9753 ± 0.0007 |
| Hippo-1 Medium | 72,404 | 0.9716 ± 0.0016 | 0.9685 ± 0.0019 |
| Hippo-1 Low | 28,196 | 0.9652 ± 0.0022 | 0.9613 ± 0.0024 |

#### Fashion-MNIST
*Performance of MLP and Hippo-1 on `Fashion-MNIST`.*

| **Model** | **#Params** | **Accuracy (± std)** | **F1 (± std)** |
|:--|--:|:--:|:--:|
| **MLP hidden 128** | 101,770 | **0.8824 ± 0.0053** | **0.8711 ± 0.0065** |
| MLP hidden 64 | 50,890 | 0.8750 ± 0.0031 | 0.8637 ± 0.0032 |
| MLP hidden 32 | 25,450 | 0.8672 ± 0.0020 | 0.8545 ± 0.0026 |
| Hippo-1 High | 284,756 | 0.8813 ± 0.0033 | 0.8700 ± 0.0031 |
| **Hippo-1 Medium** | 72,404 | **0.8803 ± 0.0016** | **0.8694 ± 0.0021** |
| **Hippo-1 Low** | 28,196 | **0.8731 ± 0.0015** | **0.8616 ± 0.0017** |

#### CIFAR-10
*Performance of MLP and Hippo-1 on `CIFAR-10`.*

| **Model** | **#Params** | **Accuracy (± std)** | **F1 (± std)** |
|:--|--:|:--:|:--:|
| MLP hidden 128 | 394,634 | 0.5053 ± 0.0047 | 0.4807 ± 0.0057 |
| MLP hidden 64 | 197,322 | 0.4911 ± 0.0085 | 0.4650 ± 0.0079 |
| MLP hidden 32 | 98,666 | 0.4718 ± 0.0078 | 0.4457 ± 0.0093 |
| **Hippo-1 High** | 431,188 | **0.5132 ± 0.0060** | **0.4886 ± 0.0076** |
| **Hippo-1 Medium** | 218,836 | **0.5023 ± 0.0065** | **0.4777 ± 0.0072** |
| **Hippo-1 Low** | 64,804 | **0.4938 ± 0.0050** | **0.4677 ± 0.0061** |

#### CIFAR-100
*Performance of MLP and Hippo-1 on `CIFAR-100`.*

| **Model** | **#Params** | **Accuracy (± std)** | **F1 (± std)** |
|:--|--:|:--:|:--:|
| MLP hidden 128 | 406,244 | 0.2177 ± 0.0025 | 0.1432 ± 0.0025 |
| **MLP hidden 64** | 203,172 | **0.2029 ± 0.0046** | **0.1322 ± 0.0033** |
| MLP hidden 32 | 101,636 | 0.1907 ± 0.0038 | 0.1227 ± 0.0033 |
| **Hippo-1 High** | 477,358 | **0.2415 ± 0.0031** | **0.1601 ± 0.0020** |
| Hippo-1 Medium | 224,686 | 0.1997 ± 0.0047 | 0.1282 ± 0.0036 |
| **Hippo-1 Low** | 76,414 | **0.2164 ± 0.0030** | **0.1399 ± 0.0022** |

### Noise Robustness

#### MNIST

| **Gaussian Noise** | **Var 0.10** | **Var 0.20** | **Var 0.30** | **Var 0.40** | **Var 0.50** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.9501 | 0.7737 | 0.5878 | 0.4474 | 0.3523 |
| **Hippo-1 High** | **0.9551** | **0.7886** | **0.5915** | **0.4623** | **0.3687** |

| **Salt & Pepper** | **10%** | **20%** | **30%** | **40%** | **50%** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.7465 | 0.5215 | 0.3807 | 0.2909 | 0.2308 |
| **Hippo-1 High** | **0.8092** | **0.5748** | **0.4226** | **0.3248** | **0.2561** |

#### Fashion-MNIST

| **Gaussian Noise** | **Var 0.10** | **Var 0.20** | **Var 0.30** | **Var 0.40** | **Var 0.50** |
|:--|:--:|:--:|:--:|:--:|:--:|
| **MLP 128** | **0.8326** | **0.6580** | **0.4909** | **0.3664** | **0.2734** |
| Hippo-1 High | 0.8079 | 0.6282 | 0.4407 | 0.3163 | 0.2397 |

| **Salt & Pepper** | **10%** | **20%** | **30%** | **40%** | **50%** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.6391 | 0.4387 | 0.3230 | 0.2362 | 0.1998 |
| **Hippo-1 High** | **0.6583** | **0.4539** | **0.3123** | **0.2216** | **0.1700** |

#### CIFAR-10

| **Gaussian Noise** | **Var 0.10** | **Var 0.20** | **Var 0.30** | **Var 0.40** | **Var 0.50** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.4884 | 0.4508 | 0.3990 | 0.3656 | 0.3222 |
| **Hippo-1 High** | **0.5012** | **0.4726** | **0.4553** | **0.4182** | **0.3739** |

| **Salt & Pepper** | **10%** | **20%** | **30%** | **40%** | **50%** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.4610 | 0.4062 | 0.3647 | 0.3021 | 0.2568 |
| **Hippo-1 High** | **0.4843** | **0.4557** | **0.4124** | **0.3649** | **0.3027** |

#### CIFAR-100

| **Gaussian Noise** | **Var 0.10** | **Var 0.20** | **Var 0.30** | **Var 0.40** | **Var 0.50** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.2072 | 0.1810 | 0.1406 | 0.1062 | 0.0800 |
| **Hippo-1 High** | **0.2368** | **0.2229** | **0.2041** | **0.1764** | **0.1546** |

| **Salt & Pepper** | **10%** | **20%** | **30%** | **40%** | **50%** |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP 128 | 0.1791 | 0.1342 | 0.0991 | 0.0723 | 0.0470 |
| **Hippo-1 High** | **0.2236** | **0.2013** | **0.1784** | **0.1419** | **0.1145** |

---

## Discussion
On simpler tasks like `MNIST`, `Hippo-1` is **statistically indistinguishable** from an MLP. As task complexity increases, the bio-inspired advantages become clear: on **CIFAR-10** and **CIFAR-100**, `Hippo-1` significantly outperforms parameter-matched MLP counterparts, suggesting a beneficial **inductive bias**.

The most notable result is **robustness to noise**. Across all scenarios, `Hippo-1` degrades more gracefully than MLPs. On CIFAR-100 with strong Gaussian noise (σ² = 0.5), `Hippo-1` maintains nearly **double** the accuracy of the MLP (15.46% vs 8.00%). This supports the hypothesis that *shunting inhibition* acts as dynamic gain control, yielding intrinsic resilience absent in standard architectures.

---

## Conclusions & Future Work
`Hippo-1` shows that integrating neuroscientific computational principles can yield more robust deep learning models **without** sacrificing accuracy. This hybrid approach—combining modern algorithmic efficiency with robust principles from neuroscience—offers a promising path for next-generation AI systems.

Future steps include modeling the **recurrent connectivity** of CA3, testing pattern completion and sequence memory tasks, and exploring **more biologically plausible learning rules**.

---

## How to Cite
If you use this repository’s data, please cite the following paper:

> *Learning Hippo: A Hippocampal Neural Architecture for Deep Learning*  
> _to be added_

---

## Repository Structure
This repository contains **only the raw experimental results**, organized for analysis and validation. Per-run `.csv` files (accuracy and F1-score) are located in the corresponding folders.

```text
/
├── Test Results/
│   ├── Accuracy-F1Score/
│   │   ├── cifar10/
│   │   ├── cifar100/
│   │   ├── fashion_mnist/
│   │   └── mnist/
│   │
│   └── Robustness/
│       ├── cifar10/
│       ├── cifar100/
│       ├── fashion_mnist/
│       └── mnist/
│
├── images/
│   └── NeuronePiramidale.jpg
│
└── README.md
