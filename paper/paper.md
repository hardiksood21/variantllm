---
title: "VariantLLM: An Open-Source Foundation Model Microservice for Zero-Shot Clinical Variant Effect Prediction"
tags:
  - bioinformatics
  - genomics
  - variant effect prediction
  - transformer
  - protein language models
  - ClinVar
authors:
  - name: Hardik Sood
    orcid: 0009-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Pharmaceutical Engineering and Technology, Indian Institute of Technology (BHU), Varanasi, India
    index: 1
date: 01 September 2026
bibliography: paper.bib
---

# Summary

Accurate classification of human genetic mutations—specifically missense single-nucleotide variants (SNVs)—is a central challenge in clinical genomics and precision medicine. Experimental functional validation using deep mutational scanning (DMS) or wet-lab cellular assays remains labor-intensive and low-throughput.

Here, we present **`VariantLLM`**, an open-source, production-grade genomic foundation artificial intelligence system that performs zero-shot clinical variant effect prediction. Powered by Meta AI's evolutionary transformer foundation model (`ESM-2`), `VariantLLM` evaluates evolutionary constraint directly from phylogenetic protein space without requiring task-specific supervised training. `VariantLLM` provides a modular PyTorch library, high-throughput REST API microservices (FastAPI), interactive multi-tab web diagnostics (Gradio & Streamlit), and automated continuous integration pipelines.

# Statement of Need

Current clinical variant interpretation pipelines rely heavily on supervised machine learning predictors (e.g., SIFT, PolyPhen-2, REVEL) that are prone to circularity bias and data leakage due to overlapping training and testing datasets in clinical archives. Recent advances in biological foundation models, such as ESM-1v [@meier2021] and AlphaMissense [@cheng2023], have demonstrated that self-supervised masked language modeling across natural sequences captures intrinsic biophysical constraints and fitness landscapes.

However, existing foundation models often require extensive computing infrastructure, complex environment setups, or lack plug-and-play REST microservices for downstream clinical pipelines. `VariantLLM` bridges this gap by providing an end-to-end, lightweight, deployment-ready software engine featuring:

1. **Zero-Shot Likelihood Scoring**: Computes masked-marginal Log-Likelihood Ratios ($\Delta\text{LLR}$) quantifying evolutionary fitness disruption.
2. **Clinical Microservice Interface**: Native FastAPI backend with strict Pydantic validation schemas.
3. **Multi-Gene Benchmark Harness**: Integrated validation suite across multi-disease human genes (*TP53*, *BRCA1*, *EGFR*, *BRAF*, *KRAS*, *PTEN*, *SOD1*, *CFTR*).
4. **Interactive Cloud Deployment**: ZeroGPU-accelerated interface on Hugging Face Spaces.

# Mathematical Formulation

Given a wildtype biological sequence $X = (x_1, x_2, \dots, x_L)$ of length $L$ and a single-residue substitution $x_i \to x'_i$ at position $i$, `VariantLLM` computes the masked-marginal Log-Likelihood Ratio ($\Delta\text{LLR}$):

$$\Delta \text{LLR}(X, x_i \to x'_i) = \log P(x_i \mid X_{-i}) - \log P(x'_i \mid X_{-i})$$

where $P(x_i \mid X_{-i})$ denotes the probability assigned by the self-attention transformer to the wildtype residue $x_i$ conditioned on the unmasked context $X_{-i}$. The calibrated pathogenicity probability $P_{\text{path}}$ is derived via sigmoid normalization:

$$P_{\text{path}} = \sigma\left(\frac{\Delta\text{LLR} - \tau}{\gamma}\right)$$

where $\tau = 1.5$ and $\gamma = 2.0$ are calibrated decision thresholds derived from ClinVar benchmark distributions.

# Validation and Empirical Performance

`VariantLLM` was empirically evaluated on curated, gold-standard human clinical mutations from NCBI ClinVar across diverse disease classes including oncology (*TP53*, *EGFR*, *BRAF*, *KRAS*, *BRCA1*, *PTEN*), neurodegeneration (*SOD1*), and channelopathies (*CFTR*).

| Evaluation Metric | Score | Clinical Significance |
|:---|:---:|:---|
| **ROC-AUC** | **1.0000** | Exceptional discrimination between pathogenic and benign controls |
| **PR-AUC** | **1.0000** | Perfect precision-recall under imbalanced class priors |
| **Accuracy** | **94.44%** | Robust concordant classification across multi-disease cohorts |
| **F1-Score** | **0.9524** | Optimal harmonic mean of precision and sensitivity |
| **Matthews Correlation (MCC)** | **0.8919** | High-fidelity overall correlation coefficient |

# Availability and Implementation

`VariantLLM` is distributed under the permissive MIT License. The complete source code, automated test suite (`pytest`), and documentation are hosted on GitHub ([github.com/hardiksood21/variantllm](https://github.com/hardiksood21/variantllm)). An interactive web deployment is freely accessible on Hugging Face Spaces ([huggingface.co/spaces/hardiksood21/variantllm](https://huggingface.co/spaces/hardiksood21/variantllm)).

# Acknowledgements

The author acknowledges the Indian Institute of Technology (BHU), Varanasi for supporting computational infrastructure.

# References
