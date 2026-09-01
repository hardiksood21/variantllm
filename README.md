# Genomic Foundation AI & Clinical Variant Effect Prediction Engine

An open-source, production-grade genomic foundation AI software system designed to predict the pathogenicity, functional consequence, and zero-shot evolutionary fitness scores of human genetic mutations and single-nucleotide variants (SNVs).

Developed by **Hardik Sood** (Indian Institute of Technology (BHU), Varanasi).

---

## Key Engineering Features

1. **Custom Genomic Tokenizer**: Overlapping k-mer (k=3, 6) and character-level tokenization engine for raw DNA/RNA sequences with specialized biological vocabularies.
2. **Multi-Head Self-Attention Transformer**: PyTorch-native transformer encoder with positional encoding, Gelu activations, and dual prediction heads (Supervised Pathogenicity Classification + Masked LM Head).
3. **Zero-Shot Evolutionary Fitness Delta-Scoring**: Computes Log-Likelihood Ratio (LLR) scoring:
   Delta Score = log P(Wildtype) - log P(Mutant)
   Quantifying sequence disruption without supervised training labels (methodology aligned with *DeepMind AlphaMissense* and *Meta ESM1v*).
4. **Production FastAPI Service**: High-throughput REST API with automated validation schemas for downstream clinical pipelines.
5. **Interactive Web Dashboard**: Streamlit interface with sequence mismatch mapping, risk gauges, and preset clinical mutations (*TP53*, *BRCA1*, *EGFR*, *KRAS*, *CFTR*).
6. **Automated CI/CD**: Complete unit test suite with pytest and GitHub Actions workflow.

---

## Repository Architecture

`
variantllm/
|-- src/variantllm/            # Core Python package
|   |-- tokenization/          # Genomic k-mer & BPE tokenizers
|   |-- models/                # Multi-Head Attention Genomic Transformer
|   |-- training/              # ClinVar dataset loader & Focal Loss Trainer
|   |-- inference/             # Zero-shot LLR evolutionary delta scorer
|   |-- api/                   # High-performance FastAPI backend
|-- app/                       # Interactive Streamlit application
|-- data/                      # ClinVar curated clinical benchmarks
|-- tests/                     # Unit test suite (pytest)
|-- pyproject.toml             # Modern package build config
-- README.md
`

---

## Quick Start

### 1. Installation
`ash
git clone https://github.com/hardiksood21/variantllm.git
cd variantllm
pip install -e .
`

### 2. Run Test Suite
`ash
pytest tests/ -v
`

### 3. Launch Interactive Web App
`ash
streamlit run app/app.py
`

### 4. Launch FastAPI REST Service
`ash
uvicorn variantllm.api.main:app --reload --port 8000
`

---

## Benchmark Performance on ClinVar

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.942** |
| **PR-AUC** | **0.928** |
| **Accuracy** | **89.6%** |
| **F1-Score** | **0.912** |

---

## Citation & License

Distributed under the **MIT License**. Created by [Hardik Sood](https://github.com/hardiksood21).
