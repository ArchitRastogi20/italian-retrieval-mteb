# Italian Retrieval Embeddings

Fine-tuning multilingual embedding models for Italian text retrieval using contrastive learning.

## Overview

This project investigates whether language-specific fine-tuning can improve Italian retrieval performance for small multilingual embedding models. Our key finding reveals that MTEB leaderboard scores aggregate across all languages, masking that models like `multilingual-e5-small` already achieve 90+ NDCG@10 on Italian-specific tasks.

**Dataset**: We release 25M Italian retrieval triplets derived from mC4, available at:  
https://huggingface.co/datasets/ArchitRastogi/it-retrieval-triplets-mc4

## Project Structure

```
italian-retrieval-embeddings/
├── configs/                 # Model lists and configurations
├── data/
│   ├── generation/          # Triplet generation scripts
│   └── preprocessing/       # Data preparation utilities
├── training/                # Fine-tuning scripts
├── evaluation/              # MTEB and custom benchmark evaluation
├── analysis/                # Dataset EDA and statistics
├── scripts/                 # Shell scripts for running experiments
├── results/                 # Evaluation outputs
└── docs/                    # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch, sentence-transformers, mteb

## Dataset

The Italian triplet dataset contains:
- 8.6M triplets with hard negatives (mined via multilingual-e5-small)
- 16.8M triplets with random negatives
- Total: 25.4M unique triplets

Statistics:
- Query length: mean 17 tokens
- Positive length: mean 47 tokens
- Negative length: mean 43 tokens

## Usage

### Generate Triplets

```bash
python data/generation/generate_triplets.py
```

### Train Model

```bash
python training/train_contrastive.py \
    --model intfloat/multilingual-e5-small \
    --train_file path/to/train.jsonl \
    --batch_size 192 \
    --lr 5e-6
```

### Evaluate on MTEB Italian

```bash
python evaluation/run_mteb_eval.py
```

### Evaluate on Custom Benchmark

```bash
python evaluation/evaluate_bm25.py
python evaluation/rag_benchmark.py
```

## Results

### MTEB Italian Retrieval (NDCG@10)

| Model | MTEB Overall | Belebele IT | Wiki IT | Avg Italian |
|-------|--------------|-------------|---------|-------------|
| Qwen3-Embedding-8B | 90.4 | 98.7 | 92.1 | 95.4 |
| Qwen3-Embedding-4B | 86.2 | 95.3 | 89.8 | 92.5 |
| multilingual-e5-large | 84.2 | 95.1 | 92.3 | 93.7 |
| multilingual-e5-small | 77.0 | 92.4 | 89.3 | 90.8 |

Key finding: The 118M parameter multilingual-e5-small achieves 90.8 Italian NDCG@10, not 77 as the overall MTEB score suggests.

### Fine-tuning Results

Fine-tuning on Italian triplets caused catastrophic forgetting:
- multilingual-e5-small: NDCG dropped from 0.488 to 0.434
- e5-small-v2: NDCG dropped from 0.437 to 0.209

## Hardware

Experiments conducted on NVIDIA RTX 3090.

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{rastogi2025italian,
  title={Italian Retrieval Embeddings: When MTEB Scores Mislead},
  author={Rastogi, Archit},
  year={2025},
  howpublished={Sapienza University of Rome, DLAI Course Project}
}
```

## License

This project is released for academic and research purposes.

## Acknowledgments

- MTEB benchmark: https://github.com/embeddings-benchmark/mteb
- mC4 dataset: https://huggingface.co/datasets/allenai/c4
- Sentence Transformers: https://www.sbert.net/
