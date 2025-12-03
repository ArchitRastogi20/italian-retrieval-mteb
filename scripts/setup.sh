#!/bin/bash

echo "=========================================="
echo "MTEB Italian Retrieval Evaluation"
echo "=========================================="
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q mteb pandas tqdm torch sentenceTransformers --break-system-packages

echo ""
echo "Setup complete!"
echo ""
echo "=========================================="
echo "To start the evaluation:"
echo "  python3 evaluation/run_mteb_eval.py"
echo ""
echo "To parse existing JSON results:"
echo "  python3 evaluation/parse_results.py"
echo ""
echo "Results will be saved to:"
echo "  - results/<model_name>/results.json"
echo "  - results/italian_retrieval_results.csv"
echo "=========================================="
