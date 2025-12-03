#!/bin/bash

# Activate your venv if needed
# source venv/bin/activate

# Ensure sentence-transformers is installed
pip install --quiet sentence-transformers

# Test loading the model
python3 << 'EOF'
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer("BAAI/bge-m3")
    print("Model loaded successfully!")
    # quick test
    embeddings = model.encode(["Ciao mondo!"])
    print("Embedding shape:", embeddings.shape)
except Exception as e:
    print("Failed to load model:", e)
EOF
