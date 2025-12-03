#!/bin/bash

# Embedding Model Benchmarking Script
# Configure your benchmark parameters here

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data and model files
DATA_FILE="./data/your_evaluation_data.jsonl"  # Update with your data file path
MODELS_FILE="./models.txt"
OUTPUT_DIR="./benchmark_results"

# Text chunking parameters
CHUNK_SIZE=512  # Can be changed to test different chunk sizes: 256, 512, 1024
OVERLAP=50      # Can be changed to test different overlaps: 0, 25, 50, 100

# Retrieval parameters
TOP_K_VALUES="1 5 10"  # Space-separated list of top-k values to test

# Language and device settings
LANGUAGE="en"    # "en" for English, "zh" for Chinese
DEVICE="cuda"    # "cuda" for GPU, "cpu" for CPU

# OpenAI settings (for generation evaluation)
USE_OPENAI="--use_openai"  # Remove this flag to disable OpenAI evaluation
OPENAI_MODEL="gpt-4o"      # gpt-4o, gpt-4o-mini, etc.
OPENAI_VERSION="v2"        # v1 or v2

# Processing settings
PARALLEL_MODELS=""  # Add "--parallel_models" to enable parallel processing (limited by GPU memory)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Set your OpenAI API key (if using OpenAI evaluation)
export OPENAI_API_KEY="your-openai-api-key-here"
export BASE_URL=""  # Set if using a custom OpenAI endpoint

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# MAIN BENCHMARK EXECUTION
# ============================================================================

echo "======================================================================"
echo "EMBEDDING MODEL BENCHMARK"
echo "======================================================================"
echo "Data file: $DATA_FILE"
echo "Models file: $MODELS_FILE" 
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo "Overlap: $OVERLAP"
echo "Top-K values: $TOP_K_VALUES"
echo "Language: $LANGUAGE"
echo "Device: $DEVICE"
echo "======================================================================"

# Check if required files exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Please update the DATA_FILE variable with the correct path."
    exit 1
fi

if [ ! -f "$MODELS_FILE" ]; then
    echo "Error: Models file not found: $MODELS_FILE"
    echo "Please ensure models.txt exists in the current directory."
    exit 1
fi

# Install requirements (uncomment if needed)
# echo "Installing requirements..."
# pip install -r requirements.txt

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
fi

echo "Starting benchmark..."
echo "======================================================================"

# Run the benchmark
python evaluation/rag_benchmark.py \
    --models_file "$MODELS_FILE" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --overlap "$OVERLAP" \
    --top_k_values $TOP_K_VALUES \
    --language "$LANGUAGE" \
    --device "$DEVICE" \
    $USE_OPENAI \
    --openai_model "$OPENAI_MODEL" \
    --openai_version "$OPENAI_VERSION" \
    $PARALLEL_MODELS

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo "======================================================================"
    echo "BENCHMARK COMPLETED SUCCESSFULLY!"
    echo "======================================================================"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "- benchmark_results_*.csv (aggregated results)"
    echo "- detailed_results_*.json (detailed results)"
    echo "- results/*.json (individual model results)"
    echo "- logs/*.log (execution logs)"
    echo ""
    echo "You can now analyze the results using the CSV file or the analysis script."
    echo "======================================================================"
else
    echo "======================================================================"
    echo "BENCHMARK FAILED!"
    echo "======================================================================"
    echo "Check the logs in $OUTPUT_DIR/logs/ for error details."
    echo "Common issues:"
    echo "- CUDA out of memory: Try reducing batch size or using CPU"
    echo "- Model download failures: Check internet connection"
    echo "- OpenAI API issues: Verify API key and rate limits"
    echo "======================================================================"
    exit 1
fi

# ============================================================================
# ADDITIONAL BENCHMARK CONFIGURATIONS (EXAMPLES)
# ============================================================================

# Uncomment and modify the sections below to run additional benchmarks
# with different configurations

# # Test different chunk sizes
# echo "Running benchmark with different chunk sizes..."
# for CHUNK_SIZE in 256 512 1024; do
#     echo "Testing chunk size: $CHUNK_SIZE"
#     python evaluation/rag_benchmark.py \
#         --models_file "$MODELS_FILE" \
#         --data_file "$DATA_FILE" \
#         --output_dir "${OUTPUT_DIR}/chunk_${CHUNK_SIZE}" \
#         --chunk_size "$CHUNK_SIZE" \
#         --overlap "$OVERLAP" \
#         --top_k_values $TOP_K_VALUES \
#         --language "$LANGUAGE" \
#         --device "$DEVICE" \
#         $USE_OPENAI \
#         --openai_model "$OPENAI_MODEL" \
#         --openai_version "$OPENAI_VERSION"
# done

# # Test different overlap values
# echo "Running benchmark with different overlap values..."
# for OVERLAP in 0 25 50 100; do
#     echo "Testing overlap: $OVERLAP"
#     python evaluation/rag_benchmark.py \
#         --models_file "$MODELS_FILE" \
#         --data_file "$DATA_FILE" \
#         --output_dir "${OUTPUT_DIR}/overlap_${OVERLAP}" \
#         --chunk_size "$CHUNK_SIZE" \
#         --overlap "$OVERLAP" \
#         --top_k_values $TOP_K_VALUES \
#         --language "$LANGUAGE" \
#         --device "$DEVICE" \
#         $USE_OPENAI \
#         --openai_model "$OPENAI_MODEL" \
#         --openai_version "$OPENAI_VERSION"
# done

echo "All benchmarks completed!"