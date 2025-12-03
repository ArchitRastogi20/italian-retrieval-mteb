#!/bin/bash

# IR-Only Benchmark Configuration - OPTIMIZED FOR HIGH-PERFORMANCE GPU
# Optimized for 2x RTX 4090, 64 vCPU, 251GB RAM

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data file - UPDATE THIS with your actual file name
DATA_FILE="./data/evaluation_data_filtered.jsonl"  # Change to your file name

# Models and output
MODELS_FILE="./models.txt"
OUTPUT_DIR="./benchmark_results"

# Text chunking parameters
# CHUNK_SIZES=(128 256 512 1024)
# OVERLAPS=(25 50 100 200)
CHUNK_SIZES=(128)
OVERLAPS=(200)
# Retrieval parameters  
TOP_K_VALUES="1 5 10"  # Space-separated list

# Language and device settings
LANGUAGE="it"    # Italian language


# High-performance settings
# High-performance settings - OPTIMIZED FOR CPU
PARALLEL_MODELS="--parallel_models"  # Enable parallel processing
# In run_ir_benchmark.sh:
DEVICE="cuda"    # Change back to cuda for embeddings
# Keep the CPU thread exports for FAISS:
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export FAISS_NUM_THREADS=64

# Memory optimization
# And add GPU memory optimization:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# EXECUTION
# ============================================================================

echo "======================================================================"
echo "IR BENCHMARK - HIGH PERFORMANCE MODE"
echo "======================================================================"
echo "Data file: $DATA_FILE"
echo "Models file: $MODELS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo "Overlap: $OVERLAP"
echo "Top-K values: $TOP_K_VALUES"
echo "Language: $LANGUAGE"
echo "Device: $DEVICE"
echo "Parallel processing: ENABLED"
echo "======================================================================"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo " ERROR: Data file not found: $DATA_FILE"
    echo ""
    echo "Please:"
    echo "1. Run: python merge_data.py"
    echo "2. Verify: ls -la data/"
    echo "3. Check: head -n 1 data/evaluation_data.jsonl"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# GPU optimization check
echo "ðŸ”§ GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Check optimal batch size for your GPUs
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'GPU {i}: {gpu_mem:.1f}GB - Recommended batch size: {int(gpu_mem * 10)}')
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "ðŸš€ Starting OPTIMIZED IR benchmark..."
echo "======================================================================"

# Run the benchmark with optimized settings
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    for OVERLAP in "${OVERLAPS[@]}"; do
        echo "Testing: Chunk Size = $CHUNK_SIZE, Overlap = $OVERLAP"
        
        OUTPUT_DIR_CURRENT="./benchmark_results/chunk_${CHUNK_SIZE}_overlap_${OVERLAP}"
        mkdir -p "$OUTPUT_DIR_CURRENT"
        
        python evaluation/rag_benchmark.py \
            --models_file "$MODELS_FILE" \
            --data_file "$DATA_FILE" \
            --output_dir "$OUTPUT_DIR_CURRENT" \
            --chunk_size "$CHUNK_SIZE" \
            --overlap "$OVERLAP" \
            --top_k_values $TOP_K_VALUES \
            --language "$LANGUAGE" \
            --device "$DEVICE" \
            $PARALLEL_MODELS
    done
done

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo " OPTIMIZED IR BENCHMARK COMPLETED!"
    echo "======================================================================"
    echo "Results location: $OUTPUT_DIR"
    echo ""
    echo "Performance Summary:"
    echo "$(find $OUTPUT_DIR -name "*.log" -exec tail -n 5 {} \;)"
    echo ""
    echo "  Quick Analysis:"
    python utils.py --action top_models --metric precision --top_n 5
    echo ""
    echo "ðŸ” Next steps:"
    echo "1. Full analysis: python evaluation/analyze_results.py --results_dir $OUTPUT_DIR"
    echo "2. View results: head -n 10 $OUTPUT_DIR/benchmark_results_*.csv"
    echo "======================================================================"
else
    echo ""
    echo " BENCHMARK FAILED!"
    echo "Check GPU memory: nvidia-smi"
    echo "Check logs: tail -n 20 $OUTPUT_DIR/logs/*.log"
    exit 1
fi