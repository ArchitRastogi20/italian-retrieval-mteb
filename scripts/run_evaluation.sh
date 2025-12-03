#!/bin/bash
QUERIES_FILE="./data/evaluation_data_answer_focused.jsonl"
CHUNK_SIZE=512
STRIDE=256
TOPK_VALUES="1 5 10 20 50 100"
DEVICE="cuda"
PREDICTIONS_DIR="./results/predictions"
METRICS_DIR="./results/metrics"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PREDICTIONS_DIR" "$METRICS_DIR"

declare -A MODELS_BATCH_SIZES=(
    ["Qwen/Qwen3-Embedding-4B"]=16
    ["intfloat/multilingual-e5-large-instruct"]=32
    ["intfloat/multilingual-e5-large"]=32
    ["BAAI/bge-m3"]=32
    ["Snowflake/snowflake-arctic-embed-l-v2.0"]=32
    ["Lajavaness/bilingual-embedding-large"]=32
    ["manu/bge-fr-en"]=32
    ["google/embeddinggemma-300m"]=64
    ["deepvk/USER-bge-m3"]=32
    ["jinaai/jina-embeddings-v3"]=32
    # ["jinaai/jina-embeddings-v4"]=32
    # ["Alibaba-NLP/gte-multilingual-base"]=64
    ["Lajavaness/bilingual-embedding-base"]=64
    ["intfloat/multilingual-e5-base"]=64
    ["Lajavaness/bilingual-embedding-small"]=128
    # ["infly/inf-retriever-v1-1.5b"]=32
    ["Qwen/Qwen3-Embedding-0.6B"]=64
    ["intfloat/multilingual-e5-small"]=128
    # ["Snowflake/snowflake-arctic-embed-m-v2.0"]=64
    # ["NovaSearch/jasper_en_vision_language_v1"]=32
    ["sergeyzh/LaBSE-ru-turbo"]=64
    ["HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"]=64
    # ["NovaSearch/stella_en_1.5B_v5"]=32
    ["Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka"]=64
    ["sentence-transformers/LaBSE"]=64
    ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]=128
    ["Gameselo/STS-multilingual-mpnet-base-v2"]=128
    ["Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet"]=128
)

SUMMARY_FILE="$SCRIPT_DIR/results/all_models_summary.csv"
echo "model_name,k,recall,eir,ndcg,mrr" > "$SUMMARY_FILE"

echo "ðŸš€ Full Evaluation: 28 Models Ã— 6 TopK"
echo " Dataset: 1000 queries"
echo "â±ï¸  Estimated: 2-3 hours"

TOTAL_MODELS=${#MODELS_BATCH_SIZES[@]}
CURRENT=0
START_TIME=$(date +%s)

for MODEL in "${!MODELS_BATCH_SIZES[@]}"; do
    CURRENT=$((CURRENT + 1))
    BATCH_SIZE=${MODELS_BATCH_SIZES[$MODEL]}
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
    MODEL_OUTPUT_DIR="$SCRIPT_DIR/$PREDICTIONS_DIR/$MODEL_SAFE"
    
    echo ""
    echo "[$CURRENT/$TOTAL_MODELS] ðŸ”„ $MODEL"
    
    ALL_EXIST=true
    for K in $TOPK_VALUES; do
        if [ ! -f "$MODEL_OUTPUT_DIR/k${K}_predictions.jsonl" ]; then
            ALL_EXIST=false
            break
        fi
    done
    
    if [ "$ALL_EXIST" = true ]; then
        echo "â­ï¸  Predictions exist"
    else
        echo "ðŸ” Retrieval..."
        python "$SCRIPT_DIR/rageval/evaluation/evaluate_dense_biencoder_faiss.py" \
            --query_file "$SCRIPT_DIR/$QUERIES_FILE" \
            --output_dir "$MODEL_OUTPUT_DIR" \
            --model_name "$MODEL" \
            --topk_values $TOPK_VALUES \
            --chunk_size $CHUNK_SIZE \
            --chunk_stride $STRIDE \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --use_gpu_faiss
        
        if [ $? -ne 0 ]; then
            echo " Failed"
            continue
        fi
    fi
    
    for K in $TOPK_VALUES; do
        PRED_FILE="$MODEL_OUTPUT_DIR/k${K}_predictions.jsonl"
        METRICS_OUTPUT_DIR="$SCRIPT_DIR/$METRICS_DIR/${MODEL_SAFE}_k${K}"
        
        [ ! -f "$PRED_FILE" ] && continue
        
        echo " Metrics k=$K..."
        
        python "$SCRIPT_DIR/rageval/evaluation/calculate_metrics_parallel.py" \
            --input_file "$PRED_FILE" \
            --output_dir "$METRICS_OUTPUT_DIR" \
            --metrics recall eir ndcg mrr \
            --language "it" \
            --num_workers 4 2>&1 | grep "âœ…"
        
        python << EOF
import json, sys
metrics_dir = "$METRICS_OUTPUT_DIR"
results = {'k': $K}

for metric in ['recall', 'eir', 'ndcg', 'mrr']:
    filepath = f"{metrics_dir}/{metric}.jsonl"
    try:
        with open(filepath) as f:
            scores = []
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    score = data.get(metric.upper(), data.get(metric.lower(), 0))
                    scores.append(score)
            results[metric] = sum(scores) / len(scores) if scores else 0
    except:
        results[metric] = 0

with open("$SUMMARY_FILE", "a") as f:
    f.write(f"$MODEL,{results['k']},{results.get('recall',0):.4f},{results.get('eir',0):.4f},{results.get('ndcg',0):.4f},{results.get('mrr',0):.4f}\n")

print(f"  k={results['k']:3d}: R={results.get('recall',0):.4f} E={results.get('eir',0):.4f} N={results.get('ndcg',0):.4f} M={results.get('mrr',0):.4f}")
EOF
    done
    
    # Cleanup predictions
    rm -rf "$MODEL_OUTPUT_DIR"
    
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    AVG=$((ELAPSED / CURRENT))
    ETA=$(( (TOTAL_MODELS - CURRENT) * AVG / 60))
    echo "â±ï¸  ETA: ${ETA}m"
    
    python -c "import torch, gc; torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()" 2>/dev/null
done

echo ""
echo "ðŸŽ‰ Done!"
column -t -s',' "$SUMMARY_FILE" | head -n 31
