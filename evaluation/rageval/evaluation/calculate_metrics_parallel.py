import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from metrics import get_metric

def calculate_single_metric(metric_name, input_file, output_file, language="it"):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    metric_class = get_metric(metric_name)
    metric_instance = metric_class()
    
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ground_truth = item["ground_truth"]
            
            score = metric_instance(item, ground_truth, None, language=language)
            
            item[metric_instance.name.upper()] = score
            results.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return metric_name, len(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics", nargs='+', type=str, default=['recall', 'eir', 'ndcg', 'mrr'])
    parser.add_argument("--language", type=str, default="it")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Calculating {len(args.metrics)} metrics in parallel...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for metric in args.metrics:
            output_file = output_dir / f"{metric}.jsonl"
            future = executor.submit(
                calculate_single_metric,
                metric,
                args.input_file,
                output_file,
                args.language
            )
            futures[future] = metric
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Metrics"):
            metric = futures[future]
            try:
                metric_name, count = future.result()
                print(f"  ✅ {metric_name.upper()}: {count} queries processed")
            except Exception as e:
                print(f"  ❌ {metric}: {e}")
    
    print(f"✅ All metrics calculated!")

if __name__ == '__main__':
    main()
