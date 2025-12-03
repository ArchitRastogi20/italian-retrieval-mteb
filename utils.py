#!/usr/bin/env python3
"""
Utility functions for embedding benchmarking
Provides helper functions for data preparation, model management, and result processing
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer
import torch

def setup_logger(name: str = "utils") -> logging.Logger:
    """Setup logger for utilities"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

class DataConverter:
    """Convert various data formats to benchmark format"""
    
    @staticmethod
    def csv_to_jsonl(csv_file: str, output_file: str, 
                     query_col: str = "query", 
                     doc_col: str = "document",
                     ground_truth_col: str = "ground_truth") -> bool:
        """Convert CSV file to JSONL benchmark format"""
        logger = setup_logger()
        
        try:
            df = pd.read_csv(csv_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for idx, row in df.iterrows():
                    item = {
                        "query": {
                            "query_id": f"item_{idx}",
                            "content": str(row[query_col])
                        },
                        "documents": [
                            {"content": str(row[doc_col])}
                        ],
                        "ground_truth": {
                            "content": str(row[ground_truth_col]),
                            "references": [str(row[doc_col])],
                            "keypoints": []
                        },
                        "language": "en"
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Converted {len(df)} items from CSV to JSONL")
            return True
            
        except Exception as e:
            logger.error(f"Error converting CSV to JSONL: {e}")
            return False
    
    @staticmethod
    def json_to_jsonl(json_file: str, output_file: str) -> bool:
        """Convert JSON array to JSONL format"""
        logger = setup_logger()
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Converted {len(data)} items from JSON to JSONL")
            return True
            
        except Exception as e:
            logger.error(f"Error converting JSON to JSONL: {e}")
            return False

class ModelManager:
    """Manage embedding models and their metadata"""
    
    def __init__(self, models_file: str = "models.txt"):
        self.models_file = models_file
        self.logger = setup_logger()
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate that all models in models.txt can be loaded"""
        models = self.load_model_list()
        validation_results = {}
        
        for model_name in models:
            try:
                self.logger.info(f"Validating model: {model_name}")
                model = SentenceTransformer(model_name)
                validation_results[model_name] = True
                del model  # Free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {e}")
                validation_results[model_name] = False
        
        return validation_results
    
    def load_model_list(self) -> List[str]:
        """Load model names from models.txt"""
        models = []
        
        if not Path(self.models_file).exists():
            self.logger.error(f"Models file not found: {self.models_file}")
            return models
        
        with open(self.models_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append(line)
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            model = SentenceTransformer(model_name)
            
            info = {
                "model_name": model_name,
                "max_seq_length": model.max_seq_length,
                "embedding_dimension": model.get_sentence_embedding_dimension(),
                "tokenizer": str(type(model.tokenizer).__name__),
                "device": str(model.device)
            }
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name}: {e}")
            return {"error": str(e)}
    
    def create_model_info_file(self, output_file: str = "model_info.json"):
        """Create a file with information about all models"""
        models = self.load_model_list()
        model_info = {}
        
        for model_name in models:
            self.logger.info(f"Getting info for: {model_name}")
            model_info[model_name] = self.get_model_info(model_name)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Model information saved to: {output_file}")

class ResultsProcessor:
    """Process and analyze benchmark results"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.logger = setup_logger()
    
    def merge_results(self, output_file: str = "merged_results.csv") -> bool:
        """Merge multiple CSV result files into one"""
        try:
            csv_files = list(self.results_dir.glob("benchmark_results_*.csv"))
            
            if not csv_files:
                self.logger.error("No benchmark result CSV files found")
                return False
            
            all_dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['result_file'] = csv_file.name
                all_dfs.append(df)
            
            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_df.to_csv(self.results_dir / output_file, index=False)
            
            self.logger.info(f"Merged {len(csv_files)} files into {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging results: {e}")
            return False
    
    def filter_results(self, input_file: str, output_file: str, 
                      filters: Dict[str, Any]) -> bool:
        """Filter results based on criteria"""
        try:
            df = pd.read_csv(self.results_dir / input_file)
            
            filtered_df = df.copy()
            for column, value in filters.items():
                if column in filtered_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
            
            filtered_df.to_csv(self.results_dir / output_file, index=False)
            
            self.logger.info(f"Filtered {len(df)} -> {len(filtered_df)} results")
            return True
            
        except Exception as e:
            self.logger.error(f"Error filtering results: {e}")
            return False
    
    def get_top_models(self, metric: str = "precision", top_n: int = 10) -> List[Dict]:
        """Get top N models by a specific metric"""
        try:
            csv_files = list(self.results_dir.glob("benchmark_results_*.csv"))
            
            if not csv_files:
                return []
            
            # Use most recent file
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            if metric not in df.columns:
                self.logger.error(f"Metric '{metric}' not found in results")
                return []
            
            # Get average performance per model
            top_models = (df.groupby('model')[metric]
                         .mean()
                         .sort_values(ascending=False)
                         .head(top_n))
            
            result = []
            for model, score in top_models.items():
                result.append({
                    "model": model,
                    f"avg_{metric}": score,
                    "rank": len(result) + 1
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting top models: {e}")
            return []

class ConfigManager:
    """Manage benchmark configurations"""
    
    def __init__(self, config_file: str = "benchmark_config.json"):
        self.config_file = config_file
        self.logger = setup_logger()
    
    def create_config_template(self):
        """Create a configuration template"""
        template = {
            "chunk_sizes": [256, 512, 1024],
            "overlaps": [0, 25, 50, 100],
            "top_k_values": [1, 5, 10],
            "languages": ["en", "zh"],
            "devices": ["cuda", "cpu"],
            "batch_sizes": {
                "embedding": 32,
                "evaluation": 16
            },
            "models": {
                "small": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "intfloat/e5-small-v2"
                ],
                "medium": [
                    "sentence-transformers/all-mpnet-base-v2",
                    "intfloat/e5-base-v2"
                ],
                "large": [
                    "intfloat/e5-large-v2",
                    "BAAI/bge-large-en-v1.5"
                ]
            },
            "evaluation": {
                "use_openai": False,
                "openai_model": "gpt-4o-mini",
                "openai_version": "v2"
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration template created: {self.config_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not Path(self.config_file).exists():
            self.logger.warning(f"Config file not found: {self.config_file}")
            return {}
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_benchmark_commands(self, data_file: str) -> List[str]:
        """Generate benchmark commands for different configurations"""
        config = self.load_config()
        commands = []
        
        if not config:
            return commands
        
        # Generate commands for different configurations
        for chunk_size in config.get("chunk_sizes", [512]):
            for overlap in config.get("overlaps", [0]):
                for model_size, models in config.get("models", {}).items():
                    # Create temporary model file
                    temp_models_file = f"models_{model_size}.txt"
                    with open(temp_models_file, 'w') as f:
                        for model in models:
                            f.write(f"{model}\n")
                    
                    command = f"""python embedding_benchmark.py \\
    --models_file {temp_models_file} \\
    --data_file {data_file} \\
    --output_dir benchmark_results/{model_size}_chunk{chunk_size}_overlap{overlap} \\
    --chunk_size {chunk_size} \\
    --overlap {overlap} \\
    --top_k_values {' '.join(map(str, config.get('top_k_values', [5])))} \\
    --language {config.get('languages', ['en'])[0]} \\
    --device {config.get('devices', ['cuda'])[0]}"""
                    
                    if config.get("evaluation", {}).get("use_openai", False):
                        command += " --use_openai"
                    
                    commands.append(command)
        
        return commands

def main():
    """Main utility function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Benchmark Utilities")
    parser.add_argument("--action", type=str, required=True,
                       choices=["validate_models", "convert_csv", "convert_json", 
                               "model_info", "merge_results", "top_models", "config_template"],
                       help="Action to perform")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--models_file", type=str, default="models.txt", help="Models file")
    parser.add_argument("--metric", type=str, default="precision", help="Metric for analysis")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top results")
    
    args = parser.parse_args()
    
    if args.action == "validate_models":
        manager = ModelManager(args.models_file)
        results = manager.validate_models()
        for model, valid in results.items():
            status = " " if valid else "âœ—"
            print(f"{status} {model}")
    
    elif args.action == "convert_csv":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for CSV conversion")
            return
        DataConverter.csv_to_jsonl(args.input_file, args.output_file)
    
    elif args.action == "convert_json":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for JSON conversion")
            return
        DataConverter.json_to_jsonl(args.input_file, args.output_file)
    
    elif args.action == "model_info":
        manager = ModelManager(args.models_file)
        output_file = args.output_file or "model_info.json"
        manager.create_model_info_file(output_file)
    
    elif args.action == "merge_results":
        processor = ResultsProcessor()
        output_file = args.output_file or "merged_results.csv"
        processor.merge_results(output_file)
    
    elif args.action == "top_models":
        processor = ResultsProcessor()
        top_models = processor.get_top_models(args.metric, args.top_n)
        print(f"\nTop {args.top_n} models by {args.metric}:")
        print("-" * 50)
        for model_info in top_models:
            print(f"{model_info['rank']:2d}. {model_info['model']} ({model_info[f'avg_{args.metric}']:.4f})")
    
    elif args.action == "config_template":
        manager = ConfigManager(args.output_file or "benchmark_config.json")
        manager.create_config_template()

if __name__ == "__main__":
    main()