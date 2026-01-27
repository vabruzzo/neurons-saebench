#!/usr/bin/env python
"""
Download SAEBench results from HuggingFace and extract baselines.

Usage:
    python scripts/download_saebench_baselines.py

This will:
1. Download results from adamkarvonen/new_sae_bench_results
2. Parse the JSON files to extract baselines
3. Save to saebench_baselines.json for easy reference
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def download_results(local_dir: str = "./sae_bench_results_cache"):
    """Download SAEBench results from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
        
        # The correct repo name from SAEBench README
        repo_id = "adamkarvonen/sae_bench_results_0125"
        
        print("Downloading SAEBench results from HuggingFace...")
        print(f"  Repo: {repo_id}")
        print("  This may take a few minutes...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            ignore_patterns=[
                "*autointerp_with_generations*",
                "*core_with_feature_statistics*",
            ],
        )
        print(f"  Downloaded to {local_dir}")
        return local_dir
        
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("  Run: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"ERROR downloading: {e}")
        return None


def find_result_files(results_dir: str, eval_type: str) -> list[Path]:
    """Find all result JSON files for an eval type."""
    results_path = Path(results_dir)
    
    # Try different directory structures
    patterns = [
        f"{eval_type}/**/*_eval_results.json",
        f"{eval_type}/*_eval_results.json",
        f"**/{eval_type}/**/*.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(results_path.glob(pattern))
    
    return list(set(files))


def extract_sparse_probing_baselines(results_dir: str) -> dict:
    """Extract sparse probing baselines from results."""
    files = find_result_files(results_dir, "sparse_probing")
    
    if not files:
        print("  No sparse_probing results found")
        return {}
    
    print(f"  Found {len(files)} sparse_probing result files")
    
    # Group by model
    by_model = defaultdict(list)
    
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            model = data.get("eval_config", {}).get("model_name", "unknown")
            metrics = data.get("eval_result_metrics", {})
            
            by_model[model].append({
                "file": str(f),
                "sae_id": data.get("sae_lens_id", "unknown"),
                "llm": metrics.get("llm", {}),
                "sae": metrics.get("sae", {}),
            })
        except Exception as e:
            print(f"    Error parsing {f}: {e}")
    
    # Aggregate by model
    baselines = {}
    for model, results in by_model.items():
        if not results:
            continue
            
        # Use the first result's LLM baseline (should be same across SAEs)
        llm_baseline = results[0].get("llm", {})
        
        # Find best SAE results
        best_sae = {}
        for r in results:
            sae = r.get("sae", {})
            for k, v in sae.items():
                if v is not None:
                    if k not in best_sae or (best_sae[k] is not None and v > best_sae[k]):
                        best_sae[k] = v
        
        baselines[model] = {
            "n_saes_evaluated": len(results),
            "llm_baseline": llm_baseline,
            "sae_best": best_sae,
        }
    
    return baselines


def extract_scr_baselines(results_dir: str) -> dict:
    """Extract SCR baselines from results."""
    files = find_result_files(results_dir, "scr")
    
    if not files:
        # Try scr_and_tpp
        files = find_result_files(results_dir, "scr_and_tpp")
        files = [f for f in files if "scr" in f.name.lower()]
    
    if not files:
        print("  No SCR results found")
        return {}
    
    print(f"  Found {len(files)} SCR result files")
    
    by_model = defaultdict(list)
    
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            model = data.get("eval_config", {}).get("model_name", "unknown")
            metrics = data.get("eval_result_metrics", {}).get("scr_metrics", {})
            
            by_model[model].append({
                "file": str(f),
                "metrics": metrics,
            })
        except Exception as e:
            print(f"    Error parsing {f}: {e}")
    
    baselines = {}
    for model, results in by_model.items():
        # Find best SCR metric across thresholds
        best_metric = None
        best_threshold = None
        
        for r in results:
            metrics = r.get("metrics", {})
            for k, v in metrics.items():
                if "scr_metric" in k and v is not None:
                    if best_metric is None or v > best_metric:
                        best_metric = v
                        best_threshold = k.replace("scr_metric_threshold_", "")
        
        baselines[model] = {
            "n_results": len(results),
            "best_metric": best_metric,
            "best_threshold": best_threshold,
        }
    
    return baselines


def extract_tpp_baselines(results_dir: str) -> dict:
    """Extract TPP baselines from results."""
    files = find_result_files(results_dir, "tpp")
    
    if not files:
        files = find_result_files(results_dir, "scr_and_tpp")
        files = [f for f in files if "tpp" in f.name.lower()]
    
    if not files:
        print("  No TPP results found")
        return {}
    
    print(f"  Found {len(files)} TPP result files")
    
    by_model = defaultdict(list)
    
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            model = data.get("eval_config", {}).get("model_name", "unknown")
            metrics = data.get("eval_result_metrics", {}).get("tpp_metrics", {})
            
            by_model[model].append({
                "file": str(f),
                "metrics": metrics,
            })
        except Exception as e:
            print(f"    Error parsing {f}: {e}")
    
    baselines = {}
    for model, results in by_model.items():
        best_metric = None
        best_threshold = None
        
        for r in results:
            metrics = r.get("metrics", {})
            for k, v in metrics.items():
                if "total_metric" in k and v is not None:
                    if best_metric is None or v > best_metric:
                        best_metric = v
                        best_threshold = k.replace("tpp_threshold_", "").replace("_total_metric", "")
        
        baselines[model] = {
            "n_results": len(results),
            "best_total_metric": best_metric,
            "best_threshold": best_threshold,
        }
    
    return baselines


def format_baselines_for_output(sparse_probing: dict, scr: dict, tpp: dict) -> dict:
    """Format all baselines into a clean output structure."""
    output = {
        "_generated_by": "scripts/download_saebench_baselines.py",
        "_source": "HuggingFace: adamkarvonen/new_sae_bench_results",
        "models": {}
    }
    
    # Get all models
    all_models = set(sparse_probing.keys()) | set(scr.keys()) | set(tpp.keys())
    
    for model in sorted(all_models):
        model_data = {"model_name": model}
        
        # Sparse probing
        if model in sparse_probing:
            sp = sparse_probing[model]
            model_data["sparse_probing"] = {
                "n_saes_evaluated": sp.get("n_saes_evaluated", 0),
                "llm_baseline": {
                    "full": sp.get("llm_baseline", {}).get("llm_test_accuracy"),
                    "top_1": sp.get("llm_baseline", {}).get("llm_top_1_test_accuracy"),
                    "top_2": sp.get("llm_baseline", {}).get("llm_top_2_test_accuracy"),
                    "top_5": sp.get("llm_baseline", {}).get("llm_top_5_test_accuracy"),
                    "top_10": sp.get("llm_baseline", {}).get("llm_top_10_test_accuracy"),
                    "top_20": sp.get("llm_baseline", {}).get("llm_top_20_test_accuracy"),
                    "top_50": sp.get("llm_baseline", {}).get("llm_top_50_test_accuracy"),
                    "top_100": sp.get("llm_baseline", {}).get("llm_top_100_test_accuracy"),
                },
                "sae_best": {
                    "full": sp.get("sae_best", {}).get("sae_test_accuracy"),
                    "top_1": sp.get("sae_best", {}).get("sae_top_1_test_accuracy"),
                    "top_2": sp.get("sae_best", {}).get("sae_top_2_test_accuracy"),
                    "top_5": sp.get("sae_best", {}).get("sae_top_5_test_accuracy"),
                    "top_10": sp.get("sae_best", {}).get("sae_top_10_test_accuracy"),
                    "top_20": sp.get("sae_best", {}).get("sae_top_20_test_accuracy"),
                    "top_50": sp.get("sae_best", {}).get("sae_top_50_test_accuracy"),
                    "top_100": sp.get("sae_best", {}).get("sae_top_100_test_accuracy"),
                },
            }
        
        # SCR
        if model in scr:
            model_data["scr"] = scr[model]
        
        # TPP
        if model in tpp:
            model_data["tpp"] = tpp[model]
        
        output["models"][model] = model_data
    
    return output


def main():
    print("="*60)
    print("SAEBench Baseline Downloader")
    print("="*60)
    
    # Check if already downloaded
    cache_dir = "./sae_bench_results_cache"
    
    if os.path.exists(cache_dir):
        print(f"\nUsing cached results in {cache_dir}")
        print("  (Delete this folder to re-download)")
    else:
        print("\nDownloading results...")
        result = download_results(cache_dir)
        if result is None:
            print("\nFailed to download. Using test data from SAEBench repo instead.")
            cache_dir = "./SAEBench/tests/acceptance/test_data"
            if not os.path.exists(cache_dir):
                print("ERROR: No results available")
                return
    
    # Extract baselines
    print("\nExtracting baselines...")
    
    print("\n1. Sparse Probing:")
    sparse_probing = extract_sparse_probing_baselines(cache_dir)
    
    print("\n2. SCR:")
    scr = extract_scr_baselines(cache_dir)
    
    print("\n3. TPP:")
    tpp = extract_tpp_baselines(cache_dir)
    
    # Format output
    output = format_baselines_for_output(sparse_probing, scr, tpp)
    
    # Save
    output_path = Path(__file__).parent.parent / "saebench_baselines.json"
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Baselines saved to: {output_path}")
    print("="*60)
    
    # Print summary
    print("\nModels found:")
    for model, data in output["models"].items():
        print(f"  • {model}")
        if "sparse_probing" in data:
            sp = data["sparse_probing"]
            llm_10 = sp["llm_baseline"].get("top_10")
            sae_10 = sp["sae_best"].get("top_10")
            if llm_10 and sae_10:
                print(f"    Sparse Probing k=10: LLM={llm_10:.1%}, SAE={sae_10:.1%}")
        if "scr" in data and data["scr"].get("best_metric"):
            print(f"    SCR: {data['scr']['best_metric']:.3f}")
        if "tpp" in data and data["tpp"].get("best_total_metric"):
            print(f"    TPP: {data['tpp']['best_total_metric']:.3f}")


if __name__ == "__main__":
    main()
