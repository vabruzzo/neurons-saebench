"""
Download SAEBench datasets for sparse probing evaluation.

These are the actual datasets used in the SAEBench paper.
"""

import os
from datasets import load_dataset

# SAEBench sparse probing datasets
DATASETS = {
    "bias_in_bios": {
        "hf_name": "LabHC/bias_in_bios",
        "text_field": "hard_text",
        "label_field": "profession",
        "description": "Predict occupation from bio text",
    },
    "imdb": {
        "hf_name": "stanfordnlp/imdb",
        "text_field": "text",
        "label_field": "label",  # 0=neg, 1=pos
        "description": "Sentiment from movie reviews",
    },
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "text_field": "text",
        "label_field": "label",  # 0-3: World, Sports, Business, Sci/Tech
        "description": "News topic classification",
    },
}


def download_and_prepare_datasets(output_dir: str = "data/saebench", max_train: int = 4000, max_test: int = 1000):
    """
    Download SAEBench datasets and prepare them in a consistent format.
    
    Args:
        output_dir: Where to save datasets
        max_train: Maximum training examples per dataset
        max_test: Maximum test examples per dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            if "subset" in config:
                ds = load_dataset(config["hf_name"], config["subset"])
            else:
                ds = load_dataset(config["hf_name"])
            
            print(f"Raw splits: {list(ds.keys())}")
            
            # Get train and test splits
            if "train" in ds and "test" in ds:
                train_ds = ds["train"]
                test_ds = ds["test"]
            elif "train" in ds and "validation" in ds:
                train_ds = ds["train"]
                test_ds = ds["validation"]
            else:
                # Split manually
                split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
                train_ds = split_ds["train"]
                test_ds = split_ds["test"]
            
            # Subsample if needed
            if len(train_ds) > max_train:
                train_ds = train_ds.shuffle(seed=42).select(range(max_train))
            if len(test_ds) > max_test:
                test_ds = test_ds.shuffle(seed=42).select(range(max_test))
            
            # Rename columns for consistency
            text_field = config["text_field"]
            label_field = config["label_field"]
            
            def standardize(example):
                return {
                    "text": example[text_field],
                    "label": example[label_field],
                }
            
            train_ds = train_ds.map(standardize, remove_columns=train_ds.column_names)
            test_ds = test_ds.map(standardize, remove_columns=test_ds.column_names)
            
            # Get label info
            unique_labels = set(train_ds["label"])
            print(f"Train size: {len(train_ds)}")
            print(f"Test size: {len(test_ds)}")
            print(f"Unique labels: {len(unique_labels)}")
            
            # Save
            save_path = os.path.join(output_dir, name)
            os.makedirs(save_path, exist_ok=True)
            train_ds.save_to_disk(os.path.join(save_path, "train"))
            test_ds.save_to_disk(os.path.join(save_path, "test"))
            
            print(f"Saved to: {save_path}")
            
        except Exception as e:
            print(f"ERROR downloading {name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")


def load_saebench_dataset(name: str, data_dir: str = "data/saebench"):
    """Load a prepared SAEBench dataset."""
    from datasets import load_from_disk
    
    path = os.path.join(data_dir, name)
    train = load_from_disk(os.path.join(path, "train"))
    test = load_from_disk(os.path.join(path, "test"))
    
    return {"train": train, "test": test}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/saebench")
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-test", type=int, default=1000)
    
    args = parser.parse_args()
    
    download_and_prepare_datasets(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_test=args.max_test,
    )
