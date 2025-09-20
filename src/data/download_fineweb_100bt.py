#!/usr/bin/env python3
"""
Script to download the FineWeb 100BT sample dataset.
This predownloads the data so it can be processed locally.
"""

import os
from huggingface_hub import snapshot_download


def download_fineweb_100bt(local_dir="./fineweb/"):
    """
    Download the FineWeb 100BT sample dataset.
    
    Args:
        local_dir (str): Local directory to download the dataset to
        
    Returns:
        str: Path to the downloaded dataset directory
    """
    print("Starting download of FineWeb 100BT sample...")
    print(f"Downloading to: {local_dir}")
    
    try:
        folder = snapshot_download(
            "HuggingFaceFW/fineweb", 
            repo_type="dataset",
            local_dir=local_dir,
            # Use the 100BT sample pattern
            allow_patterns="sample/100BT/*"
        )
        
        print(f"Successfully downloaded dataset to: {folder}")
        return folder
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FineWeb 100BT sample dataset")
    parser.add_argument(
        "--local-dir", 
        default="./fineweb/", 
        help="Local directory to download the dataset to (default: ./fineweb/)"
    )
    
    args = parser.parse_args()
    
    download_fineweb_100bt(args.local_dir)
