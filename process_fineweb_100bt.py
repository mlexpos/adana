#!/usr/bin/env python3
"""
Main script to download and process FineWeb 100BT sample dataset.
This script handles both downloading the dataset and tokenizing it.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from download_fineweb_100bt import download_fineweb_100bt
from data.fineweb_100_local import get_fineweb_100_data_local


def main():
    parser = argparse.ArgumentParser(
        description="Download and process FineWeb 100BT sample dataset"
    )
    parser.add_argument(
        "--download-dir", 
        default="./fineweb-edu/", 
        help="Directory to download the dataset to (default: ./fineweb/)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./datasets/", 
        help="Directory to save processed tokenized data (default: ./datasets/)"
    )
    parser.add_argument(
        "--skip-download", 
        action="store_true", 
        help="Skip download step if data already exists"
    )
    parser.add_argument(
        "--num-proc", 
        type=int, 
        default=40, 
        help="Number of processes for parallel processing (default: 40)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1000, 
        help="Batch size for processing (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Download the dataset (if not skipping)
    if not args.skip_download:
        print("=" * 60)
        print("STEP 1: Downloading FineWeb 100BT sample dataset")
        print("=" * 60)
        
        try:
            download_path = download_fineweb_100bt(args.download_dir)
            print(f"✓ Download completed successfully")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            sys.exit(1)
    else:
        print("Skipping download step...")
        download_path = args.download_dir
    
    # Step 2: Process and tokenize the dataset
    print("\n" + "=" * 60)
    print("STEP 2: Processing and tokenizing dataset")
    print("=" * 60)
    
    try:
        result = get_fineweb_100_data_local(
            datasets_dir=args.output_dir,
            fineweb_dir=args.download_dir,
            num_proc=args.num_proc,
            batch_size=args.batch_size
        )
        
        print("\n✓ Processing completed successfully!")
        print("\nGenerated files:")
        for key, path in result.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024**3)  # Size in GB
                print(f"  {key}: {path} ({file_size:.2f} GB)")
            else:
                print(f"  {key}: {path} (not found)")
                
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Dataset downloaded to: {args.download_dir}")
    print(f"Tokenized data saved to: {args.output_dir}/fineweb-100BT/")
    print("\nYou can now use the tokenized data for training your model.")


if __name__ == "__main__":
    main()
