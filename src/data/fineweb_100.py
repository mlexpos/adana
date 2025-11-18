import os
import glob

import numpy as np
import tiktoken
from datasets import Dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def get_fineweb_100_data(
    datasets_dir: str,
    fineweb_dir: str = "./fineweb/",
    num_proc: int = 40,
    validation_examples: int = 1470,  # 1470 examples for validation (147M * 1E-5)
):
    """
    Process locally downloaded FineWeb 100BT sample data using Hugging Face
    Datasets map() API for multiprocessing, and write train/val .bin files.

    This expects the local FineWeb files to be under
    {fineweb_dir}/sample/100BT/*.parquet, matching the Hugging Face layout.

    Returns a dict with paths to generated files.
    """
    fweb_data_path = os.path.join(datasets_dir, "fineweb-100BT/")
    os.makedirs(fweb_data_path, exist_ok=True)
    print(f"checking fineweb data is to {fweb_data_path}")

    val_file_path = os.path.join(fweb_data_path, "val.bin")

    # Check if all train files already exist (expecting 150 files)
    #expected_train_files = 150
    sufficient_train_files = 3
    train_pattern = os.path.join(fweb_data_path, "train_*.bin")
    existing_train_files = sorted(glob.glob(train_pattern))
    
    if len(existing_train_files) >= sufficient_train_files and os.path.exists(val_file_path):
        print(f"Found at least {sufficient_train_files} train files and validation file. Skipping tokenization.")
        # Return existing files without processing
        result = {"val": val_file_path}
        for i, train_file in enumerate(existing_train_files):
            result[f"train_{i:04d}"] = train_file
        return result

    parquet_pattern = os.path.join(fineweb_dir, "sample", "100BT", "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {fineweb_dir}/sample/100BT/")

    print(f"Found {len(parquet_files)} parquet files.")

    def process(example):
        ids = tknzr.encode_ordinary(example["text"])  # ignores any special tokens
        ids.append(tknzr.eot_token)  # add end-of-text token
        return {"ids": ids, "len": len(ids)}

    # Process validation set from first parquet file
    if not os.path.exists(val_file_path):
        print(f"Processing validation set from first parquet file with {validation_examples:,} examples...")
        first_dataset = Dataset.from_parquet(parquet_files[0])

        # Take first validation_examples rows for validation
        val_dataset = first_dataset.select(range(min(validation_examples, len(first_dataset))))

        print("Tokenizing validation set...")
        tokenized_val = val_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing validation",
            num_proc=num_proc,
        )

        # Write validation file
        val_arr_len = int(np.sum(tokenized_val["len"]))
        dtype = np.uint16  # gpt2 vocab fits

        print(f"Writing validation to {val_file_path} ({val_arr_len:,} tokens)...")
        val_arr = np.memmap(val_file_path, dtype=dtype, mode="w+", shape=(val_arr_len,))

        # Write in batches for better performance
        batch_size = 1000  # Process 1000 examples at a time
        idx = 0
        for batch_start in tqdm(range(0, len(tokenized_val), batch_size), desc="writing validation"):
            batch_end = min(batch_start + batch_size, len(tokenized_val))
            batch = tokenized_val[batch_start:batch_end]
            # HuggingFace datasets return dict of lists when sliced, so batch["ids"] is a list of token lists
            batch_tokens = np.concatenate([np.array(ids, dtype=dtype) for ids in batch["ids"]])
            val_arr[idx:idx + len(batch_tokens)] = batch_tokens
            idx += len(batch_tokens)

        val_arr.flush()

    # Process each parquet file for train data
    train_files = []
    for i, parquet_file in enumerate(parquet_files):
        train_file_path = os.path.join(fweb_data_path, f"train_{i:04d}.bin")

        if os.path.exists(train_file_path):
            print(f"Train file {train_file_path} already exists, skipping...")
            train_files.append(train_file_path)
            continue

        print(f"Processing train data from {parquet_file}...")

        # For the first parquet file, skip the validation examples
        if i == 0:
            dataset = Dataset.from_parquet(parquet_file)
            train_dataset = dataset.select(range(validation_examples, len(dataset)))
        else:
            train_dataset = Dataset.from_parquet(parquet_file)

        if len(train_dataset) == 0:
            print(f"No train data left in {parquet_file}, skipping...")
            continue

        print(f"Dataset size (rows): {len(train_dataset):,}")

        print("Tokenizing train data...")
        tokenized_train = train_dataset.map(
            process,
            remove_columns=["text"],
            desc=f"tokenizing train {i:04d}",
            num_proc=num_proc,
        )

        # Write train file
        train_arr_len = int(np.sum(tokenized_train["len"]))

        print(f"Writing train to {train_file_path} ({train_arr_len:,} tokens)...")
        train_arr = np.memmap(train_file_path, dtype=dtype, mode="w+", shape=(train_arr_len,))

        # Write in batches for better performance
        batch_size = 1000  # Process 1000 examples at a time
        idx = 0
        for batch_start in tqdm(range(0, len(tokenized_train), batch_size), desc=f"writing train {i:04d}"):
            batch_end = min(batch_start + batch_size, len(tokenized_train))
            batch = tokenized_train[batch_start:batch_end]
            # HuggingFace datasets return dict of lists when sliced, so batch["ids"] is a list of token lists
            batch_tokens = np.concatenate([np.array(ids, dtype=dtype) for ids in batch["ids"]])
            train_arr[idx:idx + len(batch_tokens)] = batch_tokens
            idx += len(batch_tokens)

        train_arr.flush()
        train_files.append(train_file_path)

    # Return in format expected by main.py - individual train_XXXX keys
    result = {"val": val_file_path}
    for i, train_file in enumerate(train_files):
        result[f"train_{i:04d}"] = train_file
    
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process local FineWeb 100BT parquet with datasets.map and write binaries",
    )
    parser.add_argument("--datasets-dir", default="./datasets/", help="Output directory for .bin files")
    parser.add_argument("--fineweb-dir", default="./fineweb/", help="Directory containing local FineWeb data")
    parser.add_argument("--num-proc", type=int, default=40, help="Number of processes for datasets.map")
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=1470,
        help="Number of examples to use for validation from first parquet file",
    )

    args = parser.parse_args()

    result = get_fineweb_100_data(
        datasets_dir=args.datasets_dir,
        fineweb_dir=args.fineweb_dir,
        num_proc=args.num_proc,
        validation_examples=args.validation_examples,
    )

    print("Processing complete!")
    print("Generated files:")
    for key, path in result.items():
        print(f"  {key}: {path}")


