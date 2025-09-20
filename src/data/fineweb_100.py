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
    test_size: float = 0.00001,
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

    train_file_path = os.path.join(fweb_data_path, "train.bin")
    val_file_path = os.path.join(fweb_data_path, "val.bin")

    if os.path.exists(train_file_path) and os.path.exists(val_file_path):
        return {"train": train_file_path, "val": val_file_path}

    parquet_pattern = os.path.join(fineweb_dir, "sample", "100BT", "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {fineweb_dir}/sample/100BT/")

    print(f"Found {len(parquet_files)} parquet files. Building dataset...")

    # Build a single dataset from many local parquet files (arrow-mmap under the hood)
    dataset = Dataset.from_parquet(parquet_files)

    print(f"Dataset size (rows): {len(dataset):,}")

    print(f"Creating train/val split with test_size={test_size}...")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = tknzr.encode_ordinary(example["text"])  # ignores any special tokens
        ids.append(tknzr.eot_token)  # add end-of-text token
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing with datasets.map (multiprocessing)...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Concatenate all ids and write per split
    for split, dset in tokenized.items():
        arr_len = int(np.sum(dset["len"]))
        out_path = val_file_path if split == "val" else train_file_path
        dtype = np.uint16  # gpt2 vocab fits

        print(f"Writing {split} to {out_path} ({arr_len:,} tokens)...")
        arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(arr_len,))

        total_batches = min(1024, len(dset)) if len(dset) > 0 else 0
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f"writing {split}"):
            batch = (
                dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
                .with_format("numpy")
            )
            if len(batch) == 0:
                continue
            arr_batch = np.concatenate(batch["ids"]) if len(batch) > 0 else np.array([], dtype=dtype)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

    return {"train": train_file_path, "val": val_file_path}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process local FineWeb 100BT parquet with datasets.map and write binaries",
    )
    parser.add_argument("--datasets-dir", default="./datasets/", help="Output directory for .bin files")
    parser.add_argument("--fineweb-dir", default="./fineweb/", help="Directory containing local FineWeb data")
    parser.add_argument("--num-proc", type=int, default=40, help="Number of processes for datasets.map")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.00001,
        help="Validation fraction; 0.00001 keeps absolute size similar to 10BT baseline",
    )

    args = parser.parse_args()

    result = get_fineweb_100_data(
        datasets_dir=args.datasets_dir,
        fineweb_dir=args.fineweb_dir,
        num_proc=args.num_proc,
        test_size=args.test_size,
    )

    print("Processing complete!")
    print("Generated files:")
    for key, path in result.items():
        print(f"  {key}: {path}")


