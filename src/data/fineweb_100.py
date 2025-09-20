import os

import numpy as np
import tiktoken
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def get_fineweb_100_data(datasets_dir, num_proc=40, batch_size=1000):
    """To change the cache dir, run `export HF_HOME=/path/to/cache/` before running the code."""
    FWEB_DATA_PATH = os.path.join(datasets_dir, "fineweb-100BT/")

    # Check if all files already exist
    train_files = [os.path.join(FWEB_DATA_PATH, f"train_{i:02d}.bin") for i in range(10)]
    val_file = os.path.join(FWEB_DATA_PATH, "val.bin")

    if not all(os.path.exists(f) for f in train_files + [val_file]):
        os.makedirs(FWEB_DATA_PATH, exist_ok=True)

        print("Loading dataset...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-100BT",
            split="train",
            streaming=True,  # Use streaming to avoid loading entire dataset
            verification_mode="no_checks",
            download_config=DownloadConfig(max_retries=10),
        )

        # Calculate test_size to maintain same absolute size as 10BT version
        # Original 10BT had test_size=0.0001, so absolute size was 0.0001 * 10BT = 0.001BT
        # For 100BT: 0.001BT / 100BT = 0.00001
        test_size = 0.00001

        print(f"Splitting dataset with test_size={test_size}...")
        split_dataset = dataset.train_test_split(
            test_size=test_size, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        print("Processing validation set...")
        # Process validation set (smaller, can be handled normally)
        val_dataset = split_dataset["val"]
        val_tokenized = val_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing validation split",
            num_proc=num_proc,
            batched=False
        )

        # Write validation set
        print("Writing validation set...")
        val_lengths = [item["len"] for item in val_tokenized]
        val_arr_len = sum(val_lengths)
        val_filename = os.path.join(FWEB_DATA_PATH, "val.bin")
        dtype = np.uint16
        val_arr = np.memmap(val_filename, dtype=dtype, mode="w+", shape=(val_arr_len,))

        val_idx = 0
        for item in tqdm(val_tokenized, desc="writing validation data"):
            ids = item["ids"]
            val_arr[val_idx:val_idx + len(ids)] = ids
            val_idx += len(ids)
        val_arr.flush()

        print("Processing training set in batches...")
        # Process training set in streaming batches to avoid memory issues
        train_dataset = split_dataset["train"]

        # Estimate total size and prepare file writers
        train_files_writers = []
        train_files_indices = []

        for i in range(10):
            filename = os.path.join(FWEB_DATA_PATH, f"train_{i:02d}.bin")
            # Pre-allocate with estimated size (will resize if needed)
            estimated_size = 10_000_000_000  # 10B tokens per file estimate
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(estimated_size,))
            train_files_writers.append(arr)
            train_files_indices.append(0)

        current_file = 0
        total_processed = 0
        tokens_per_file = []

        # Process in batches
        batch_iter = train_dataset.iter(batch_size=batch_size)

        for batch in tqdm(batch_iter, desc="processing training batches"):
            # Tokenize batch
            tokenized_batch = []
            for text in batch["text"]:
                ids = tknzr.encode_ordinary(text)
                ids.append(tknzr.eot_token)
                tokenized_batch.extend(ids)

            # Determine which file(s) to write to
            batch_array = np.array(tokenized_batch, dtype=dtype)
            batch_size_tokens = len(batch_array)

            # Write to current file, switching files when we reach capacity
            remaining_tokens = batch_size_tokens
            batch_offset = 0

            while remaining_tokens > 0:
                current_idx = train_files_indices[current_file]
                current_writer = train_files_writers[current_file]
                available_space = len(current_writer) - current_idx

                # If current file is getting full, switch to next file
                if available_space < remaining_tokens and current_file < 9:
                    # Write what fits in current file
                    if available_space > 0:
                        current_writer[current_idx:current_idx + available_space] = batch_array[batch_offset:batch_offset + available_space]
                        train_files_indices[current_file] += available_space
                        batch_offset += available_space
                        remaining_tokens -= available_space

                    # Resize current file to actual size used
                    actual_size = train_files_indices[current_file]
                    tokens_per_file.append(actual_size)
                    current_writer.flush()

                    # Resize the memory map to actual size
                    del train_files_writers[current_file]
                    filename = os.path.join(FWEB_DATA_PATH, f"train_{current_file:02d}.bin")
                    resized_arr = np.memmap(filename, dtype=dtype, mode="r+", shape=(actual_size,))
                    train_files_writers[current_file] = resized_arr

                    # Move to next file
                    current_file += 1
                    if current_file >= 10:
                        print("Warning: Exceeded 10 files, truncating data")
                        break
                else:
                    # Write remaining tokens to current file
                    tokens_to_write = min(remaining_tokens, available_space)

                    # Expand file if needed
                    if current_idx + tokens_to_write > len(current_writer):
                        current_writer.flush()
                        del train_files_writers[current_file]
                        filename = os.path.join(FWEB_DATA_PATH, f"train_{current_file:02d}.bin")
                        new_size = current_idx + tokens_to_write + 1_000_000  # Add buffer
                        expanded_arr = np.memmap(filename, dtype=dtype, mode="r+", shape=(new_size,))
                        train_files_writers[current_file] = expanded_arr
                        current_writer = expanded_arr

                    current_writer[current_idx:current_idx + tokens_to_write] = batch_array[batch_offset:batch_offset + tokens_to_write]
                    train_files_indices[current_file] += tokens_to_write
                    remaining_tokens = 0

            total_processed += batch_size_tokens

        # Finalize all files - resize to actual content
        for i in range(10):
            if i < len(train_files_writers):
                actual_size = train_files_indices[i]
                if actual_size > 0:
                    train_files_writers[i].flush()
                    del train_files_writers[i]

                    # Final resize to exact size
                    filename = os.path.join(FWEB_DATA_PATH, f"train_{i:02d}.bin")
                    final_arr = np.memmap(filename, dtype=dtype, mode="r+", shape=(actual_size,))
                    final_arr.flush()
                    tokens_per_file.append(actual_size)
                    print(f"File train_{i:02d}.bin: {actual_size:,} tokens")

        print(f"Total training tokens processed: {total_processed:,}")
        print(f"Split into {len([f for f in tokens_per_file if f > 0])} files")

    # Return paths to all files
    result = {"val": os.path.join(FWEB_DATA_PATH, "val.bin")}
    for i in range(10):
        train_file = os.path.join(FWEB_DATA_PATH, f"train_{i:02d}.bin")
        if os.path.exists(train_file):
            result[f"train_{i:02d}"] = train_file

    return result


if __name__ == "__main__":
    get_fineweb_100_data("./datasets/")