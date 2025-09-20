from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

from .arxiv import get_arxiv_2000, get_arxiv_full
from .c4 import get_c4_data
from .fineweb import get_fineweb_data
from .fineweb_100 import get_fineweb_100_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "fineweb_100":
        return get_fineweb_100_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


class MultiFileDataReader:
    def __init__(
        self,
        data_files,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        """
        DataReader that handles multiple files transparently.
        data_files: list of file paths or single file path
        """
        if isinstance(data_files, (str, Path)):
            # Single file, use regular DataReader
            self.readers = [DataReader(
                data_files, batch_size, sequence_length, seed,
                with_replacement, auto_shard, keep_in_ram
            )]
            self.is_single_file = True
        elif isinstance(data_files, dict):
            # Multiple files from get_fineweb_100_data format
            train_files = [v for k, v in data_files.items() if k.startswith('train_')]
            if train_files:
                self.readers = []
                for i, file_path in enumerate(train_files):
                    # Adjust seed for each file to ensure different sampling
                    file_seed = seed + i * 1000
                    reader = DataReader(
                        file_path, batch_size, sequence_length, file_seed,
                        with_replacement, auto_shard, keep_in_ram
                    )
                    self.readers.append(reader)
                self.is_single_file = False
            else:
                raise ValueError("No train files found in data_files dict")
        elif isinstance(data_files, list):
            # List of file paths
            self.readers = []
            for i, file_path in enumerate(data_files):
                file_seed = seed + i * 1000
                reader = DataReader(
                    file_path, batch_size, sequence_length, file_seed,
                    with_replacement, auto_shard, keep_in_ram
                )
                self.readers.append(reader)
            self.is_single_file = False
        else:
            raise ValueError(f"Unsupported data_files type: {type(data_files)}")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        # Calculate total tokens across all files
        self.num_tokens = sum(reader.num_tokens for reader in self.readers)
        self.current_file_idx = 0
        self.step = 0

    def __len__(self):
        return sum(len(reader) for reader in self.readers)

    def __getitem__(self, idx):
        if self.is_single_file:
            return self.readers[0][idx]

        # Find which file contains this index
        cumulative_len = 0
        for reader in self.readers:
            if idx < cumulative_len + len(reader):
                local_idx = idx - cumulative_len
                return reader[local_idx]
            cumulative_len += len(reader)

        raise IndexError(f"Index {idx} out of range")

    def set_step(self, step):
        self.step = step
        for reader in self.readers:
            reader.set_step(step)

    def sample_batch(self):
        if self.is_single_file:
            return self.readers[0].sample_batch()

        # Round-robin sampling across files or weighted sampling
        if len(self.readers) == 1:
            return self.readers[0].sample_batch()

        # Use weighted random selection based on file sizes
        weights = [reader.num_tokens for reader in self.readers]
        total_weight = sum(weights)

        # Generate deterministic file selection based on step
        rng = np.random.default_rng(self.seed + self.step)
        file_idx = rng.choice(len(self.readers), p=[w/total_weight for w in weights])

        self.step += 1
        return self.readers[file_idx].sample_batch()

    def num_batches(self):
        if self.is_single_file:
            return self.readers[0].num_batches()

        # Return sum of batches across all files
        return sum(reader.num_batches() for reader in self.readers)


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen
