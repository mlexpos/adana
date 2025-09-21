from pathlib import Path
from typing import Dict
import threading
import queue
import time

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


class AsyncFileLoader:
    """Asynchronous file loader that preloads the next file in background"""
    def __init__(self):
        self.load_queue = queue.Queue(maxsize=1)  # Only keep one file in queue
        self.loader_thread = None
        self.shutdown_flag = threading.Event()
        
    def start_loading(self, file_path):
        """Start loading a file asynchronously"""
        if self.loader_thread is not None and self.loader_thread.is_alive():
            return  # Already loading
            
        self.shutdown_flag.clear()
        self.loader_thread = threading.Thread(
            target=self._load_file_worker, 
            args=(file_path,),
            daemon=True
        )
        self.loader_thread.start()
        
    def _load_file_worker(self, file_path):
        """Worker thread to load file into memory"""
        try:
            if self.shutdown_flag.is_set():
                return
            print(f"Async loading file: {file_path}")
            start_time = time.time()
            data = np.array(np.memmap(file_path, dtype=np.uint16, mode="r"))
            load_time = time.time() - start_time
            print(f"Async loaded {file_path} in {load_time:.2f}s ({len(data)/1e6:.1f}M tokens)")
            
            if not self.shutdown_flag.is_set():
                # Try to put in queue, but don't block if queue is full
                try:
                    self.load_queue.put((file_path, data), timeout=0.1)
                except queue.Full:
                    pass  # Queue full, discard this load
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
    def get_loaded_file(self, timeout=0.1):
        """Get a loaded file if available"""
        try:
            return self.load_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def shutdown(self):
        """Shutdown the async loader"""
        self.shutdown_flag.set()
        if self.loader_thread is not None and self.loader_thread.is_alive():
            self.loader_thread.join(timeout=1.0)


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
        Optimized DataReader that handles multiple files with dual buffering.
        Always keeps current file and next file in RAM, with async loading.
        """
        # Parse file paths
        if isinstance(data_files, (str, Path)):
            self.file_paths = [Path(data_files)]
            self.is_single_file = True
        elif isinstance(data_files, dict):
            train_files = [Path(v) for k, v in data_files.items() if k.startswith('train_')]
            if not train_files:
                raise ValueError("No train files found in data_files dict")
            self.file_paths = sorted(train_files)  # Sort for consistent ordering
            self.is_single_file = False
        elif isinstance(data_files, list):
            self.file_paths = [Path(f) for f in data_files]
            self.is_single_file = False
        else:
            raise ValueError(f"Unsupported data_files type: {type(data_files)}")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.auto_shard = auto_shard

        # Initialize distributed settings
        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Dual buffer system: current file and next file in RAM
        self.current_file_idx = 0
        self.current_data = None
        self.current_reader = None
        self.next_data = None
        
        # Async file loader for preloading next file
        self.async_loader = AsyncFileLoader()
        
        # Load initial files
        self._load_current_file()
        if len(self.file_paths) > 1:
            self._start_loading_next_file()

        # Calculate total tokens (approximate, will be exact once all files are seen)
        self._calculate_total_tokens()
        
        self.step = 0
        print(f"MultiFileDataReader initialized with {len(self.file_paths)} files, "
              f"~{self.num_tokens/1e6:.1f}M total tokens")

    def _load_current_file(self):
        """Load current file into RAM and create DataReader"""
        file_path = self.file_paths[self.current_file_idx]
        print(f"Loading current file: {file_path}")
        start_time = time.time()
        
        # Load file data into RAM
        self.current_data = np.array(np.memmap(file_path, dtype=np.uint16, mode="r"))
        load_time = time.time() - start_time
        print(f"Loaded {file_path} in {load_time:.2f}s ({len(self.current_data)/1e6:.1f}M tokens)")
        
        # Create DataReader with data already in RAM
        file_seed = self.seed + self.current_file_idx * 1000
        self.current_reader = DataReader(
            data_src=self.current_data,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            seed=file_seed,
            with_replacement=self.with_replacement,
            auto_shard=self.auto_shard,
            keep_in_ram=True  # Data already in RAM
        )

    def _start_loading_next_file(self):
        """Start asynchronously loading the next file"""
        if len(self.file_paths) <= 1:
            return
            
        next_idx = (self.current_file_idx + 1) % len(self.file_paths)
        next_file_path = self.file_paths[next_idx]
        self.async_loader.start_loading(next_file_path)

    def _switch_to_next_file(self):
        """Switch current file to next file and start loading the new next file"""
        # Check if next file is ready
        loaded_file = self.async_loader.get_loaded_file(timeout=5.0)  # Wait up to 5 seconds
        
        if loaded_file is not None:
            file_path, data = loaded_file
            print(f"Switching to pre-loaded file: {file_path}")
            self.next_data = data
        else:
            # Fallback: load synchronously
            next_idx = (self.current_file_idx + 1) % len(self.file_paths)
            file_path = self.file_paths[next_idx]
            print(f"Loading next file synchronously (async not ready): {file_path}")
            self.next_data = np.array(np.memmap(file_path, dtype=np.uint16, mode="r"))
        
        # Switch files
        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_paths)
        self.current_data = self.next_data
        self.next_data = None
        
        # Create new DataReader for current file
        file_seed = self.seed + self.current_file_idx * 1000
        self.current_reader = DataReader(
            data_src=self.current_data,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            seed=file_seed,
            with_replacement=self.with_replacement,
            auto_shard=self.auto_shard,
            keep_in_ram=True
        )
        
        # Start loading the new next file
        if len(self.file_paths) > 1:
            self._start_loading_next_file()

    def _calculate_total_tokens(self):
        """Calculate total tokens across all files (approximate initially)"""
        if self.current_data is not None:
            # For now, estimate based on current file
            tokens_per_file = len(self.current_data)
            self.num_tokens = tokens_per_file * len(self.file_paths)
        else:
            self.num_tokens = 0

    def __len__(self):
        if self.current_reader is not None:
            # Estimate based on current file
            return len(self.current_reader) * len(self.file_paths)
        return 0

    def __getitem__(self, idx):
        # For indexing, we need to figure out which file and local index
        if self.current_reader is None:
            raise RuntimeError("No data loaded")
            
        # Simple approach: use current file only for indexing
        # This is mainly for compatibility, most usage will be through sample_batch
        return self.current_reader[idx % len(self.current_reader)]

    def set_step(self, step):
        self.step = step
        if self.current_reader is not None:
            self.current_reader.set_step(step)

    def sample_batch(self):
        """Sample a batch from current file, switching files when needed"""
        if self.current_reader is None:
            raise RuntimeError("No data loaded")
        
        # For single file, just use the current reader
        if self.is_single_file:
            self.step += 1
            return self.current_reader.sample_batch()
        
        # For multiple files, check if we should switch files
        # We'll switch files periodically to ensure good mixing
        # Switch after consuming roughly 1/4 of the current file's batches
        batches_per_file_switch = max(1, self.current_reader.num_batches() // 4)
        
        if self.step > 0 and self.step % batches_per_file_switch == 0:
            print(f"Switching files after {batches_per_file_switch} batches")
            self._switch_to_next_file()
        
        self.step += 1
        return self.current_reader.sample_batch()

    def num_batches(self):
        if self.current_reader is not None:
            return self.current_reader.num_batches() * len(self.file_paths)
        return 0

    def __del__(self):
        """Cleanup async loader on deletion"""
        if hasattr(self, 'async_loader'):
            self.async_loader.shutdown()


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
