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
        local_batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        """
        Optimized DataReader that handles multiple files with dual buffering.
        Always keeps current file and next file in RAM, with async loading.

        Args:
            data_files: Dictionary of file paths (train_00, train_01, etc.) or list of paths
            local_batch_size: Number of samples per batch PER WORKER (not global batch size)
            sequence_length: Length of each sequence
            seed: Random seed for reproducibility
            with_replacement: Whether to sample with replacement
            auto_shard: Whether to enable automatic sharding for distributed training
            keep_in_ram: Force keep data in RAM (always True for MultiFileDataReader)

        Note:
            In distributed training, local_batch_size is the per-worker batch size.
            The global batch size is local_batch_size * world_size.
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

        self.local_batch_size = local_batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.auto_shard = auto_shard

        # Initialize distributed settings
        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(f"Distributed MultiFileDataReader Initialized for Worker {self.rank}/{self.world_size}")
            if self.rank == 0:
                print("Rank 0: Reporting for duty")
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

        print(f"Number of batches in current file: {self.current_reader.num_batches()}")

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
            local_batch_size=self.local_batch_size,
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
        next_idx = (self.current_file_idx + 1) % len(self.file_paths)
        file_path = self.file_paths[next_idx]
        
        # Coordinate file loading across all workers to prevent desynchronization
        if self.auto_shard and dist.is_initialized():
            # Step 1: Check if this worker's async load is ready
            loaded_file = self.async_loader.get_loaded_file(timeout=5.0)
            async_ready = loaded_file is not None
            
            # Step 2: All workers report their async status to master (use GPU tensors for NCCL)
            async_status = torch.tensor([1 if async_ready else 0], dtype=torch.int32, device='cuda')
            if self.rank == 0:
                # Master collects status from all workers
                all_status = [torch.tensor([0], dtype=torch.int32, device='cuda') for _ in range(self.world_size)]
                dist.all_gather(all_status, async_status)
                all_ready = all(status.item() == 1 for status in all_status)
                
                if all_ready:
                    print(f"All workers ready - switching to pre-loaded file: {file_path}")
                else:
                    print(f"Some workers not ready - all workers will load synchronously: {file_path}")
                
                # Broadcast decision to all workers
                use_async = torch.tensor([1 if all_ready else 0], dtype=torch.int32, device='cuda')
                dist.broadcast(use_async, src=0)
            else:
                # Workers gather their status and receive decision
                all_status = [torch.tensor([0], dtype=torch.int32, device='cuda') for _ in range(self.world_size)]
                dist.all_gather(all_status, async_status)
                
                use_async = torch.tensor([0], dtype=torch.int32, device='cuda')
                dist.broadcast(use_async, src=0)
            
            # Step 3: All workers use the same loading method
            if use_async.item() == 1 and async_ready:
                # Use pre-loaded file
                _, data = loaded_file
                self.next_data = data
            else:
                # All workers load synchronously together
                print(f"Loading next file synchronously (coordinated): {file_path}")
                self.next_data = np.array(np.memmap(file_path, dtype=np.uint16, mode="r"))
                
        else:
            # Non-distributed case - use original logic
            loaded_file = self.async_loader.get_loaded_file(timeout=5.0)
            if loaded_file is not None:
                file_path, data = loaded_file
                print(f"Switching to pre-loaded file: {file_path}")
                self.next_data = data
            else:
                print(f"Loading next file synchronously (async not ready): {file_path}")
                self.next_data = np.array(np.memmap(file_path, dtype=np.uint16, mode="r"))
        
        # Switch files
        self.current_file_idx = next_idx
        self.current_data = self.next_data
        self.next_data = None
        
        # Create new DataReader for current file
        file_seed = self.seed + self.current_file_idx * 1000
        self.current_reader = DataReader(
            data_src=self.current_data,
            local_batch_size=self.local_batch_size,
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
        """Sample a batch from current file, switching files when current file is exhausted"""
        if self.current_reader is None:
            raise RuntimeError("No data loaded")
        
        # For single file, just use the current reader
        if self.is_single_file:
            self.step += 1
            return self.current_reader.sample_batch()
        
        # For multiple files, check if current file can provide a full batch
        # If not, switch to next file with master-worker coordination
        should_switch = False
        
        if self.auto_shard and dist.is_initialized():
            # Master-worker coordination for file switching
            if self.rank == 0:  # Master worker
                # Check if current file is exhausted (can't provide full batch)
                # Use current_file_num_batches() to get batches for current file only
                local_batches_available = self.current_file_num_batches()
                local_batches_consumed = self.step  # Counts number of local batches consumed by this worker
                should_switch = (local_batches_consumed >= local_batches_available)
                if should_switch:
                    print(f"Triggering switch to next file after exhausting current file ({self.step} local batches consumed, {local_batches_available} available per worker)")
                # Broadcast decision to all workers (use GPU tensor for NCCL)
                switch_tensor = torch.tensor([1 if should_switch else 0], dtype=torch.int32, device='cuda')
                dist.broadcast(switch_tensor, src=0)
            else:  # Worker processes
                # Receive decision from master (use GPU tensor for NCCL)
                switch_tensor = torch.tensor([0], dtype=torch.int32, device='cuda')
                dist.broadcast(switch_tensor, src=0)
                should_switch = bool(switch_tensor.item())
        else:
            # Fallback for non-distributed case
            local_batches_available = self.current_file_num_batches()
            should_switch = (self.step >= local_batches_available)
        
        # All workers switch together if current file is exhausted
        if should_switch:
            print(f"Switching files after exhausting current file ({self.step} batches)")
            self._switch_to_next_file()
            # Reset step counter for new file
            self.step = 0
        
        self.step += 1
        return self.current_reader.sample_batch()

    def num_batches(self):
        """
        Return total number of local batches across all files for this worker.

        In distributed training, this returns the per-worker batch count,
        not the global batch count across all workers.
        """
        if self.current_reader is not None:
            return self.current_reader.num_batches() * len(self.file_paths)
        return 0

    def current_file_num_batches(self):
        """Return number of local batches available in current file only for switching logic"""
        if self.current_reader is not None:
            return self.current_reader.num_batches_for_switching()
        return 0

    def state_dict(self):
        """Get the current state of the MultiFileDataReader for checkpointing"""
        state = {
            "current_file_idx": self.current_file_idx,
            "step": self.step,
        }
        # Also save the underlying DataReader state if it has state_dict
        if self.current_reader is not None and hasattr(self.current_reader, 'state_dict'):
            state["current_reader_state"] = self.current_reader.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Restore the state of the MultiFileDataReader from a checkpoint"""
        current_file_idx = state_dict.get("current_file_idx", 0)
        step = state_dict.get("step", 0)
        
        # Check if we need to switch to a different file
        if current_file_idx != self.current_file_idx:
            print(f"Restoring MultiFileDataReader: switching from file {self.current_file_idx} to file {current_file_idx}")
            # Switch to the correct file
            self.current_file_idx = current_file_idx
            # Load the correct file
            self._load_current_file()
            # Start loading next file if there are multiple files
            if len(self.file_paths) > 1:
                self._start_loading_next_file()
        else:
            # Same file, just need to reload it to ensure it's in memory
            print(f"Restoring MultiFileDataReader: staying on file {current_file_idx}, reloading data")
            self._load_current_file()
            if len(self.file_paths) > 1:
                self._start_loading_next_file()
        
        # Restore the step counter
        self.step = step
        
        # Restore the underlying DataReader state if available
        if self.current_reader is not None:
            if "current_reader_state" in state_dict and hasattr(self.current_reader, 'load_state_dict'):
                self.current_reader.load_state_dict(state_dict["current_reader_state"])
            else:
                # Fallback: just set the step
                self.current_reader.set_step(step)
        
        print(f"MultiFileDataReader restored: file_idx={self.current_file_idx}, step={self.step}")

    def __del__(self):
        """Cleanup async loader on deletion"""
        if hasattr(self, 'async_loader'):
            self.async_loader.shutdown()


class DataReader:
    """
    DataReader for single data file with support for distributed training.

    Args:
        data_src: Path to data file or numpy array
        local_batch_size: Number of samples per batch PER WORKER (not global batch size)
        sequence_length: Length of each sequence
        seed: Random seed for reproducibility
        with_replacement: Whether to sample with replacement
        auto_shard: Whether to enable automatic sharding for distributed training
        keep_in_ram: Whether to keep data in RAM

    Note:
        In distributed training, local_batch_size is the per-worker batch size.
        The global batch size is local_batch_size * world_size.
        Each worker gets different samples automatically when auto_shard=True.
    """
    def __init__(
        self,
        data_src,
        local_batch_size,
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

        self.local_batch_size = local_batch_size
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

        # Sampling without replacement state
        self.last_epoch = None
        self.order = None  # Permutation of sequence indices for current epoch
        self.epoch_offset = None  # Random offset within sequences for current epoch
        self.step = 0  # Number of sample_batch() calls made by this worker

        # Number of local batches this worker will process from the current file/epoch
        # In distributed training: total_possible_batches // world_size
        # In single worker: total_possible_batches
        # This represents how many times this worker can call sample_batch()
        # before exhausting its assigned portion of the data
        self.num_batches_of_seqlen = 0

        # Initialize epoch data (sets num_batches_of_seqlen for both cases)
        if not with_replacement:
            self._shuffle_epoch(0)
        else:
            # For with_replacement, calculate finite batch count for file switching
            self._calculate_finite_batches()

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
        # Return an array of token indices of length self.local_batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.local_batch_size)

    def _calculate_finite_batches(self):
        """Calculate finite number of batches for file switching in with_replacement mode"""
        if self.world_size > 1:
            # Calculate based on available sequences, similar to without_replacement logic
            total_sequences = (len(self)) // self.sequence_length - 1
            total_possible_local_batches = total_sequences // self.local_batch_size
            self.num_batches_of_seqlen = total_possible_local_batches // self.world_size
        else:
            total_sequences = (len(self)) // self.sequence_length - 1
            self.num_batches_of_seqlen = total_sequences // self.local_batch_size

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        # Calculate num_batches_of_seqlen: number of local batches this worker will process
        #
        # Example with 1000 sequences, local_batch_size=16, world_size=2:
        # - total_possible_local_batches = 1000 // 16 = 62 local batches possible
        # - In distributed: each worker processes 62 // 2 = 31 local batches
        # - Worker 0 gets global batches [0, 2, 4, ...] -> 31 batches total
        # - Worker 1 gets global batches [1, 3, 5, ...] -> 31 batches total
        # - self.step goes from 0 to 30 for each worker
        if self.world_size > 1:
            # In distributed training, each worker processes every world_size-th batch
            self.total_possible_local_batches = len(self.order) // self.local_batch_size
            # Each worker gets every world_size-th batch, so divide by world_size
            self.num_batches_of_seqlen = self.total_possible_local_batches // self.world_size
            #print(f"Shuffling. Rank {self.rank}. length of order: {len(self.order)}, total_possible_local_batches: {self.total_possible_local_batches}, num_batches_of_seqlen: {self.num_batches_of_seqlen}, sequence_length: {self.sequence_length}")
        else:
            # Single worker case: process all possible batches
            #print(f"single worker case, world_size: {self.world_size}, local_batch_size: {self.local_batch_size}, sequence_length: {self.sequence_length}")
            self.total_possible_local_batches = len(self.order) // self.local_batch_size
            self.num_batches_of_seqlen = self.total_possible_local_batches

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.local_batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        #
        # Distributed sampling logic:
        # - All workers share the same permutation (from same epoch seed)
        # - Worker 0 gets batches 0, world_size, 2*world_size, ...
        # - Worker 1 gets batches 1, world_size+1, 2*world_size+1, ...
        # - This ensures no overlap between workers
        batch_idx = self.world_size * step + self.rank
        
        # epoch_length must be the GLOBAL number of batches in an epoch across all ranks
        # When batch_idx exceeds this, we move to the next epoch (reshuffle data)
        epoch_length = self.total_possible_local_batches
        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            #print(f"Shuffling. Rank {self.rank}.")
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.local_batch_size
        end = start + self.local_batch_size
        #print(f"rank: {self.rank}, epoch: {epoch}, epoch_length: {epoch_length}, epoch_idx: {epoch_idx}, start: {start}, end: {end}")
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        """
        Return number of local batches available for this worker.

        In distributed training, this returns the per-worker batch count,
        not the global batch count across all workers.

        For with_replacement=True: Returns theoretical maximum based on data size
        For with_replacement=False: Returns actual batches this worker will process
        """
        if self.with_replacement:
            # With replacement: theoretically unlimited, but limit to data size
            return self.num_tokens // self.local_batch_size
        return self.num_batches_of_seqlen

    def num_batches_for_switching(self):
        """
        Return number of local batches for file switching logic.

        This returns the finite number of batches this worker should process
        from the current file before switching, ensuring consistent behavior
        for both with_replacement and without_replacement modes.
        """
        return self.num_batches_of_seqlen

    def state_dict(self):
        """Get the current state of the DataReader for checkpointing"""
        state = {
            "step": self.step,
        }
        # Save epoch-related state for without_replacement mode
        if not self.with_replacement:
            state["last_epoch"] = self.last_epoch if self.last_epoch is not None else 0
            state["order"] = self.order.copy() if self.order is not None else None
            state["epoch_offset"] = self.epoch_offset if self.epoch_offset is not None else 0
            state["total_possible_local_batches"] = getattr(self, 'total_possible_local_batches', 0)
            state["num_batches_of_seqlen"] = self.num_batches_of_seqlen
        return state

    def load_state_dict(self, state_dict):
        """Restore the state of the DataReader from a checkpoint"""
        self.step = state_dict.get("step", 0)
        
        # Restore epoch-related state for without_replacement mode
        if not self.with_replacement:
            last_epoch = state_dict.get("last_epoch", 0)
            order = state_dict.get("order", None)
            epoch_offset = state_dict.get("epoch_offset", 0)
            total_possible_local_batches = state_dict.get("total_possible_local_batches", 0)
            num_batches_of_seqlen = state_dict.get("num_batches_of_seqlen", 0)
            
            if order is not None:
                self.last_epoch = last_epoch
                self.order = order
                self.epoch_offset = epoch_offset
                self.total_possible_local_batches = total_possible_local_batches
                self.num_batches_of_seqlen = num_batches_of_seqlen
            else:
                # If order is not available, reinitialize epoch state from step
                # This may not perfectly restore state but is better than nothing
                print(f"Warning: DataReader state_dict missing order, reinitializing epoch state from step={self.step}")
                # Calculate current epoch from step
                if hasattr(self, 'total_possible_local_batches') and self.total_possible_local_batches > 0:
                    batch_idx = self.world_size * self.step + self.rank
                    epoch = batch_idx // self.total_possible_local_batches
                    self._shuffle_epoch(epoch)
                else:
                    # Fallback: initialize first epoch
                    self._shuffle_epoch(0)
