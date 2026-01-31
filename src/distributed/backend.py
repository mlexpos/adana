from typing import List

import torch


class DistributedBackend(object):
    def __init__(self, args):
        pass

    def transform_model(self, model):
        raise NotImplementedError

    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        raise NotImplementedError

    def is_master_process(self) -> bool:
        raise NotImplementedError

    def get_adjusted_args_for_process(self, args):
        raise NotImplementedError

    def get_raw_model(self, model):
        raise NotImplementedError

    def translate_model_parameter_name_for_node(self, parameter_name) -> List[str]:
        raise NotImplementedError

    def get_world_size(self):
        raise NotImplementedError

    def clip_grad_norm_(self, parameters, max_norm):
        """Clip gradient norm.  Override in subclasses for custom DTensor handling."""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def all_ranks_checkpoint(self):
        """Whether checkpoint save/load requires all ranks to participate.

        Returns True for FSDP (sharded state dicts), False for DDP/single.
        """
        return False

    def finalize(self):
        pass
