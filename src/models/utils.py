import torch

from .base import GPTBase, LayerNorm
from .llama import Llama, RMSNorm
from .test import RMSNorm2, Test
from .enoki import Enoki
from .diloco import DiLoCo
from .qwen3 import Qwen3, RMSNorm as Qwen3RMSNorm
from .qwen3next import Qwen3Next

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    RMSNorm2,
    Qwen3RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        if args.use_pretrained != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "llama":
        model = Llama(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "test":
        model = Test(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "enoki":
        model = Enoki(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "diloco":
        # Backward compatibility alias for enoki
        model = DiLoCo(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "qwen3":
        model = Qwen3(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "qwen3next":
        model = Qwen3Next(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
