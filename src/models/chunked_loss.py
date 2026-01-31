import torch
from torch.nn import functional as F

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False


def liger_cross_entropy(hidden_states, weight, targets,
                        z_loss_coeff=0.0, compute_z_loss=False,
                        compute_accuracy=False):
    """Compute lm_head projection + cross-entropy using Liger fused kernel.

    Drop-in replacement for chunked_cross_entropy that uses Liger's Triton
    kernel to fuse the linear projection and CE loss, avoiding materializing
    the full (B, T, vocab_size) logits tensor.

    Args:
        hidden_states: (B, T, C) final hidden states
        weight: lm_head.weight parameter, shape (V, C)
        targets: (B, T) target token ids
        z_loss_coeff: coefficient for z-loss
        compute_z_loss: whether to compute z-loss
        compute_accuracy: whether to compute token-level accuracy

    Returns:
        loss: scalar cross-entropy (+ z_loss if provided)
        z_loss_value: scalar z-loss or None
        accuracy: scalar accuracy or None (only when compute_accuracy=True)
    """
    if not _LIGER_AVAILABLE:
        raise ImportError(
            "liger-kernel is not installed. Install with: pip install liger-kernel"
        )

    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-1,
        lse_square_scale=z_loss_coeff if compute_z_loss else 0.0,
        reduction="mean",
        return_z_loss=compute_z_loss,
        return_token_accuracy=compute_accuracy,
        accum_dtype=torch.float32,
    )

    # Liger signature: (lin_weight, _input, target)
    # _input must be 2D: (B*T, C), target must be 1D: (B*T,)
    B, T, C = hidden_states.shape
    flat_hidden = hidden_states.reshape(B * T, C)
    flat_targets = targets.reshape(B * T)

    result = loss_fn(weight, flat_hidden, flat_targets)

    if compute_z_loss or compute_accuracy:
        # result is a CrossEntropyOutput namedtuple-like object
        loss = result.loss
        z_loss_value = result.z_loss if compute_z_loss else None
        accuracy = result.token_accuracy if compute_accuracy else None
    else:
        loss = result
        z_loss_value = None
        accuracy = None

    return loss, z_loss_value, accuracy


def _compute_chunk_ce_zloss(hidden_chunk, weight, targets_chunk):
    """Compute cross-entropy, z-loss, correct count, and total count for a single chunk."""
    logits = F.linear(hidden_chunk, weight)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets_chunk.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='sum')
    # z-loss: mean(logsumexp(logits, -1)^2) * num_tokens
    log_z = torch.logsumexp(flat_logits, dim=-1)
    zl = log_z.pow(2).sum()
    # accuracy: count correct predictions
    correct = (flat_logits.argmax(-1) == flat_targets).sum()
    total = flat_targets.numel()
    return ce, zl, correct, total


def _compute_chunk_ce(hidden_chunk, weight, targets_chunk):
    """Compute cross-entropy, correct count, and total count for a single chunk."""
    logits = F.linear(hidden_chunk, weight)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets_chunk.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='sum')
    correct = (flat_logits.argmax(-1) == flat_targets).sum()
    total = flat_targets.numel()
    return ce, correct, total


# Compile the inner per-chunk functions; the outer loop stays eager to
# ensure each chunk's logits are freed before the next is allocated.
_compiled_chunk_ce_zloss = torch.compile(_compute_chunk_ce_zloss)
_compiled_chunk_ce = torch.compile(_compute_chunk_ce)


@torch.compiler.disable
def chunked_cross_entropy(hidden_states, weight, targets, chunk_size=256,
                          z_loss_coeff=0.0, compute_z_loss=False,
                          compute_accuracy=False):
    """Compute lm_head projection + cross-entropy in chunks along seq dim.

    Avoids materializing the full (B, T, vocab_size) logits tensor by
    processing chunk_size tokens at a time.

    Args:
        hidden_states: (B, T, C) final hidden states
        weight: lm_head.weight parameter, shape (V, C)
        targets: (B, T) target token ids
        chunk_size: number of tokens per chunk along T
        z_loss_coeff: coefficient for z-loss
        compute_z_loss: whether to compute z-loss
        compute_accuracy: whether to compute token-level accuracy

    Returns:
        loss: scalar cross-entropy (+ z_loss if provided)
        z_loss_value: scalar z-loss or None
        accuracy: scalar accuracy or None (only when compute_accuracy=True)
    """
    B, T, C = hidden_states.shape
    total_tokens = B * T
    ce_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    z_loss_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32) if compute_z_loss else None
    correct_sum = 0
    total_sum = 0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_chunk = hidden_states[:, start:end, :]
        t_chunk = targets[:, start:end]
        if compute_z_loss:
            ce, zl, correct, total = _compiled_chunk_ce_zloss(h_chunk, weight, t_chunk)
            z_loss_sum = z_loss_sum + zl
        else:
            ce, correct, total = _compiled_chunk_ce(h_chunk, weight, t_chunk)
        ce_sum = ce_sum + ce
        if compute_accuracy:
            correct_sum = correct_sum + correct
            total_sum = total_sum + total

    loss = ce_sum / total_tokens
    z_loss_value = z_loss_sum / total_tokens if compute_z_loss else None
    accuracy = (correct_sum / total_sum).float() if compute_accuracy else None

    if z_loss_value is not None:
        loss = loss + z_loss_coeff * z_loss_value

    return loss, z_loss_value, accuracy
