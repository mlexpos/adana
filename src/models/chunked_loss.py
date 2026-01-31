import torch
from torch.nn import functional as F


def _compute_chunk_ce_zloss(hidden_chunk, weight, targets_chunk):
    """Compute cross-entropy and z-loss for a single chunk. Compile-friendly."""
    logits = F.linear(hidden_chunk, weight)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets_chunk.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='sum')
    # z-loss: mean(logsumexp(logits, -1)^2) * num_tokens
    log_z = torch.logsumexp(flat_logits, dim=-1)
    zl = log_z.pow(2).sum()
    return ce, zl


def _compute_chunk_ce(hidden_chunk, weight, targets_chunk):
    """Compute cross-entropy only for a single chunk. Compile-friendly."""
    logits = F.linear(hidden_chunk, weight)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets_chunk.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='sum')
    return ce


# Compile the inner per-chunk functions; the outer loop stays eager to
# ensure each chunk's logits are freed before the next is allocated.
_compiled_chunk_ce_zloss = torch.compile(_compute_chunk_ce_zloss)
_compiled_chunk_ce = torch.compile(_compute_chunk_ce)


@torch.compiler.disable
def chunked_cross_entropy(hidden_states, weight, targets, chunk_size=256,
                          z_loss_coeff=0.0, compute_z_loss=False):
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

    Returns:
        loss: scalar cross-entropy (+ z_loss if provided)
        z_loss_value: scalar z-loss or None
    """
    B, T, C = hidden_states.shape
    total_tokens = B * T
    ce_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    z_loss_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32) if compute_z_loss else None

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_chunk = hidden_states[:, start:end, :]
        t_chunk = targets[:, start:end]
        if compute_z_loss:
            ce, zl = _compiled_chunk_ce_zloss(h_chunk, weight, t_chunk)
            z_loss_sum = z_loss_sum + zl
        else:
            ce = _compiled_chunk_ce(h_chunk, weight, t_chunk)
        ce_sum = ce_sum + ce

    loss = ce_sum / total_tokens
    z_loss_value = z_loss_sum / total_tokens if compute_z_loss else None

    if z_loss_value is not None:
        loss = loss + z_loss_coeff * z_loss_value

    return loss, z_loss_value
