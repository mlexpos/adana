import torch
from torch.nn import functional as F


def chunked_cross_entropy(hidden_states, weight, targets, chunk_size=256,
                          z_loss_fn=None, z_loss_coeff=0.0):
    """Compute lm_head projection + cross-entropy in chunks along seq dim.

    Avoids materializing the full (B, T, vocab_size) logits tensor by
    processing chunk_size tokens at a time.

    Args:
        hidden_states: (B, T, C) final hidden states
        weight: lm_head.weight parameter, shape (V, C)
        targets: (B, T) target token ids
        chunk_size: number of tokens per chunk along T
        z_loss_fn: optional callable(logits_2d) -> scalar z-loss
        z_loss_coeff: coefficient for z-loss

    Returns:
        loss: scalar cross-entropy (+ z_loss if provided)
        z_loss_value: scalar z-loss or None
    """
    B, T, C = hidden_states.shape
    total_tokens = B * T
    ce_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    z_loss_sum = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32) if z_loss_fn is not None else None

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        # (B, chunk, V)
        logits_chunk = F.linear(hidden_states[:, start:end, :], weight)
        targets_chunk = targets[:, start:end]
        ce_sum = ce_sum + F.cross_entropy(
            logits_chunk.reshape(-1, logits_chunk.size(-1)),
            targets_chunk.reshape(-1),
            ignore_index=-1,
            reduction='sum',
        )
        if z_loss_fn is not None:
            z_loss_sum = z_loss_sum + z_loss_fn(logits_chunk.reshape(-1, logits_chunk.size(-1))) * (B * (end - start))

    loss = ce_sum / total_tokens
    z_loss_value = z_loss_sum / total_tokens if z_loss_fn is not None else None

    if z_loss_value is not None:
        loss = loss + z_loss_coeff * z_loss_value

    return loss, z_loss_value
