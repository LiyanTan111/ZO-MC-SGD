"""PCGrad — Gradient Surgery for Multi-Task Learning (Yu et al., NeurIPS 2020).

RobustAnalog applies PCGrad to the K per-task critic losses and the K per-task
actor losses (paper §3.3). PCGrad resolves conflicting gradients: for each task
gradient g_i, it projects out the component of g_i that conflicts (negative
cosine) with every other task gradient g_j, then sums the de-conflicted
gradients.

This module is intentionally standalone so it can be reused and
unit-tested in isolation. It operates on flat 1-D gradient tensors; callers are
responsible for flattening per-parameter grads and scattering the result back.

Reference implementation cross-checked against
github.com/WeiChengTseng/Pytorch-PCGrad on a synthetic example (see tests).
"""
from __future__ import annotations

from typing import List, Optional

import torch


def project_conflicting(
    grads: List[torch.Tensor],
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply PCGrad to a list of K flat per-task gradients; return their sum.

    Args:
      grads: list of K 1-D tensors, each the gradient of one task's loss
             w.r.t. a shared flat parameter vector. All must share shape.
      rng:   optional torch.Generator for the random task-order permutation
             (PCGrad iterates other tasks in random order). Pass a seeded
             generator for deterministic / order-invariant behaviour.

    Returns:
      A single 1-D tensor: the sum of the de-conflicted per-task gradients.
    """
    if len(grads) == 0:
        raise ValueError("PCGrad needs at least one task gradient")
    k = len(grads)
    shape = grads[0].shape
    for g in grads:
        if g.shape != shape:
            raise ValueError("all task gradients must share shape")

    # Single task: nothing to de-conflict; return it unchanged.
    if k == 1:
        return grads[0].clone()

    pc_grads = []
    for i in range(k):
        g_i = grads[i].clone()
        # Visit the other tasks in a random order (paper's prescription).
        order = torch.randperm(k, generator=rng)
        for j in order.tolist():
            if j == i:
                continue
            g_j = grads[j]
            inner = torch.dot(g_i, g_j)
            if inner < 0:
                # Remove the conflicting component of g_i along g_j.
                denom = torch.dot(g_j, g_j)
                if denom > 0:
                    g_i = g_i - (inner / denom) * g_j
        pc_grads.append(g_i)

    return torch.stack(pc_grads, dim=0).sum(dim=0)


def pcgrad_backward(
    losses: List[torch.Tensor],
    parameters: List[torch.nn.Parameter],
    rng: Optional[torch.Generator] = None,
) -> None:
    """Compute PCGrad-combined gradients for ``losses`` and set ``.grad``.

    Convenience wrapper for the training loop: takes K scalar task losses and
    a list of shared parameters, computes each task's flat gradient, applies
    :func:`project_conflicting`, and scatters the de-conflicted sum back into
    each parameter's ``.grad`` (so a subsequent ``optimizer.step()`` uses it).

    Does NOT zero existing grads or call ``optimizer.step()`` — the caller
    controls that.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return

    # Per-task flat gradient via autograd (retain graph across tasks).
    flat_grads = []
    for t, loss in enumerate(losses):
        retain = t < len(losses) - 1
        grads = torch.autograd.grad(
            loss, params, retain_graph=retain, allow_unused=True
        )
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)
        ])
        flat_grads.append(flat)

    combined = project_conflicting(flat_grads, rng=rng)

    # Scatter the combined flat gradient back into each parameter's .grad.
    offset = 0
    for p in params:
        numel = p.numel()
        chunk = combined[offset:offset + numel].reshape(p.shape)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad = p.grad + chunk
        offset += numel
