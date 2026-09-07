"""RobustAnalog (Shi et al., DAC 2022, arXiv 2207.06412) re-implementation.

A from-scratch re-implementation of RobustAnalog as an ML/RL baseline for the
paper's §VI-B/§VI-C Tier-1 comparison. The original authors' code link is
dead, so this is implemented directly from the paper:

  - Algorithm 1 (multi-task DDPG training loop)         -> train.py
  - Reward formula, eqs 2-3                              -> multi_task_env.py
  - PCGrad gradient surgery (Yu et al. 2020)            -> pcgrad.py
  - k-means task pruning, §3.4                           -> pruning.py
  - DDPG actor/critic networks, §4.2                     -> agent.py
  - per-task replay buffers, §3.3/§4.2                   -> replay_buffer.py

Track B (our-framework integration):
  - continuous-xi -> discrete K-corner adapter           -> benchmark_adapter.py


IMPORTANT: RobustAnalog uses ITS OWN reward (relative-distance-to-
spec, clipped at 0.2 when satisfied). We do NOT substitute our softplus
surrogate — keeping the RL comparison honest depends on this separation.
"""
