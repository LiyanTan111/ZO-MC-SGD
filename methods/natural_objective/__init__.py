"""Literature-natural yield-MC objective for the BO/CMA-ES/PSO sanity check.

Standard analog-yield-optimization literature runs BO/CMA-ES/PSO on a batched
yield estimate Ŷ(x) = (1/n_mc) Σ 1{all specs satisfied at (x, ξ_i)} (n_mc
simulations per query), NOT on our single-sample softplus surrogate. This module
provides that objective so the SAME optimizer implementations can be re-run on
the literature-natural objective and compared head-to-head against our existing
softplus-objective data — validating that the softplus framing is the steel-man.

"""
