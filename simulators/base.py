"""BlackBoxSimulator abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BlackBoxSimulator(ABC):
    """Abstract base class for any zero-order black-box loss f(x, xi) -> R.

    Implementations should treat *failures* as data: instead of raising, return a
    large penalty value (``LOSS_PENALTY`` below) and increment ``self.n_failures``.
    """

    LOSS_PENALTY: float = 1.0e6

    def __init__(self) -> None:
        self.n_calls: int = 0
        self.n_failures: int = 0

    @abstractmethod
    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        """Return scalar loss f(x, xi). Must NOT raise on simulation failure."""

    def evaluate_batch(self, xs: Iterable[np.ndarray], xis: Iterable[np.ndarray]) -> np.ndarray:
        """Default serial implementation. Subclasses may override for parallelism."""
        out = []
        for x, xi in zip(xs, xis):
            out.append(self.evaluate(x, xi))
        return np.asarray(out, dtype=float)

    def reset_counters(self) -> None:
        self.n_calls = 0
        self.n_failures = 0
