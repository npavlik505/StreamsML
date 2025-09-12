from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class BaseController(ABC):
    """Abstract interface for classical control strategies."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state before a new episode."""
        raise NotImplementedError

    @abstractmethod
    def compute_action(self, observation):
        """Return action for a given observation."""
        raise NotImplementedError
