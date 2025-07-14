from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    """Abstract interface for RL agents."""

    @abstractmethod
    def choose_action(self, observation):
        """Return action for given observation."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Update agent using transition data."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, directory: Path, tag: str) -> None:
        """Save any internal model parameters to *directory* with *tag*."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint: Path) -> None:
        """Load model parameters from *checkpoint* path."""
        raise NotImplementedError
