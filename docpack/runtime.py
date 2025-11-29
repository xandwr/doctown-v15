"""
Runtime configuration for docpack.

Manages CPU/GPU mode detection and batch size optimization.
Provides a unified configuration that flows through the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RuntimeConfig:
    """
    Runtime configuration for embedding and LLM operations.

    Attributes:
        force_cpu: Force CPU-only mode (no CUDA/Metal)
        embedding_model: Ollama model for embeddings
        embedding_batch_size: Batch size for embedding operations
        llm_model: Ollama model for answer generation
        summarize_model: Ollama model for summarization
        parallel_workers: Number of parallel workers for CPU mode
        verbose: Print debug information
    """

    force_cpu: bool = False

    # Embedding settings
    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 32

    # LLM settings
    llm_model: str = "qwen3:8b"
    summarize_model: str = "qwen3:1.7b"
    summarize_batch_size: int = 8

    # Parallelism
    parallel_workers: int = 4

    # Debug
    verbose: bool = False

    def __post_init__(self):
        """Adjust settings based on CPU mode."""
        if self.force_cpu:
            # Smaller batches for CPU to avoid memory pressure
            self.embedding_batch_size = 4
            self.summarize_batch_size = 2
            # More parallel workers to compensate
            self.parallel_workers = max(4, os.cpu_count() or 4)


def detect_cuda_available() -> bool:
    """
    Check if CUDA is available via Ollama.

    Note: Ollama handles GPU detection automatically, but we can
    check for GPU availability to inform the user.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_runtime_config(
    force_cpu: bool = False,
    embedding_model: str | None = None,
    llm_model: str | None = None,
    summarize_model: str | None = None,
    verbose: bool = False,
) -> RuntimeConfig:
    """
    Create a runtime configuration with sensible defaults.

    Args:
        force_cpu: Force CPU-only mode
        embedding_model: Override embedding model
        llm_model: Override LLM model for answers
        summarize_model: Override model for summarization
        verbose: Enable verbose output

    Returns:
        Configured RuntimeConfig instance
    """
    config = RuntimeConfig(
        force_cpu=force_cpu,
        verbose=verbose,
    )

    if embedding_model:
        config.embedding_model = embedding_model
    if llm_model:
        config.llm_model = llm_model
    if summarize_model:
        config.summarize_model = summarize_model

    # Auto-detect and warn
    if verbose and not force_cpu:
        has_cuda = detect_cuda_available()
        if has_cuda:
            print("CUDA detected - Ollama will use GPU acceleration")
        else:
            print("No CUDA detected - running on CPU")

    return config


# Global config instance (can be set by CLI/web server)
_global_config: RuntimeConfig | None = None


def set_global_config(config: RuntimeConfig) -> None:
    """Set the global runtime configuration."""
    global _global_config
    _global_config = config


def get_global_config() -> RuntimeConfig:
    """Get the global runtime configuration, creating default if needed."""
    global _global_config
    if _global_config is None:
        _global_config = RuntimeConfig()
    return _global_config
