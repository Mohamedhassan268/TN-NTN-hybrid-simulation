"""Runtime device selection and provenance for reproducible SB3 runs."""
from __future__ import annotations

from typing import Any


def resolve_device(requested: str = "cpu") -> str:
    """Resolve an explicit SB3 device request without silently adopting CUDA."""
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto")
    if requested == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    return "cpu"


def device_provenance(requested: str, resolved: str) -> dict[str, Any]:
    """Return JSON-safe runtime/device details for a checkpoint or benchmark."""
    import torch

    result: dict[str, Any] = {
        "requested": requested,
        "resolved": resolved,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "cuda_device_index": index,
            "cuda_device_name": properties.name,
            "cuda_capability": list(torch.cuda.get_device_capability(index)),
            "cuda_total_memory_bytes": int(properties.total_memory),
        })
    return result


def meets_speedup_threshold(cpu_wall_seconds: float, cuda_wall_seconds: float, threshold: float = 4.0) -> bool:
    """Return whether CUDA meets the required CPU/CUDA end-to-end speed-up."""
    if cpu_wall_seconds <= 0 or cuda_wall_seconds <= 0:
        raise ValueError("benchmark wall times must be positive")
    return cpu_wall_seconds / cuda_wall_seconds >= threshold
