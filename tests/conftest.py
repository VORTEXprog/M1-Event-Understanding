"""CPU-only public fixtures. No pretrained weights or external data."""
import pytest
import torch


@pytest.fixture(autouse=True)
def ordinary_cpu_defaults(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    dtype = torch.get_default_dtype()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_default_dtype(dtype)
