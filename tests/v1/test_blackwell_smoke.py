# SPDX-License-Identifier: Apache-2.0
"""Blackwell smoke tests for the compiled LMCache CUDA extension.

Runs on any Blackwell-class GPU (compute capability >= 10.0), covering BOTH
Grace-Blackwell aarch64 (GB200 sm_100, GB300 sm_103) and x86 Blackwell
(B200 sm_100, B300 sm_103). Skips on CPU and pre-Blackwell GPUs, so it is a
no-op elsewhere in CI.

The kernel-launch test is the meaningful check: it proves the installed wheel
contains executable code for THIS device's compute capability. A wheel built
without the device's SASS/PTX imports fine (dlopen) but fails at launch with
"no kernel image is available for execution on the device".
"""

# Standard
import platform

# Third Party
import pytest
import torch


def _is_blackwell() -> bool:
    """Return True on a CUDA device with compute capability >= 10.0."""
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 10
    )


pytestmark = pytest.mark.skipif(
    not _is_blackwell(), reason="requires a Blackwell (sm_100+) GPU"
)


def test_c_ops_imports_on_blackwell() -> None:
    """The compiled extension loads on this Blackwell host (x86_64 or aarch64)."""
    # First Party
    import lmcache.c_ops  # noqa: F401


def test_c_ops_kernel_executes_on_this_device() -> None:
    """Launch a real c_ops kernel to prove the wheel covers this device's arch.

    Arch-agnostic: passes on sm_100 and sm_103, on both x86_64 and aarch64. A
    wheel missing this device's SASS/PTX raises a CUDA "no kernel image is
    available for execution on the device" error here instead of returning.
    """
    # First Party
    import lmcache.c_ops as c_ops

    cc = torch.cuda.get_device_capability(0)
    counts = torch.randint(0, 16, (2, 4, 16), dtype=torch.int8, device="cuda:0")
    cdf = c_ops.calculate_cdf(counts, 16)
    torch.cuda.synchronize()
    assert cdf.is_cuda and cdf.shape[0] == counts.shape[0], (
        f"calculate_cdf returned an unexpected result on "
        f"sm_{cc[0]}{cc[1]} / {platform.machine()}"
    )


def test_pinned_d2h_roundtrip() -> None:
    """Exercise the pinned-memory D2H path LMCache's CPU-offload tier relies on."""
    dev = torch.device("cuda:0")
    src = torch.randn(4096, 128, dtype=torch.float16, device=dev)
    dst = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.allclose(dst.to(dev), src)
