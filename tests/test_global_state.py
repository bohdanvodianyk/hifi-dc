"""Global torch settings changed inside one test must not leak into later tests.

configure_determinism() turns off the fused SDPA kernels process-wide; those flags also pick the CPU
attention kernel, so a leak made the bit-exact CPU golden tests fail when the GPU tests ran first.
The two tests below run in file order: the first changes the settings, the second checks they are back.
"""
import torch

from hifidc.bench import configure_determinism

BEFORE = {}


def test_a_changes_global_settings():
    cuda = torch.backends.cuda
    BEFORE.update(flash=cuda.flash_sdp_enabled(), mem=cuda.mem_efficient_sdp_enabled(),
                  det=torch.are_deterministic_algorithms_enabled())
    configure_determinism()
    assert not cuda.flash_sdp_enabled()


def test_b_settings_are_restored():
    cuda = torch.backends.cuda
    assert cuda.flash_sdp_enabled() == BEFORE['flash']
    assert cuda.mem_efficient_sdp_enabled() == BEFORE['mem']
    assert torch.are_deterministic_algorithms_enabled() == BEFORE['det']
