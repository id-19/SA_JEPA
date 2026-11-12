"""Name the scaled-dot-product kernel a call actually dispatched."""

import torch


def sdpa_kernels(fn):
    """Run fn() and return the ATen SDPA kernels it hit.

    The public aten::scaled_dot_product_attention op is left out.
    A flash dispatch looks like aten::_scaled_dot_product_flash_attention_for_cpu.
    A fallback looks like aten::_scaled_dot_product_attention_math
    or aten::_scaled_dot_product_attention_math_for_mps.
    """
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    skip = {"aten::scaled_dot_product_attention"}
    return [
        e.key
        for e in prof.key_averages()
        if "scaled_dot_product" in e.key and e.key not in skip
    ]
