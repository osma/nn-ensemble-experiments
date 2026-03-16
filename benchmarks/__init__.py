"""benchmarks package.

This package exists so benchmark scripts can use absolute imports like:

    from benchmarks.device import get_device

Do not place shell commands here. Run benchmark scripts from your shell, e.g.:

    uv run python benchmarks/torch_mean_residual_softmax_global_l2_anchor.py --dataset yso-fi
"""
