from __future__ import annotations


def test_nn_public_exports():
    from minigrad.nn import BCEWithLogitsLoss, Dropout2D, NLLLoss

    assert BCEWithLogitsLoss.__name__ == "BCEWithLogitsLoss"
    assert Dropout2D.__name__ == "Dropout2D"
    assert NLLLoss.__name__ == "NLLLoss"


def test_cli_entry_points_importable():
    from minigrad.cli import bench_ops, demo_linear_regression, demo_scalar, demo_xor, main

    assert callable(main)
    assert callable(demo_scalar)
    assert callable(demo_linear_regression)
    assert callable(demo_xor)
    assert callable(bench_ops)
