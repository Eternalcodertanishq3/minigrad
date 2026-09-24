from __future__ import annotations


def test_nn_public_exports():
    from minigrad.nn import (
        BCEWithLogitsLoss,
        Dropout2D,
        NLLLoss,
        NeuralODE,
        ReversibleBlock,
        ReversibleSequential,
        DistributionalLinear,
        DistributionalSequential,
        GaussianNLLLoss,
    )

    assert BCEWithLogitsLoss.__name__ == "BCEWithLogitsLoss"
    assert Dropout2D.__name__ == "Dropout2D"
    assert NLLLoss.__name__ == "NLLLoss"
    assert NeuralODE.__name__ == "NeuralODE"
    assert ReversibleBlock.__name__ == "ReversibleBlock"
    assert ReversibleSequential.__name__ == "ReversibleSequential"
    assert DistributionalLinear.__name__ == "DistributionalLinear"
    assert DistributionalSequential.__name__ == "DistributionalSequential"
    assert GaussianNLLLoss.__name__ == "GaussianNLLLoss"


def test_top_level_sutra_exports():
    from minigrad import SUTRA, NeuralODE, odeint, AdaptiveStepTelemetry

    assert callable(odeint)
    assert NeuralODE.__name__ == "NeuralODE"
    assert "dopri5" in SUTRA.solvers()
    assert AdaptiveStepTelemetry.__name__ == "AdaptiveStepTelemetry"


def test_top_level_avyaya_exports():
    from minigrad import (
        AVYAYA,
        ReversibleBlock,
        ReversibleSequential,
        ReconstructionTelemetry,
    )

    assert ReversibleBlock.__name__ == "ReversibleBlock"
    assert ReversibleSequential.__name__ == "ReversibleSequential"
    assert ReconstructionTelemetry.__name__ == "ReconstructionTelemetry"
    assert AVYAYA.ReversibleBlock is ReversibleBlock


def test_top_level_pramana_exports():
    from minigrad import (
        PRAMANA,
        DistributionalTensor,
        DistributionalLinear,
        DistributionalSequential,
        GaussianNLLLoss,
        PramanaTelemetry,
    )

    assert DistributionalTensor.__name__ == "DistributionalTensor"
    assert DistributionalLinear.__name__ == "DistributionalLinear"
    assert DistributionalSequential.__name__ == "DistributionalSequential"
    assert GaussianNLLLoss.__name__ == "GaussianNLLLoss"
    assert PramanaTelemetry.__name__ == "PramanaTelemetry"
    assert PRAMANA.DistributionalTensor is DistributionalTensor
    assert PRAMANA.DistributionalLinear is DistributionalLinear


def test_cli_entry_points_importable():
    from minigrad.cli import bench_ops, demo_linear_regression, demo_scalar, demo_xor, main

    assert callable(main)
    assert callable(demo_scalar)
    assert callable(demo_linear_regression)
    assert callable(demo_xor)
    assert callable(bench_ops)
