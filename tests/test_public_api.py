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
        NeuralPredicate,
        NeuralRelation,
        SemanticLoss,
        LIFCell,
        LIFLayer,
        SpikingLinear,
        SpikingSequential,
        MembraneDecoder,
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
    assert NeuralPredicate.__name__ == "NeuralPredicate"
    assert NeuralRelation.__name__ == "NeuralRelation"
    assert SemanticLoss.__name__ == "SemanticLoss"
    assert LIFCell.__name__ == "LIFCell"
    assert LIFLayer.__name__ == "LIFLayer"
    assert SpikingLinear.__name__ == "SpikingLinear"
    assert SpikingSequential.__name__ == "SpikingSequential"
    assert MembraneDecoder.__name__ == "MembraneDecoder"


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


def test_top_level_tarka_exports():
    from minigrad import (
        TARKA,
        LogicTensor,
        NeuralPredicate,
        NeuralRelation,
        SemanticLoss,
        TarkaTelemetry,
    )

    assert LogicTensor.__name__ == "LogicTensor"
    assert NeuralPredicate.__name__ == "NeuralPredicate"
    assert NeuralRelation.__name__ == "NeuralRelation"
    assert SemanticLoss.__name__ == "SemanticLoss"
    assert TarkaTelemetry.__name__ == "TarkaTelemetry"
    assert TARKA.LogicTensor is LogicTensor
    assert TARKA.NeuralPredicate is NeuralPredicate
    assert TARKA.NeuralRelation is NeuralRelation
    assert TARKA.SemanticLoss is SemanticLoss


def test_top_level_spanda_exports():
    from minigrad import (
        SPANDA,
        surrogate_spike,
        LIFCell,
        LIFLayer,
        SpikingLinear,
        SpikingSequential,
        RateEncoder,
        DirectEncoder,
        RateDecoder,
        MembraneDecoder,
        SpandaTelemetry,
    )

    assert callable(surrogate_spike)
    assert LIFCell.__name__ == "LIFCell"
    assert LIFLayer.__name__ == "LIFLayer"
    assert SpikingLinear.__name__ == "SpikingLinear"
    assert SpikingSequential.__name__ == "SpikingSequential"
    assert RateEncoder.__name__ == "RateEncoder"
    assert DirectEncoder.__name__ == "DirectEncoder"
    assert RateDecoder.__name__ == "RateDecoder"
    assert MembraneDecoder.__name__ == "MembraneDecoder"
    assert SpandaTelemetry.__name__ == "SpandaTelemetry"
    assert SPANDA.LIFCell is LIFCell
    assert SPANDA.SpikingLinear is SpikingLinear


def test_cli_entry_points_importable():
    from minigrad.cli import bench_ops, demo_linear_regression, demo_scalar, demo_xor, main

    assert callable(main)
    assert callable(demo_scalar)
    assert callable(demo_linear_regression)
    assert callable(demo_xor)
    assert callable(bench_ops)
