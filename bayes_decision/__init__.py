"""
bayes_decision: Bayesian Decision-Theoretic Framework for Small-Sample Experiments
==================================================================================

Core idea: instead of relying on alpha = 0.05, specify costs of false positives
and inconclusive outcomes explicitly, then derive the optimal decision rule.

Quick start::

    from bayes_decision import BayesDecision
    result = BayesDecision(y_treatment, y_control)
    print(result)

GPU acceleration::

    from bayes_decision import expected_loss_gpu
    results = expected_loss_gpu(n=5, delta=0.5, reps=1_000_000, device="cuda")
"""

from bayes_decision.core import (
    bayes_posterior,
    bayes_factor,
    bayes_decision,
    BayesDecision,
    expected_loss,
    realized_loss,
    realized_loss_magnitude,
)
from bayes_decision.calibrate import (
    calibrate_loss,
    optimal_alpha,
    base_rate_analysis,
    sequential_decision,
    estimate_pi1,
    pipeline_fdr,
)
from bayes_decision.parallel import (
    expected_loss_grid,
    calibrate_loss_parallel,
    check_system,
)
from bayes_decision.gpu import (
    expected_loss_gpu,
    expected_loss_grid_gpu,
)
from bayes_decision.plots import (
    plot_decision_boundary,
    plot_base_rate,
    plot_loss_surface,
)

__version__ = "0.1.0"
__all__ = [
    "bayes_posterior", "bayes_factor", "bayes_decision", "BayesDecision",
    "expected_loss", "realized_loss", "realized_loss_magnitude",
    "calibrate_loss", "optimal_alpha", "base_rate_analysis", "sequential_decision",
    "estimate_pi1", "pipeline_fdr",
    "expected_loss_grid", "calibrate_loss_parallel", "check_system",
    "expected_loss_gpu", "expected_loss_grid_gpu",
    "plot_decision_boundary", "plot_base_rate", "plot_loss_surface",
]
