"""Tests for bayesdecision core functionality."""

import numpy as np
import pytest

from bayesdecision import (
    bayes_posterior, bayes_factor, bayes_decision, BayesDecision,
    expected_loss, realized_loss, realized_loss_magnitude,
    optimal_alpha, sequential_decision, base_rate_analysis,
    estimate_pi1, check_system, calibrate_loss, pipeline_fdr,
)


class TestPosterior:
    def test_raw_data(self):
        rng = np.random.default_rng(42)
        y1 = rng.normal(0.5, 1, 10)
        y2 = rng.normal(0, 1, 10)
        post = bayes_posterior(y1, y2)
        assert 0 <= post.p_positive <= 1
        assert post.n1 == 10
        assert post.sigma_post > 0

    def test_summary_stats(self):
        post = bayes_posterior(2.5, 1.8, n1=10, n2=10, sd1=1.2, sd2=0.9, summary=True)
        assert abs(post.d_hat - 0.7) < 1e-10
        assert post.p_positive > 0.5

    def test_summary_requires_params(self):
        with pytest.raises(ValueError):
            bayes_posterior(2.5, 1.8, summary=True)


class TestBayesFactor:
    def test_null_evidence(self):
        assert bayes_factor(d=0, se=0.5) < 1

    def test_strong_evidence(self):
        assert bayes_factor(d=3, se=0.5) > 10


class TestDecision:
    def test_returns_all_methods(self):
        rng = np.random.default_rng(42)
        y1 = rng.normal(1, 1, 8)
        y2 = rng.normal(0, 1, 8)
        dec = bayes_decision(y1, y2)
        assert isinstance(dec, BayesDecision)
        assert dec.nhst_decision in ("positive", "negative", "inconclusive")
        assert dec.bayes95_decision in ("positive", "negative", "inconclusive")
        assert dec.bayesopt_decision in ("positive", "negative", "inconclusive")
        assert len(dec.expected_losses) == 3
        assert 0.5 < dec.threshold < 1

    def test_repr(self):
        rng = np.random.default_rng(42)
        dec = bayes_decision(rng.normal(1, 1, 5), rng.normal(0, 1, 5))
        text = repr(dec)
        assert "Bayesian Decision" in text
        assert "Optimal action" in text


class TestExpectedLoss:
    def test_returns_four_methods(self):
        el = expected_loss(n=5, delta=0.5, reps=500)
        assert len(el) == 4
        methods = {e["method"] for e in el}
        assert methods == {"NHST", "Bayes-0.95", "Bayes-Optimal", "Bayes-Factor"}
        assert all(e["expected_loss"] >= 0 for e in el)


class TestRealizedLoss:
    def test_correct_positive(self):
        assert realized_loss("positive", 0.5) == 0

    def test_waste(self):
        assert realized_loss("inconclusive", 0.5) == 0.2

    def test_false_positive(self):
        assert realized_loss("positive", 0) == 1.0

    def test_correct_inconclusive(self):
        assert realized_loss("inconclusive", 0) == 0


class TestOptimalAlpha:
    def test_returns_valid(self):
        oa = optimal_alpha(n=10, delta1=0.5)
        assert 0 < oa["alpha_opt"] < 1
        assert oa["expected_loss_opt"] <= oa["expected_loss_005"]


class TestSequential:
    def test_runs(self):
        sq = sequential_decision(n_pilot=5, n_confirm=20, delta=0.5, reps=500)
        assert 0 <= sq["proceed_rate"] <= 1
        assert 0 <= sq["discovery_rate"] <= 1


class TestCheckSystem:
    def test_returns_dict(self):
        info = check_system(verbose=False)
        assert "cpu_cores" in info
        assert "has_torch" in info
        assert info["cpu_cores"] >= 1


# ==============================================================
# New DGP models
# ==============================================================

class TestExpectedLossModels:
    @pytest.mark.parametrize("model", [
        "gaussian", "student_t", "heteroskedastic",
        "contaminated_normal", "lognormal", "zero_inflated",
    ])
    def test_all_six_models_effect(self, model):
        el = expected_loss(n=5, delta=0.5, reps=300, model=model)
        assert len(el) == 4
        assert all(e["expected_loss"] >= 0 for e in el)

    @pytest.mark.parametrize("model", [
        "gaussian", "student_t", "heteroskedastic",
        "contaminated_normal", "lognormal", "zero_inflated",
    ])
    def test_all_six_models_null(self, model):
        el = expected_loss(n=5, delta=0, reps=300, model=model)
        assert len(el) == 4

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            expected_loss(5, 0.5, model="xyz")


# ==============================================================
# realized_loss_magnitude
# ==============================================================

class TestRealizedLossMagnitude:
    def test_correct_positive(self):
        assert realized_loss_magnitude("positive", 0.5) == 0.0

    def test_correct_negative(self):
        assert realized_loss_magnitude("negative", -0.5) == 0.0

    def test_scales_with_effect(self):
        l1 = realized_loss_magnitude("positive", -0.5)
        l2 = realized_loss_magnitude("positive", -2.0)
        assert l2 > l1

    def test_inconclusive_scales(self):
        l1 = realized_loss_magnitude("inconclusive", 0.5)
        l2 = realized_loss_magnitude("inconclusive", 2.0)
        assert l2 > l1

    def test_null_scenario(self):
        assert realized_loss_magnitude("inconclusive", 0) == 0.0
        assert realized_loss_magnitude("positive", 0) == 1.0

    def test_magnitude_false_equals_realized_loss(self):
        for dec in ("positive", "negative", "inconclusive"):
            for d in (0, 0.5, -0.5):
                assert realized_loss_magnitude(dec, d, magnitude=False) == \
                       realized_loss(dec, d)


# ==============================================================
# estimate_pi1
# ==============================================================

class TestEstimatePi1:
    def test_storey(self):
        rng = np.random.default_rng(123)
        pvals = np.concatenate([rng.uniform(0, 0.05, 70), rng.uniform(0, 1, 30)])
        res = estimate_pi1(p_values=pvals, method="storey")
        assert 0 <= res["pi1"] <= 1
        assert res["method"] == "storey"
        assert res["pi1"] > 0.3

    def test_storey_mostly_null(self):
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, 100)
        res = estimate_pi1(p_values=pvals, method="storey")
        assert res["pi0"] > 0.7

    def test_storey_error(self):
        with pytest.raises(ValueError):
            estimate_pi1(method="storey")

    def test_mixture(self):
        rng = np.random.default_rng(99)
        d = np.concatenate([rng.normal(0, 0.3, 60), rng.normal(1.0, 0.5, 40)])
        res = estimate_pi1(effect_sizes=d, method="mixture")
        assert 0 <= res["pi1"] <= 1
        assert res["method"] == "mixture"

    def test_mixture_error(self):
        with pytest.raises(ValueError):
            estimate_pi1(method="mixture")

    def test_historical(self):
        res = estimate_pi1(replication_rate=0.40, method="historical")
        assert res["pi1"] == 0.40
        assert res["pi0"] == 0.60

    def test_historical_error(self):
        with pytest.raises(ValueError):
            estimate_pi1(replication_rate=-0.1, method="historical")
        with pytest.raises(ValueError):
            estimate_pi1(method="historical")


# ==============================================================
# Sequential decision with updated default
# ==============================================================

class TestSequentialUpdated:
    def test_default_threshold(self):
        sq = sequential_decision(delta=0.5, reps=100)
        # The function uses 0.80 by default now
        assert 0 <= sq["proceed_rate"] <= 1

    def test_null_low_discovery(self):
        sq = sequential_decision(delta=0, reps=300)
        assert sq["discovery_rate"] < 0.1


# ==============================================================
# Additional edge cases
# ==============================================================

class TestEdgeCases:
    def test_posterior_large_effect(self):
        rng = np.random.default_rng(1)
        post = bayes_posterior(rng.normal(5, 1, 10), rng.normal(0, 1, 10))
        assert post.p_positive > 0.99

    def test_posterior_no_effect_summary(self):
        post = bayes_posterior(5.0, 5.0, n1=50, n2=50, sd1=1, sd2=1, summary=True)
        assert abs(post.p_positive - 0.5) < 0.01

    def test_bf_increases_with_effect(self):
        bf1 = bayes_factor(d=0.5, se=0.5)
        bf2 = bayes_factor(d=2.0, se=0.5)
        assert bf2 > bf1

    def test_decision_strong_effect(self):
        dec = bayes_decision(5.0, 0.0, n1=20, n2=20, sd1=1.0, sd2=1.0, summary=True)
        assert dec.nhst_decision == "positive"
        assert dec.bayesopt_decision == "positive"

    def test_base_rate_analysis_methods(self):
        br = base_rate_analysis(n=5, deltas=(0, 0.5),
                                pi1_range=[0.1, 0.5, 0.9], reps=200)
        methods = {r["method"] for r in br["results"]}
        assert methods == {"NHST", "Bayes-0.95", "Bayes-Optimal", "Bayes-Factor"}

    def test_base_rate_analysis_with_data(self):
        import numpy as np
        es = np.random.default_rng(42).normal(0.5, 1.0, 30)
        br = base_rate_analysis(n=5, reps=200, effect_sizes=es, B=100)
        assert "pi1_estimate" in br
        assert "crossover" in br
        assert br["recommendation"] in ("use", "do_not_use")
        assert 0 <= br["pi1_estimate"]["pi1_hat"] <= 1

    def test_pipeline_fdr(self):
        from bayesdecision import pipeline_fdr
        r = pipeline_fdr(0.80, 1.0, 5, 0.5, 0.5)
        assert 0 <= r["fdr_pipeline"] <= 1
        assert 0 <= r["fpr_stage1"] <= 1
        assert 0 <= r["power_stage1"] <= 1

    def test_calibrate_loss_simulate(self):
        r = calibrate_loss(n=5, delta=(0, 0.5), c_fp_range=(1.0,),
                           c_i_range=(0.2,), reps=200, simulate=True)
        assert len(r) == 2
        assert r[0]["el_bayesopt"] >= 0
