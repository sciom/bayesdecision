"""Core functions: posterior, decisions, expected loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------

@dataclass
class BayesPosterior:
    """Conjugate normal posterior for a two-group mean difference."""
    d_hat: float
    se: float
    sp: float
    df: int
    mu_post: float
    sigma_post: float
    p_positive: float
    cri_low: float
    cri_high: float
    tau: float
    n1: int
    n2: int


def bayes_posterior(
    y1, y2=None, *, n1=None, n2=None, sd1=None, sd2=None,
    tau: float = 1.0, summary: bool = False,
) -> BayesPosterior:
    """Compute conjugate normal posterior for mean difference.

    Parameters
    ----------
    y1, y2 : array-like or float
        Treatment and control observations (arrays) or group means
        (floats, if ``summary=True``).
    n1, n2, sd1, sd2 : float, optional
        Required when ``summary=True``.
    tau : float
        Prior SD on the effect. Default 1.0.
    summary : bool
        If True, treat y1/y2 as group means.

    Returns
    -------
    BayesPosterior
    """
    if summary:
        if any(v is None for v in (n1, n2, sd1, sd2)):
            raise ValueError("n1, n2, sd1, sd2 required when summary=True")
        m1, m2 = float(y1), float(y2)
        v1, v2 = sd1**2, sd2**2
    else:
        y1, y2 = np.asarray(y1, dtype=float), np.asarray(y2, dtype=float)
        n1, n2 = len(y1), len(y2)
        m1, m2 = y1.mean(), y2.mean()
        v1, v2 = y1.var(ddof=1), y2.var(ddof=1)
        sd1, sd2 = np.sqrt(v1), np.sqrt(v2)

    d_hat = m1 - m2
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    sp = np.sqrt(sp2)
    se2 = sp2 * (1 / n1 + 1 / n2)
    se = np.sqrt(se2)

    sigma2_post = 1.0 / (1.0 / tau**2 + 1.0 / se2)
    mu_post = sigma2_post * (d_hat / se2)
    sigma_post = np.sqrt(sigma2_post)
    p_positive = 1.0 - stats.norm.cdf(0, mu_post, sigma_post)

    return BayesPosterior(
        d_hat=d_hat, se=se, sp=sp, df=df,
        mu_post=mu_post, sigma_post=sigma_post,
        p_positive=p_positive,
        cri_low=mu_post - 1.96 * sigma_post,
        cri_high=mu_post + 1.96 * sigma_post,
        tau=tau, n1=n1, n2=n2,
    )


def bayes_factor(d: float, se: float, tau: float = 1.0) -> float:
    """Analytical BF10 under normal prior N(0, tau^2).

    Returns
    -------
    float : BF10 (>1 supports H1, <1 supports H0).
    """
    se2, tau2 = se**2, tau**2
    return float(
        np.sqrt(se2 / (se2 + tau2))
        * np.exp(d**2 * tau2 / (2 * se2 * (se2 + tau2)))
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class BayesDecision:
    """Result of a Bayesian decision-theoretic analysis."""
    decision: str
    nhst_decision: str
    bayes95_decision: str
    bayesopt_decision: str
    bf_decision: str
    posterior: BayesPosterior
    expected_losses: dict
    threshold: float
    pvalue: float
    ci_low: float
    ci_high: float
    bf10: float
    c_fp: float
    c_i: float

    def __repr__(self) -> str:
        p = self.posterior
        lines = [
            "Bayesian Decision-Theoretic Analysis",
            "=" * 40,
            f"  Observed difference:  d = {p.d_hat:.4f} (SE = {p.se:.4f})",
            f"  Sample sizes:         n1 = {p.n1}, n2 = {p.n2}",
            f"  Prior:                N(0, {p.tau:.2f}²)",
            "",
            f"  NHST p-value:         {self.pvalue:.4f}  -->  {self.nhst_decision}",
            f"  P(δ > 0 | data):      {p.p_positive:.4f}",
            f"  Bayes Factor BF10:    {self.bf10:.3f}",
            "",
            "  Decisions:",
            f"    NHST (α=0.05):           {self.nhst_decision}",
            f"    Bayes-0.95:              {self.bayes95_decision}",
            f"    Bayes-Optimal (t*={self.threshold:.2f}): {self.bayesopt_decision}",
            f"    Bayes Factor (BF>3):     {self.bf_decision}",
            "",
            "  Expected losses:",
            f"    L(positive)     = {self.expected_losses['positive']:.4f}",
            f"    L(negative)     = {self.expected_losses['negative']:.4f}",
            f"    L(inconclusive) = {self.expected_losses['inconclusive']:.4f}",
            f"  --> Optimal action: {self.decision}",
        ]
        return "\n".join(lines)


def bayes_decision(
    y1, y2=None, *, n1=None, n2=None, sd1=None, sd2=None,
    tau: float = 1.0, c_fp: float = 1.0, c_i: float = 0.2,
    summary: bool = False,
) -> BayesDecision:
    """All-in-one decision analysis for a two-group comparison.

    Applies NHST, Bayes-0.95, Bayes-Optimal, and Bayes Factor rules.

    Parameters
    ----------
    y1, y2 : array-like or float
        Data or group means.
    tau : float
        Prior SD.
    c_fp : float
        Cost of false directional claim.
    c_i : float
        Cost of inconclusive outcome.

    Returns
    -------
    BayesDecision
    """
    post = bayes_posterior(y1, y2, n1=n1, n2=n2, sd1=sd1, sd2=sd2,
                           tau=tau, summary=summary)

    # NHST
    tstat = post.d_hat / post.se
    pval = 2 * (1 - stats.t.cdf(abs(tstat), post.df))
    crit = stats.t.ppf(0.975, post.df)
    ci_low = post.d_hat - crit * post.se
    ci_high = post.d_hat + crit * post.se

    if pval < 0.05 and post.d_hat > 0:
        nhst = "positive"
    elif pval < 0.05 and post.d_hat < 0:
        nhst = "negative"
    else:
        nhst = "inconclusive"

    # Bayes-0.95
    if post.p_positive > 0.95:
        b95 = "positive"
    elif post.p_positive < 0.05:
        b95 = "negative"
    else:
        b95 = "inconclusive"

    # Bayes-Optimal
    threshold = 1.0 - c_i / c_fp
    if post.p_positive > threshold:
        bopt = "positive"
    elif post.p_positive < (1 - threshold):
        bopt = "negative"
    else:
        bopt = "inconclusive"

    el_pos = c_fp * (1 - post.p_positive)
    el_neg = c_fp * post.p_positive
    el_inc = c_i

    # BF
    bf10 = bayes_factor(post.d_hat, post.se, tau)
    if bf10 > 3 and post.d_hat > 0:
        bf_dec = "positive"
    elif bf10 > 3 and post.d_hat < 0:
        bf_dec = "negative"
    elif bf10 < 1 / 3:
        bf_dec = "null_support"
    else:
        bf_dec = "inconclusive"

    return BayesDecision(
        decision=bopt, nhst_decision=nhst,
        bayes95_decision=b95, bayesopt_decision=bopt,
        bf_decision=bf_dec, posterior=post,
        expected_losses={"positive": el_pos, "negative": el_neg, "inconclusive": el_inc},
        threshold=threshold, pvalue=pval,
        ci_low=ci_low, ci_high=ci_high,
        bf10=bf10, c_fp=c_fp, c_i=c_i,
    )


# ---------------------------------------------------------------------------
# Expected loss (NumPy vectorised, single-core)
# ---------------------------------------------------------------------------

def expected_loss(
    n: int, delta: float, c_fp: float = 1.0, c_i: float = 0.2,
    tau: float = 1.0, reps: int = 5000, model: str = "gaussian",
) -> dict:
    """Monte Carlo expected loss for four methods.

    Returns dict mapping method name to dict of metrics.
    """
    threshold = 1 - c_i / c_fp
    rng = np.random.default_rng()

    if model == "gaussian":
        yc = rng.standard_normal((reps, n))
        yt = rng.standard_normal((reps, n)) + delta
    elif model == "student_t":
        sc = np.sqrt(1 / 3)
        yc = rng.standard_t(3, (reps, n)) * sc
        yt = rng.standard_t(3, (reps, n)) * sc + delta
    elif model == "heteroskedastic":
        yc = rng.standard_normal((reps, n))
        yt = rng.standard_normal((reps, n)) * 1.5 + delta
    elif model == "contaminated_normal":
        u_c = rng.random((reps, n))
        yc = np.where(u_c < 0.8, rng.standard_normal((reps, n)),
                       rng.standard_normal((reps, n)) * 3)
        u_t = rng.random((reps, n))
        yt = np.where(u_t < 0.8, rng.normal(delta, 1, (reps, n)),
                       rng.normal(delta, 3, (reps, n)))
    elif model == "lognormal":
        shift = np.exp(0 + 0.5 ** 2 / 2)
        yc = rng.lognormal(0, 0.5, (reps, n)) - shift
        yt = rng.lognormal(0, 0.5, (reps, n)) - shift + delta
    elif model == "zero_inflated":
        u_c = rng.random((reps, n))
        yc = np.where(u_c < 0.3, 0.0, rng.standard_normal((reps, n)))
        u_t = rng.random((reps, n))
        yt = np.where(u_t < 0.3, delta, rng.normal(delta, 1, (reps, n)))
    else:
        raise ValueError(f"Unknown model: {model}")

    mc, mt = yc.mean(1), yt.mean(1)
    vc, vt = yc.var(1, ddof=1), yt.var(1, ddof=1)
    sp2 = ((n - 1) * vc + (n - 1) * vt) / (2 * n - 2)
    se2 = 2 * sp2 / n
    d = mt - mc

    # NHST
    tstat = d / np.sqrt(se2)
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), 2 * n - 2))
    nhst = np.where((pval < 0.05) & (d > 0), "positive",
           np.where((pval < 0.05) & (d < 0), "negative", "inconclusive"))

    # Bayes
    sigma2_post = 1 / (1 / tau**2 + 1 / se2)
    mu_post = sigma2_post * (d / se2)
    sigma_post = np.sqrt(sigma2_post)
    p_pos = 1 - stats.norm.cdf(0, mu_post, sigma_post)

    b95 = np.where(p_pos > 0.95, "positive",
          np.where(p_pos < 0.05, "negative", "inconclusive"))
    bopt = np.where(p_pos > threshold, "positive",
           np.where(p_pos < (1 - threshold), "negative", "inconclusive"))

    bf10 = np.sqrt(se2 / (se2 + tau**2)) * np.exp(d**2 * tau**2 / (2 * se2 * (se2 + tau**2)))
    bf_dec = np.where((bf10 > 3) & (d > 0), "positive",
             np.where((bf10 > 3) & (d < 0), "negative",
             np.where(bf10 < 1/3, "null_support", "inconclusive")))

    def _rl(dec):
        if delta > 0.05:
            return np.where(dec == "positive", 0,
                   np.where(dec == "negative", c_fp, c_i))
        else:
            return np.where((dec == "inconclusive") | (dec == "null_support"), 0, c_fp)

    def _metrics(dec, name):
        loss = _rl(dec)
        is_eff = delta > 0.05
        return {
            "method": name,
            "expected_loss": float(loss.mean()),
            "accuracy": float((dec == "positive").mean() if is_eff else ((dec == "inconclusive") | (dec == "null_support")).mean()),
            "fpr": float(np.nan if is_eff else ((dec != "inconclusive") & (dec != "null_support")).mean()),
            "fnr": float(np.nan if not is_eff else (dec != "positive").mean()),
            "inconclusive_rate": float(((dec == "inconclusive") | (dec == "null_support")).mean()),
        }

    return [
        _metrics(nhst, "NHST"),
        _metrics(b95, "Bayes-0.95"),
        _metrics(bopt, "Bayes-Optimal"),
        _metrics(bf_dec, "Bayes-Factor"),
    ]


def realized_loss(decision: str, true_delta: float,
                  c_fp: float = 1.0, c_i: float = 0.2) -> float:
    """Compute realized loss for a single decision."""
    if true_delta > 0.05:
        if decision == "positive":
            return 0.0
        elif decision == "negative":
            return c_fp
        else:
            return c_i
    else:
        if decision in ("inconclusive", "null_support"):
            return 0.0
        else:
            return c_fp


def realized_loss_magnitude(
    decision: str, true_delta: float,
    c_fp: float = 1.0, c_i: float = 0.2, magnitude: bool = True,
) -> float:
    """Magnitude-sensitive realized loss.

    When ``magnitude=True``, costs scale with ``max(1, |true_delta|)``.
    When ``magnitude=False``, equivalent to :func:`realized_loss`.
    """
    if not magnitude:
        return realized_loss(decision, true_delta, c_fp, c_i)
    scale = max(1.0, abs(true_delta))
    if true_delta > 0.05:
        if decision == "positive":
            return 0.0
        elif decision == "negative":
            return c_fp * scale
        else:
            return c_i * scale
    elif true_delta < -0.05:
        if decision == "negative":
            return 0.0
        elif decision == "positive":
            return c_fp * scale
        else:
            return c_i * scale
    else:
        if decision in ("inconclusive", "null_support"):
            return 0.0
        return c_fp
