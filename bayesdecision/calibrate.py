"""Calibration, optimal alpha, base rate, sequential decision."""

from __future__ import annotations

import numpy as np
from scipy import stats, optimize

from bayesdecision.core import expected_loss


def calibrate_loss(
    n=5, delta=(0, 0.5), c_fp_range=(0.5, 1.0, 1.5, 2.0),
    c_i_range=(0.05, 0.1, 0.2, 0.3, 0.5),
    tau=1.0, reps=2000, pi1=0.5, simulate=False,
) -> list[dict]:
    """Sweep loss parameters and find optimal method for each.

    Parameters
    ----------
    simulate : bool
        If True, compute expected loss via Monte Carlo simulation rather than
        the conjugate-normal analytic formula. Recommended for non-Gaussian
        data. Default False.
    """
    rng = np.random.default_rng()
    results = []
    for cfp in c_fp_range:
        for ci in c_i_range:
            threshold = 1 - ci / cfp
            if threshold <= 0.5 or threshold >= 1:
                continue
            for nv in (n if hasattr(n, "__iter__") else [n]):
                for dv in delta:
                    if not simulate:
                        el = expected_loss(nv, dv, cfp, ci, tau, reps)
                        el_bopt = next(e["expected_loss"] for e in el if e["method"] == "Bayes-Optimal")
                        el_nhst = next(e["expected_loss"] for e in el if e["method"] == "NHST")
                    else:
                        yc = rng.standard_normal((reps, nv))
                        yt = rng.standard_normal((reps, nv)) + dv
                        mc, mt = yc.mean(1), yt.mean(1)
                        vc = yc.var(1, ddof=1)
                        vt = yt.var(1, ddof=1)
                        sp2 = ((nv - 1) * vc + (nv - 1) * vt) / (2 * nv - 2)
                        se2 = 2 * sp2 / nv
                        d_obs = mt - mc

                        sigma2_post = 1 / (1 / tau**2 + 1 / se2)
                        mu_post = sigma2_post * (d_obs / se2)
                        p_pos = 1 - stats.norm.cdf(0, mu_post, np.sqrt(sigma2_post))

                        bo_pos = p_pos > threshold
                        bo_neg = p_pos < (1 - threshold)

                        t_stat = d_obs / np.sqrt(se2)
                        df = 2 * nv - 2
                        pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
                        nhst_pos = (pval < 0.05) & (d_obs > 0)
                        nhst_neg = (pval < 0.05) & (d_obs < 0)

                        truly_pos = dv > 0.05
                        loss_bo = np.where(
                            bo_pos, np.where(d_obs > 0 if truly_pos else True, 0.0 if truly_pos else cfp, cfp),
                            np.where(bo_neg, np.where(d_obs < 0 if truly_pos else True, 0.0 if truly_pos else 0.0, cfp),
                                     ci if truly_pos else 0.0))
                        if truly_pos:
                            loss_bo = np.where(bo_pos, 0.0, np.where(bo_neg, cfp, ci))
                        else:
                            loss_bo = np.where(bo_pos, cfp, np.where(bo_neg, cfp, 0.0))

                        if truly_pos:
                            loss_nhst = np.where(nhst_pos, 0.0, np.where(nhst_neg, cfp, ci))
                        else:
                            loss_nhst = np.where(nhst_pos, cfp, np.where(nhst_neg, cfp, 0.0))

                        el_bopt = float(loss_bo.mean())
                        el_nhst = float(loss_nhst.mean())

                    results.append({
                        "c_fp": cfp, "c_i": ci, "threshold": threshold,
                        "n": nv, "delta": dv,
                        "el_bayesopt": el_bopt, "el_nhst": el_nhst,
                        "advantage": el_nhst - el_bopt,
                        "optimal_method": "Bayes-Optimal" if el_bopt < el_nhst else "NHST",
                    })
    return results


def optimal_alpha(
    n: int, delta1: float, c_fp=1.0, c_i=0.2, pi1=0.5,
) -> dict:
    """Decision-theoretic frequentist: find alpha minimising expected loss."""
    df = 2 * n - 2
    se = np.sqrt(2 / n)
    ncp = delta1 / se

    def el_func(alpha):
        crit = stats.t.ppf(1 - alpha / 2, df)
        power = 1 - stats.t.cdf(crit, df, loc=ncp) + stats.t.cdf(-crit, df, loc=ncp)
        return (1 - pi1) * alpha * c_fp + pi1 * (1 - power) * c_i

    result = optimize.minimize_scalar(el_func, bounds=(1e-6, 0.5), method="bounded")
    alpha_opt = result.x

    # At alpha=0.05
    crit_005 = stats.t.ppf(0.975, df)
    power_005 = 1 - stats.t.cdf(crit_005, df, loc=ncp) + stats.t.cdf(-crit_005, df, loc=ncp)
    el_005 = (1 - pi1) * 0.05 * c_fp + pi1 * (1 - power_005) * c_i

    crit_opt = stats.t.ppf(1 - alpha_opt / 2, df)
    power_opt = 1 - stats.t.cdf(crit_opt, df, loc=ncp) + stats.t.cdf(-crit_opt, df, loc=ncp)

    return {
        "alpha_opt": float(alpha_opt),
        "power_at_opt": float(power_opt),
        "expected_loss_opt": float(result.fun),
        "expected_loss_005": float(el_005),
    }


def base_rate_analysis(
    n=5, deltas=(0, 0.2, 0.5, 0.8), pi1_range=None,
    c_fp=1.0, c_i=0.2, tau=1.0, reps=3000,
    effect_sizes=None, p_values=None, conf_level=0.95, B=1000,
) -> dict:
    """Global expected loss as function of base rate P(H1).

    Parameters
    ----------
    effect_sizes : array-like, optional
        Observed standardised effect sizes for pi1 estimation via mixture model.
    p_values : array-like, optional
        Observed p-values for pi1 estimation via Storey's method.
    conf_level : float
        Confidence level for bootstrap interval. Default 0.95.
    B : int
        Number of bootstrap resamples. Default 1000.

    Returns
    -------
    dict with keys:
        'results': list of dicts with pi1, method, global_el
        'pi1_estimate': dict with pi1_hat, ci_lower, ci_upper (if data provided)
        'crossover': dict with pi1_star, p_above_star (if data provided)
        'recommendation': 'use' or 'do_not_use' (if data provided)
    """
    if pi1_range is None:
        pi1_range = np.linspace(0.01, 0.99, 99)

    el_by_delta = {}
    for d in deltas:
        el_by_delta[d] = expected_loss(n, d, c_fp, c_i, tau, reps)

    methods = ["NHST", "Bayes-0.95", "Bayes-Optimal", "Bayes-Factor"]
    results = []

    for meth in methods:
        el_null = next(e["expected_loss"] for e in el_by_delta[0] if e["method"] == meth)
        el_effects = [
            next(e["expected_loss"] for e in el_by_delta[d] if e["method"] == meth)
            for d in deltas if d > 0
        ]
        el_effect_mean = np.mean(el_effects)

        for pi1 in pi1_range:
            results.append({
                "pi1": float(pi1), "method": meth,
                "global_el": float((1 - pi1) * el_null + pi1 * el_effect_mean),
            })

    output = {"results": results}

    if effect_sizes is not None or p_values is not None:
        if effect_sizes is not None:
            data = np.asarray(effect_sizes, dtype=float)
            est_method = "mixture"
        else:
            data = np.asarray(p_values, dtype=float)
            est_method = "storey"

        if est_method == "mixture":
            pi1_est = estimate_pi1(effect_sizes=data, method="mixture")
        else:
            pi1_est = estimate_pi1(p_values=data, method="storey")
        pi1_hat = pi1_est["pi1"]

        rng = np.random.default_rng()
        boot_pi1 = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, len(data), size=len(data))
            boot_data = data[idx]
            try:
                if est_method == "mixture":
                    boot_est = estimate_pi1(effect_sizes=boot_data, method="mixture")
                else:
                    boot_est = estimate_pi1(p_values=boot_data, method="storey")
                boot_pi1[b] = boot_est["pi1"]
            except Exception:
                boot_pi1[b] = np.nan

        boot_pi1 = boot_pi1[~np.isnan(boot_pi1)]
        alpha_ci = 1 - conf_level
        ci_lower = float(np.quantile(boot_pi1, alpha_ci / 2))
        ci_upper = float(np.quantile(boot_pi1, 1 - alpha_ci / 2))

        output["pi1_estimate"] = {
            "pi1_hat": pi1_hat, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "conf_level": conf_level, "method": est_method,
        }

        pi1_arr = np.array(pi1_range)
        bo_el = np.array([r["global_el"] for r in results if r["method"] == "Bayes-Optimal"])
        nhst_el = np.array([r["global_el"] for r in results if r["method"] == "NHST"])
        diff = bo_el - nhst_el
        pi1_star = None
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                w = abs(diff[i + 1]) / (abs(diff[i]) + abs(diff[i + 1]))
                pi1_star = float(w * pi1_arr[i] + (1 - w) * pi1_arr[i + 1])
                break

        if pi1_star is not None:
            p_above = float(np.mean(boot_pi1 > pi1_star))
        else:
            p_above = 1.0 if pi1_hat > 0.9 else 0.0

        output["crossover"] = {"pi1_star": pi1_star, "p_above_star": p_above}
        output["recommendation"] = "use" if p_above > 0.95 else "do_not_use"

    return output


def estimate_pi1(
    p_values=None, effect_sizes=None, replication_rate=None,
    method="storey", lam=0.5,
) -> dict:
    """Estimate the base rate of true effects (pi1).

    Three methods:
    - 'storey': from the flat tail of the p-value distribution.
    - 'mixture': two-component Gaussian EM on effect sizes.
    - 'historical': replication rate as lower bound.

    Parameters
    ----------
    p_values : array-like, optional
        Required for 'storey'.
    effect_sizes : array-like, optional
        Required for 'mixture'.
    replication_rate : float, optional
        Required for 'historical'.
    method : str
        One of 'storey', 'mixture', 'historical'.
    lam : float
        Lambda for Storey's method. Default 0.5.

    Returns
    -------
    dict with keys: pi1, pi0, method, details.
    """
    if method == "storey":
        if p_values is None or len(p_values) < 2:
            raise ValueError("p_values required for Storey's method (at least 2 values)")
        p_values = np.asarray(p_values, dtype=float)
        p_values = p_values[~np.isnan(p_values)]
        m = len(p_values)
        pi0_hat = min(1.0, float(np.sum(p_values > lam) / (m * (1 - lam))))
        pi1_hat = 1.0 - pi0_hat
        return {
            "pi1": pi1_hat, "pi0": pi0_hat,
            "method": "storey",
            "details": {"lambda": lam, "m": m,
                        "n_above_lambda": int(np.sum(p_values > lam))},
        }

    if method == "mixture":
        if effect_sizes is None or len(effect_sizes) < 5:
            raise ValueError("effect_sizes required for mixture method (at least 5 values)")
        d = np.asarray(effect_sizes, dtype=float)
        d = d[~np.isnan(d)]
        n = len(d)
        # EM for two-component Gaussian: N(0, sigma0^2) + N(mu1, sigma1^2)
        pi0 = 0.5
        mu1 = float(np.mean(d[np.abs(d) > np.median(np.abs(d))]))
        sigma0 = float(np.std(d) * 0.5)
        sigma1 = float(np.std(d))
        iters = 0
        for iters in range(1, 101):
            lik0 = pi0 * stats.norm.pdf(d, 0, sigma0)
            lik1 = (1 - pi0) * stats.norm.pdf(d, mu1, sigma1)
            total = lik0 + lik1
            total = np.clip(total, 1e-300, None)
            gamma0 = lik0 / total
            pi0_new = float(np.mean(gamma0))
            w1 = 1 - gamma0
            sw1 = w1.sum()
            if sw1 > 1e-10:
                mu1 = float(np.sum(w1 * d) / sw1)
                sigma1 = float(np.sqrt(np.sum(w1 * (d - mu1) ** 2) / sw1))
            sigma0_new = float(np.sqrt(np.sum(gamma0 * d ** 2) / np.sum(gamma0)))
            if sigma0_new > 1e-10:
                sigma0 = sigma0_new
            if abs(pi0_new - pi0) < 1e-6:
                break
            pi0 = pi0_new
        pi0 = float(np.clip(pi0, 0, 1))
        return {
            "pi1": 1 - pi0, "pi0": pi0,
            "method": "mixture",
            "details": {"mu_null": 0, "sigma_null": sigma0,
                        "mu_nonnull": mu1, "sigma_nonnull": sigma1,
                        "iterations": iters},
        }

    if method == "historical":
        if replication_rate is None or not (0 <= replication_rate <= 1):
            raise ValueError("replication_rate must be between 0 and 1")
        return {
            "pi1": float(replication_rate), "pi0": float(1 - replication_rate),
            "method": "historical",
            "details": {"replication_rate": replication_rate,
                        "note": "Lower bound: pi1 >= replication_rate"},
        }

    raise ValueError(f"Unknown method: {method}. Use 'storey', 'mixture', or 'historical'.")


def sequential_decision(
    n_pilot=5, n_confirm=20, delta=0.5, screen_threshold=0.80,
    c_pilot=1, c_confirm=4, benefit=20, tau=1.0, reps=5000,
) -> dict:
    """Two-stage pilot-to-confirmatory pipeline."""
    rng = np.random.default_rng()

    yc1 = rng.standard_normal((reps, n_pilot))
    yt1 = rng.standard_normal((reps, n_pilot)) + delta

    mc, mt = yc1.mean(1), yt1.mean(1)
    vc, vt = yc1.var(1, ddof=1), yt1.var(1, ddof=1)
    sp2 = ((n_pilot - 1) * vc + (n_pilot - 1) * vt) / (2 * n_pilot - 2)
    se2 = 2 * sp2 / n_pilot
    d1 = mt - mc

    sigma2_post = 1 / (1 / tau**2 + 1 / se2)
    mu_post = sigma2_post * (d1 / se2)
    p_pos = 1 - stats.norm.cdf(0, mu_post, np.sqrt(sigma2_post))

    proceed = p_pos > screen_threshold

    stage2_pos = np.zeros(reps, dtype=bool)
    for j in np.where(proceed)[0]:
        yc2 = rng.standard_normal(n_confirm)
        yt2 = rng.standard_normal(n_confirm) + delta
        d2 = yt2.mean() - yc2.mean()
        sp2_2 = ((n_confirm - 1) * yc2.var(ddof=1) + (n_confirm - 1) * yt2.var(ddof=1)) / (2 * n_confirm - 2)
        se2_2 = 2 * sp2_2 / n_confirm
        t2 = d2 / np.sqrt(se2_2)
        p2 = 2 * (1 - stats.t.cdf(abs(t2), 2 * n_confirm - 2))
        stage2_pos[j] = (p2 < 0.05) and (d2 > 0)

    proceed_rate = proceed.mean()
    discovery_rate = stage2_pos.mean()
    expected_cost = c_pilot + proceed_rate * c_confirm
    true_pos_rate = discovery_rate if delta > 0.05 else 0
    expected_utility = benefit * true_pos_rate - expected_cost

    return {
        "proceed_rate": float(proceed_rate),
        "discovery_rate": float(discovery_rate),
        "expected_cost": float(expected_cost),
        "expected_utility": float(expected_utility),
    }


def pipeline_fdr(
    t_star: float, tau: float, n1: int, delta: float, pi1: float,
    alpha2: float = 0.05, n2: int = 20,
) -> dict:
    """Compute false discovery rate for a two-stage screening pipeline.

    FDR = (1-pi1)*FPR1*alpha2 / [(1-pi1)*FPR1*alpha2 + pi1*Pow1*(1-beta2)]

    Parameters
    ----------
    t_star : float
        Stage-1 posterior probability screening threshold.
    tau : float
        Prior scale for the effect size.
    n1 : int
        Stage-1 sample size per group.
    delta : float
        True effect size under the alternative.
    pi1 : float
        Base rate of true effects.
    alpha2 : float
        Stage-2 significance level. Default 0.05.
    n2 : int
        Stage-2 sample size per group. Default 20.

    Returns
    -------
    dict with keys: fdr_pipeline, fpr_stage1, power_stage1, power_stage2,
        fpr_pipeline.
    """
    se = np.sqrt(2.0 / n1)
    rho = tau / np.sqrt(tau**2 + se**2)
    crit = stats.norm.ppf(t_star) / rho

    fpr = 2 * (1 - stats.norm.cdf(crit))

    ncp = delta / se
    power_stage1 = 1 - stats.norm.cdf(crit - rho * ncp) + stats.norm.cdf(-crit - rho * ncp)

    se2 = np.sqrt(2.0 / n2)
    ncp2 = delta / se2
    z_alpha2 = stats.norm.ppf(1 - alpha2 / 2)
    power_stage2 = 1 - stats.norm.cdf(z_alpha2 - ncp2) + stats.norm.cdf(-z_alpha2 - ncp2)

    fpr_pipe = fpr * alpha2
    num = (1 - pi1) * fpr * alpha2
    den = num + pi1 * power_stage1 * power_stage2
    fdr = num / den if den > 1e-10 else np.nan

    return {
        "fdr_pipeline": float(fdr),
        "fpr_stage1": float(fpr),
        "power_stage1": float(power_stage1),
        "power_stage2": float(power_stage2),
        "fpr_pipeline": float(fpr_pipe),
    }
