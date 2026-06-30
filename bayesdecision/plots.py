"""Publication-quality visualisations."""

from __future__ import annotations

import numpy as np
from scipy import stats


def plot_decision_boundary(
    c_fp=1.0, c_i=0.2, mu_range=(-2, 2), sigma_range=(0.1, 2),
    resolution=200, ax=None, add_points=None,
):
    """Plot the decision boundary in posterior (mu, sigma) space.

    Parameters
    ----------
    c_fp, c_i : float
        Loss parameters.
    ax : matplotlib Axes, optional
    add_points : array-like of shape (N, 2), optional
        Points (mu_post, sigma_post) to overlay.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    threshold = 1 - c_i / c_fp
    mu = np.linspace(*mu_range, resolution)
    sigma = np.linspace(*sigma_range, resolution)
    MU, SIG = np.meshgrid(mu, sigma)
    P_POS = stats.norm.cdf(MU / SIG)  # P(delta>0) = Phi(mu/sigma)

    # 0 = negative, 1 = inconclusive, 2 = positive
    action = np.ones_like(P_POS, dtype=int)
    action[P_POS > threshold] = 2
    action[P_POS < (1 - threshold)] = 0

    cmap = ListedColormap(["#d62728", "#cccccc", "#2ca02c"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(MU, SIG, action, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    if add_points is not None:
        pts = np.asarray(add_points)
        ax.scatter(pts[:, 0], pts[:, 1], c="black", s=10, alpha=0.4, zorder=5)

    ax.set_xlabel(r"Posterior mean $\mu_{post}$")
    ax.set_ylabel(r"Posterior SD $\sigma_{post}$")
    ax.set_title(f"Decision boundary ($c_{{FP}}$={c_fp}, $c_I$={c_i}, threshold={threshold:.2f})")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ca02c", alpha=0.6, label=f"Positive (P>{threshold:.2f})"),
        Patch(color="#cccccc", alpha=0.6, label="Inconclusive"),
        Patch(color="#d62728", alpha=0.6, label=f"Negative (P<{1-threshold:.2f})"),
    ], loc="upper right", framealpha=0.9)

    return ax


def plot_base_rate(br_data, highlight_pi1=None, ax=None):
    """Plot global expected loss vs base rate P(H1).

    Parameters
    ----------
    br_data : list of dicts from base_rate_analysis()
    highlight_pi1 : dict, optional
        e.g. {"Drug screening": 0.1, "Pilot": 0.5}
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"NHST": "black", "Bayes-0.95": "blue",
              "Bayes-Optimal": "red", "Bayes-Factor": "purple"}
    styles = {"NHST": "-", "Bayes-0.95": "--",
              "Bayes-Optimal": "-.", "Bayes-Factor": ":"}

    methods = sorted(set(d["method"] for d in br_data))
    for meth in methods:
        sub = [d for d in br_data if d["method"] == meth]
        pi1 = [d["pi1"] for d in sub]
        gel = [d["global_el"] for d in sub]
        ax.plot(pi1, gel, color=colors.get(meth, "gray"),
                linestyle=styles.get(meth, "-"), linewidth=2, label=meth)

    if highlight_pi1:
        for name, val in highlight_pi1.items():
            ax.axvline(val, color="gray", linestyle=":", linewidth=0.8)
            ax.text(val, ax.get_ylim()[1] * 0.95, f" {name}",
                    fontsize=7, rotation=90, va="top")

    ax.set_xlabel(r"Base rate $\pi_1 = P(H_1)$")
    ax.set_ylabel("Global expected loss")
    ax.set_title("Expected loss vs base rate of true effects")
    ax.legend(framealpha=0.9)
    return ax


def plot_loss_surface(cal_data, n_show=None, delta_show=None, ax=None):
    """Heatmap of NHST advantage over Bayes-Optimal across (c_FP, c_I).

    Parameters
    ----------
    cal_data : list of dicts from calibrate_loss()
    """
    import matplotlib.pyplot as plt

    if n_show is None:
        n_show = cal_data[0]["n"]
    if delta_show is None:
        delta_show = max(d["delta"] for d in cal_data if d["delta"] > 0)

    sub = [d for d in cal_data
           if d["n"] == n_show and d["delta"] == delta_show]
    if not sub:
        raise ValueError(f"No data for n={n_show}, delta={delta_show}")

    cfps = sorted(set(d["c_fp"] for d in sub))
    cis = sorted(set(d["c_i"] for d in sub))

    mat = np.full((len(cis), len(cfps)), np.nan)
    for d in sub:
        i = cis.index(d["c_i"])
        j = cfps.index(d["c_fp"])
        mat[i, j] = d["advantage"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax,
                   origin="lower", aspect="auto",
                   extent=[cfps[0], cfps[-1], cis[0], cis[-1]])

    for d in sub:
        ax.text(d["c_fp"], d["c_i"], f"{d['advantage']:.2f}",
                ha="center", va="center", fontsize=7)

    ax.set_xlabel(r"$c_{FP}$")
    ax.set_ylabel(r"$c_I$")
    ax.set_title(f"NHST advantage over Bayes-Optimal (n={n_show}, δ={delta_show})\n"
                 f"Blue = Bayes wins | Red = NHST wins")
    plt.colorbar(im, ax=ax, label="EL(NHST) - EL(Bayes-Opt)")
    return ax
