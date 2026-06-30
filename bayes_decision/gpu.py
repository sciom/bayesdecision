"""GPU-accelerated Monte Carlo via PyTorch."""

from __future__ import annotations

import math

import numpy as np


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required for GPU acceleration.\n"
            "Install: pip install torch\n"
            "See: https://pytorch.org/get-started/locally/"
        )


def expected_loss_gpu(
    n: int, delta: float, c_fp: float = 1.0, c_i: float = 0.2,
    tau: float = 1.0, reps: int = 100_000, device: str | None = None,
) -> list[dict]:
    """GPU-accelerated expected loss for four methods.

    Parameters
    ----------
    n : int
        Sample size per group.
    delta : float
        True effect size.
    device : str or None
        "cuda", "cpu", or None (auto-detect).
    reps : int
        MC replicates. GPU handles 1M+ efficiently.

    Returns
    -------
    list of dicts, one per method.
    """
    torch = _check_torch()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[bayesDecisionR] Using GPU: {gpu_name}")
        else:
            device = "cpu"
            _warn_no_cuda(torch)

    dev = torch.device(device)
    threshold = 1 - c_i / c_fp
    tau2 = tau ** 2
    sqrt2 = math.sqrt(2.0)

    # Generate on device
    yc = torch.randn(reps, n, device=dev)
    yt = torch.randn(reps, n, device=dev) + delta

    mc = yc.mean(dim=1)
    mt = yt.mean(dim=1)
    d = mt - mc

    vc = ((yc - mc.unsqueeze(1)) ** 2).sum(dim=1) / (n - 1)
    vt = ((yt - mt.unsqueeze(1)) ** 2).sum(dim=1) / (n - 1)
    sp2 = ((n - 1) * vc + (n - 1) * vt) / (2 * n - 2)
    se2 = 2 * sp2 / n

    # NHST (normal approximation for p-value)
    z = torch.abs(d) / torch.sqrt(se2)
    pval = 2 * (1 - 0.5 * (1 + torch.erf(z / sqrt2)))
    nhst_pos = (pval < 0.05) & (d > 0)
    nhst_neg = (pval < 0.05) & (d < 0)

    # Bayes posterior
    sigma2_post = 1.0 / (1.0 / tau2 + 1.0 / se2)
    mu_post = sigma2_post * (d / se2)
    sigma_post = torch.sqrt(sigma2_post)
    p_pos = 0.5 * (1 + torch.erf(mu_post / (sigma_post * sqrt2)))

    b95_pos = p_pos > 0.95
    b95_neg = p_pos < 0.05
    bopt_pos = p_pos > threshold
    bopt_neg = p_pos < (1 - threshold)

    # BF10
    bf10 = torch.sqrt(se2 / (se2 + tau2)) * torch.exp(
        d ** 2 * tau2 / (2 * se2 * (se2 + tau2))
    )
    bf_pos = (bf10 > 3) & (d > 0)
    bf_neg = (bf10 > 3) & (d < 0)
    bf_null = bf10 < (1 / 3)

    is_effect = delta > 0.05

    def _metrics(pos, neg, name, null_sup=None):
        pos_f = pos.float()
        neg_f = neg.float()
        inc_f = (~pos & ~neg).float()
        if null_sup is not None:
            inc_f = inc_f + null_sup.float()
            inc_f = inc_f.clamp(max=1.0)

        if is_effect:
            loss = neg_f * c_fp + (~pos & ~neg).float() * c_i
            acc = pos_f.mean().item()
            fpr = float("nan")
            fnr = 1 - acc
        else:
            loss = (pos_f + neg_f) * c_fp
            acc_val = 1 - (pos_f + neg_f).mean().item()
            acc = acc_val
            fpr = 1 - acc
            fnr = float("nan")

        return {
            "method": name,
            "expected_loss": loss.mean().item(),
            "accuracy": acc,
            "fpr": fpr,
            "fnr": fnr,
            "inconclusive_rate": inc_f.mean().item(),
        }

    return [
        _metrics(nhst_pos, nhst_neg, "NHST"),
        _metrics(b95_pos, b95_neg, "Bayes-0.95"),
        _metrics(bopt_pos, bopt_neg, "Bayes-Optimal"),
        _metrics(bf_pos, bf_neg, "Bayes-Factor", bf_null),
    ]


def expected_loss_grid_gpu(
    n=(3, 5, 10), delta=(0, 0.2, 0.5, 0.8),
    c_fp=1.0, c_i=0.2, tau=1.0, reps=100_000, device=None,
) -> list[dict]:
    """GPU grid across (n, delta) scenarios."""
    results = []
    for nv in n:
        for dv in delta:
            el = expected_loss_gpu(nv, dv, c_fp, c_i, tau, reps, device)
            for e in el:
                e["n"] = nv
                e["delta"] = dv
            results.extend(el)
    return results


def _warn_no_cuda(torch):
    """Print diagnostic if CUDA not available."""
    import shutil
    msg_parts = ["[bayesDecisionR] No CUDA GPU detected, using CPU tensors."]

    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")

    if nvidia_smi is None:
        msg_parts.append("  No NVIDIA driver found (nvidia-smi not in PATH).")
        msg_parts.append("  For GPU acceleration, install NVIDIA drivers + CUDA toolkit.")
    else:
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                gpu_info = out.stdout.strip()
                msg_parts.append(f"  GPU detected: {gpu_info}")
        except Exception:
            pass

    if nvcc:
        import subprocess
        try:
            out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            ver_line = [l for l in out.stdout.splitlines() if "release" in l]
            if ver_line:
                cuda_ver = ver_line[0].split("release")[-1].split(",")[0].strip()
                msg_parts.append(f"  System CUDA: {cuda_ver}")
                # Check compatibility
                torch_cuda = torch.version.cuda or "none"
                msg_parts.append(f"  PyTorch CUDA: {torch_cuda}")
                if torch_cuda == "none" or cuda_ver.split(".")[0] != torch_cuda.split(".")[0]:
                    msg_parts.append(f"  *** VERSION MISMATCH ***")
                    msg_parts.append(f"  Install matching PyTorch:")
                    msg_parts.append(f"    pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_ver.replace('.','')[:3]}")
        except Exception:
            pass
    else:
        if nvidia_smi:
            msg_parts.append("  CUDA toolkit (nvcc) not found.")
            msg_parts.append("  Install: https://developer.nvidia.com/cuda-downloads")

    msg_parts.append("  Tip: CPU parallel is also fast — use expected_loss_grid() with joblib.")
    print("\n".join(msg_parts))
