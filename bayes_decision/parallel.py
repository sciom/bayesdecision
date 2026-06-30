"""CPU parallel execution and system diagnostics."""

from __future__ import annotations

import os
import shutil
import subprocess
from multiprocessing import cpu_count

from bayes_decision.core import expected_loss


def check_system(verbose: bool = True) -> dict:
    """Diagnose system capabilities: CPU, parallel, GPU.

    Prints actionable advice on how to enable acceleration.

    Returns
    -------
    dict with keys: cpu_cores, has_joblib, has_torch, torch_cuda,
    gpu_name, cuda_system, cuda_torch, advice.
    """
    info = {
        "cpu_cores": cpu_count() or 1,
        "has_joblib": False,
        "has_torch": False,
        "torch_cuda": False,
        "gpu_name": "",
        "cuda_system": "",
        "cuda_torch": "",
        "advice": [],
    }

    # joblib
    try:
        import joblib  # noqa: F401
        info["has_joblib"] = True
    except ImportError:
        info["advice"].append("Install joblib for CPU parallel: pip install joblib")

    # torch
    try:
        import torch
        info["has_torch"] = True
        info["torch_cuda"] = torch.cuda.is_available()
        info["cuda_torch"] = torch.version.cuda or "none"
        if info["torch_cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            # Check why CUDA isn't working
            nvidia_smi = shutil.which("nvidia-smi")
            nvcc = shutil.which("nvcc")

            if nvidia_smi:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    if out.returncode == 0:
                        info["gpu_name"] = out.stdout.strip().split(",")[0].strip()
                except Exception:
                    pass

            if nvcc:
                try:
                    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
                    for line in out.stdout.splitlines():
                        if "release" in line:
                            info["cuda_system"] = line.split("release")[-1].split(",")[0].strip()
                except Exception:
                    pass

            if info["cuda_system"] and info["cuda_torch"] != "none":
                sys_major = info["cuda_system"].split(".")[0]
                torch_major = info["cuda_torch"].split(".")[0]
                if sys_major != torch_major:
                    info["advice"].append(
                        f"CUDA version mismatch: system={info['cuda_system']}, "
                        f"PyTorch={info['cuda_torch']}. "
                        f"Upgrade CUDA or reinstall PyTorch for CUDA {info['cuda_system']}."
                    )
            elif not info["cuda_system"] and info["gpu_name"]:
                info["advice"].append(
                    "GPU detected but CUDA toolkit not found. "
                    "Install: https://developer.nvidia.com/cuda-downloads"
                )
            elif info["cuda_torch"] == "none":
                info["advice"].append(
                    "PyTorch installed without CUDA. Reinstall with CUDA: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )
    except ImportError:
        info["advice"].append("Install torch for GPU: pip install torch")

    if verbose:
        print("=== bayes_decision System Diagnostics ===\n")
        print(f"CPU cores:       {info['cpu_cores']}")
        print(f"Parallel (joblib): {'READY' if info['has_joblib'] else 'NOT AVAILABLE'}")
        if info["has_joblib"]:
            n_workers = max(1, info["cpu_cores"] - 1)
            print(f"  -> Use expected_loss_grid(n_jobs={n_workers})")
        print()

        print(f"PyTorch:         {'installed' if info['has_torch'] else 'NOT INSTALLED'}")
        if info["has_torch"]:
            print(f"  CUDA available: {'YES' if info['torch_cuda'] else 'NO'}")
            if info["torch_cuda"]:
                print(f"  GPU: {info['gpu_name']}")
                print(f"  CUDA version: {info['cuda_torch']}")
                print(f"  -> Use expected_loss_gpu(device='cuda')")
            else:
                if info["gpu_name"]:
                    print(f"  GPU hardware: {info['gpu_name']}")
                if info["cuda_system"]:
                    print(f"  System CUDA: {info['cuda_system']}")
                if info["cuda_torch"] != "none":
                    print(f"  PyTorch CUDA: {info['cuda_torch']}")

        if info["advice"]:
            print(f"\n  ADVICE:")
            for a in info["advice"]:
                print(f"    - {a}")

        print(f"\n=== Recommended ===")
        if info["torch_cuda"]:
            print(f"  from bayes_decision import expected_loss_gpu")
            print(f"  results = expected_loss_gpu(n=5, delta=0.5, reps=1_000_000)")
        elif info["has_joblib"]:
            n_w = max(1, info["cpu_cores"] - 1)
            print(f"  from bayes_decision import expected_loss_grid")
            print(f"  results = expected_loss_grid([3,5,10], [0,0.2,0.5,0.8], n_jobs={n_w})")
        else:
            print(f"  from bayes_decision import expected_loss")
            print(f"  results = expected_loss(n=5, delta=0.5)")

    return info


def expected_loss_grid(
    n=(3, 5, 10), delta=(0, 0.2, 0.5, 0.8),
    c_fp=1.0, c_i=0.2, tau=1.0, reps=5000,
    model="gaussian", n_jobs=-1,
) -> list[dict]:
    """Parallel expected loss across a grid of (n, delta).

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 = all cores minus 1.
        1 = sequential. Requires joblib.
    """
    import itertools
    scenarios = list(itertools.product(
        n if hasattr(n, "__iter__") else [n],
        delta if hasattr(delta, "__iter__") else [delta],
    ))

    def _run_one(args):
        nv, dv = args
        el = expected_loss(nv, dv, c_fp, c_i, tau, reps, model)
        for e in el:
            e["n"] = nv
            e["delta"] = dv
        return el

    try:
        from joblib import Parallel, delayed
        if n_jobs == -1:
            n_jobs = max(1, (cpu_count() or 2) - 1)
        chunks = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(s) for s in scenarios
        )
    except ImportError:
        chunks = [_run_one(s) for s in scenarios]

    results = []
    for chunk in chunks:
        results.extend(chunk)
    return results


def calibrate_loss_parallel(
    n=5, delta=(0, 0.5), c_fp_range=(0.5, 1.0, 1.5, 2.0),
    c_i_range=(0.05, 0.1, 0.2, 0.3, 0.5),
    tau=1.0, reps=2000, n_jobs=-1,
) -> list[dict]:
    """Parallel calibration across (c_fp, c_i) grid."""
    import itertools

    grid = [
        (cfp, ci) for cfp, ci in itertools.product(c_fp_range, c_i_range)
        if 0.5 < (1 - ci / cfp) < 1
    ]

    n_list = n if hasattr(n, "__iter__") else [n]
    delta_list = delta if hasattr(delta, "__iter__") else [delta]

    def _run_one(args):
        cfp, ci = args
        results = []
        for nv in n_list:
            for dv in delta_list:
                el = expected_loss(nv, dv, cfp, ci, tau, reps)
                el_bopt = next(e["expected_loss"] for e in el if e["method"] == "Bayes-Optimal")
                el_nhst = next(e["expected_loss"] for e in el if e["method"] == "NHST")
                results.append({
                    "c_fp": cfp, "c_i": ci, "threshold": 1 - ci / cfp,
                    "n": nv, "delta": dv,
                    "el_bayesopt": el_bopt, "el_nhst": el_nhst,
                    "advantage": el_nhst - el_bopt,
                    "optimal_method": "Bayes-Optimal" if el_bopt < el_nhst else "NHST",
                })
        return results

    try:
        from joblib import Parallel, delayed
        if n_jobs == -1:
            n_jobs = max(1, (cpu_count() or 2) - 1)
        chunks = Parallel(n_jobs=n_jobs)(delayed(_run_one)(g) for g in grid)
    except ImportError:
        chunks = [_run_one(g) for g in grid]

    results = []
    for chunk in chunks:
        results.extend(chunk)
    return results
