#!/usr/bin/env python3
"""Post-hoc analysis of batch size effects from CMA-ES architecture search data.

Analyzes existing data from ~/elman/benchmark_results/cmaes_multicontext/e88_n{16,32}_512/
without requiring any re-training.

Experiments:
  1. Loss trajectory analysis by batch size bucket
  2. Gradient norm dynamics by batch size
  3. Learning rate sensitivity (η/B ratio)
  4. Convergence speed to loss thresholds
  5. Loss curve shape / exponential decay fitting
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will output text-only results")

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, skipping curve fitting in Experiment 5")


# --- Data structures ---

@dataclass
class TrainingStep:
    step: int
    loss: float
    lr: float
    grad_norm: float
    tok_per_sec: float


@dataclass
class EvalResult:
    eval_id: int
    batch_size: int
    lr: float
    dim: int
    depth: int
    n_heads: int
    n_state: int
    actual_params: int
    avg_loss: float          # CMA-ES fitness (average over all steps)
    final_loss: float        # FINAL_LOSS_LAST100
    phase1_loss: float
    success: bool
    steps: list[TrainingStep] = field(default_factory=list)
    total_steps: int = 0


# --- Batch size buckets ---

BATCH_BUCKETS = [
    ("bs=1", lambda b: b == 1),
    ("bs=2-4", lambda b: 2 <= b <= 4),
    ("bs=5-16", lambda b: 5 <= b <= 16),
    ("bs=17+", lambda b: b >= 17),
]


def bucket_label(batch_size: int) -> str:
    for label, pred in BATCH_BUCKETS:
        if pred(batch_size):
            return label
    return "unknown"


# --- Parsing ---

STEP_RE = re.compile(
    r'step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+grad\s+([\d.]+)\s+\|\s+tok/s\s+(\d+)'
)


def parse_training_log(log_path: Path) -> list[TrainingStep]:
    """Parse phase1_stdout.txt to extract per-step metrics."""
    steps = []
    try:
        text = log_path.read_text()
    except (FileNotFoundError, PermissionError):
        return steps

    for m in STEP_RE.finditer(text):
        steps.append(TrainingStep(
            step=int(m.group(1)),
            loss=float(m.group(2)),
            lr=float(m.group(3)),
            grad_norm=float(m.group(4)),
            tok_per_sec=int(m.group(5)),
        ))
    return steps


def load_eval(eval_dir: Path) -> EvalResult | None:
    """Load a single evaluation result."""
    done_file = eval_dir / ".done"
    if not done_file.exists():
        return None

    try:
        data = json.loads(done_file.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    if not data.get("success", False):
        return None

    params = data.get("params", {})
    steps = parse_training_log(eval_dir / "phase1_stdout.txt")

    return EvalResult(
        eval_id=data.get("eval_id", -1),
        batch_size=data.get("batch_size", params.get("batch_size", 0)),
        lr=params.get("lr", 0),
        dim=params.get("dim", 0),
        depth=params.get("depth", 0),
        n_heads=params.get("n_heads", 0),
        n_state=params.get("n_state", 0),
        actual_params=data.get("actual_params", 0),
        avg_loss=data.get("loss", 0),
        final_loss=data.get("final_loss", 0),
        phase1_loss=data.get("phase1_loss", 0),
        success=True,
        steps=steps,
        total_steps=steps[-1].step if steps else 0,
    )


def load_all_evals(base_dir: Path) -> list[EvalResult]:
    """Load all eval results from a CMA-ES search directory."""
    results = []
    # Find the timestamped search directory
    search_dirs = sorted(base_dir.glob("e88_*"))
    if not search_dirs:
        print(f"No search directories found in {base_dir}")
        return results

    search_dir = search_dirs[-1]  # Use most recent
    print(f"Loading from {search_dir}")

    for eval_dir in sorted(search_dir.glob("eval_*")):
        result = load_eval(eval_dir)
        if result is not None and result.batch_size > 0:
            results.append(result)

    print(f"  Loaded {len(results)} successful evaluations")
    return results


def group_by_bucket(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    groups = {}
    for r in results:
        label = bucket_label(r.batch_size)
        groups.setdefault(label, []).append(r)
    return groups


# --- Experiment 1: Loss Trajectory Analysis ---

def experiment1_loss_trajectory(results: list[EvalResult], output_dir: Path, label: str):
    """Plot loss vs step, loss vs tokens, loss vs wall-time by batch size bucket."""
    print(f"\n{'='*60}")
    print(f"Experiment 1: Loss Trajectory Analysis — {label}")
    print(f"{'='*60}")

    groups = group_by_bucket(results)

    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        if not bucket:
            continue
        # Pick the best 3 evals per bucket by avg_loss
        best = sorted(bucket, key=lambda r: r.avg_loss)[:3]
        avg_steps = np.mean([r.total_steps for r in bucket if r.total_steps > 0])
        avg_loss = np.mean([r.avg_loss for r in bucket])
        best_loss = min(r.avg_loss for r in bucket)
        print(f"  {bucket_name}: n={len(bucket)}, avg_steps={avg_steps:.0f}, "
              f"mean_loss={avg_loss:.4f}, best_loss={best_loss:.4f}")

    if not HAS_MATPLOTLIB:
        return

    # Plot 1a: Loss vs Step (best eval per bucket)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"bs=1": "C0", "bs=2-4": "C1", "bs=5-16": "C2", "bs=17+": "C3"}

    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        if not bucket:
            continue
        # Best eval by avg_loss that has step data
        candidates = [r for r in bucket if len(r.steps) > 5]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: r.avg_loss)
        steps = [s.step for s in best.steps]
        losses = [s.loss for s in best.steps]
        color = colors[bucket_name]

        # Smooth with running average (window=10)
        if len(losses) > 20:
            window = min(20, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
        else:
            smoothed = losses
            smooth_steps = steps

        # 1a: Loss vs step
        axes[0].plot(smooth_steps, smoothed, label=f"{bucket_name} (bs={best.batch_size})",
                     color=color, alpha=0.8)

        # 1b: Loss vs tokens seen
        tokens = [s.step * best.batch_size * 512 for s in best.steps]
        if len(tokens) > 20:
            smooth_tokens = tokens[window-1:]
        else:
            smooth_tokens = tokens
        axes[1].plot(smooth_tokens, smoothed, label=f"{bucket_name}", color=color, alpha=0.8)

        # 1c: Loss vs wall time (approximate from tok/s)
        if best.steps[0].tok_per_sec > 0:
            cumulative_tokens = np.cumsum([best.batch_size * 512 for _ in best.steps])
            avg_toks = np.mean([s.tok_per_sec for s in best.steps if s.tok_per_sec > 0])
            wall_times = cumulative_tokens / avg_toks  # seconds
            if len(wall_times) > 20:
                smooth_wall = wall_times[window-1:]
            else:
                smooth_wall = wall_times
            axes[2].plot(smooth_wall, smoothed, label=f"{bucket_name}", color=color, alpha=0.8)

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Step (best eval per bucket)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Tokens Seen")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss vs Tokens Seen")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Wall Time (seconds)")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Loss vs Wall Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Loss Trajectories by Batch Size — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"exp1_loss_trajectory_{label}.png", dpi=150)
    plt.close()
    print(f"  Saved: exp1_loss_trajectory_{label}.png")


# --- Experiment 2: Gradient Norm Dynamics ---

def experiment2_gradient_norms(results: list[EvalResult], output_dir: Path, label: str):
    """Analyze gradient norm trajectories by batch size."""
    print(f"\n{'='*60}")
    print(f"Experiment 2: Gradient Norm Dynamics — {label}")
    print(f"{'='*60}")

    groups = group_by_bucket(results)

    # Summary statistics
    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        if not bucket:
            continue
        all_grads = []
        for r in bucket:
            for s in r.steps:
                all_grads.append(s.grad_norm)
        if all_grads:
            arr = np.array(all_grads)
            print(f"  {bucket_name}: mean_grad={arr.mean():.3f}, "
                  f"median={np.median(arr):.3f}, std={arr.std():.3f}, "
                  f"max={arr.max():.3f}")

    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"bs=1": "C0", "bs=2-4": "C1", "bs=5-16": "C2", "bs=17+": "C3"}

    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        candidates = [r for r in bucket if len(r.steps) > 5]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: r.avg_loss)
        steps = [s.step for s in best.steps]
        grads = [s.grad_norm for s in best.steps]
        color = colors[bucket_name]

        # Smooth
        window = min(20, max(1, len(grads) // 5))
        if len(grads) > window:
            smoothed = np.convolve(grads, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
        else:
            smoothed = grads
            smooth_steps = steps

        axes[0].plot(smooth_steps, smoothed, label=f"{bucket_name} (bs={best.batch_size})",
                     color=color, alpha=0.8)

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_title("Gradient Norm vs Step (best eval per bucket)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution of gradient norms across all evals
    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        all_grads = []
        for r in bucket:
            # Use latter half of training (more stable)
            if r.steps:
                mid = len(r.steps) // 2
                all_grads.extend([s.grad_norm for s in r.steps[mid:]])
        if all_grads:
            axes[1].hist(all_grads, bins=50, alpha=0.4, label=bucket_name,
                         color=colors[bucket_name], density=True)

    axes[1].set_xlabel("Gradient Norm")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Gradient Norm Distribution (2nd half of training)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Gradient Norm Analysis — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"exp2_gradient_norms_{label}.png", dpi=150)
    plt.close()
    print(f"  Saved: exp2_gradient_norms_{label}.png")


# --- Experiment 3: Learning Rate Sensitivity ---

def experiment3_lr_sensitivity(results: list[EvalResult], output_dir: Path, label: str):
    """Analyze loss vs (lr, batch_size) jointly. Test if η/B is the controlling variable."""
    print(f"\n{'='*60}")
    print(f"Experiment 3: Learning Rate Sensitivity — {label}")
    print(f"{'='*60}")

    # Extract lr, batch_size, loss triples
    lrs = np.array([r.lr for r in results])
    bss = np.array([r.batch_size for r in results])
    losses = np.array([r.avg_loss for r in results])

    # Compute η/B ratio
    ratio = lrs / bss

    # Correlation analysis
    lr_corr = np.corrcoef(np.log(lrs), losses)[0, 1]
    bs_corr = np.corrcoef(np.log(bss), losses)[0, 1]
    ratio_corr = np.corrcoef(np.log(ratio), losses)[0, 1]

    print(f"  Correlation with loss:")
    print(f"    log(lr):    r = {lr_corr:.4f}")
    print(f"    log(bs):    r = {bs_corr:.4f}")
    print(f"    log(lr/bs): r = {ratio_corr:.4f}")
    print(f"  {'η/B ratio IS' if abs(ratio_corr) > max(abs(lr_corr), abs(bs_corr)) else 'η/B ratio is NOT'} "
          f"the dominant controlling variable")

    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_map = {"bs=1": "C0", "bs=2-4": "C1", "bs=5-16": "C2", "bs=17+": "C3"}

    # Color by batch size bucket
    for r in results:
        color = colors_map[bucket_label(r.batch_size)]
        axes[0].scatter(r.lr, r.avg_loss, c=color, alpha=0.4, s=10)
        axes[1].scatter(r.batch_size, r.avg_loss, c=color, alpha=0.4, s=10)
        axes[2].scatter(r.lr / r.batch_size, r.avg_loss, c=color, alpha=0.4, s=10)

    axes[0].set_xlabel("Learning Rate")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Average Loss")
    axes[0].set_title(f"Loss vs LR (r={lr_corr:.3f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Batch Size")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("Average Loss")
    axes[1].set_title(f"Loss vs Batch Size (r={bs_corr:.3f})")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("η/B Ratio")
    axes[2].set_xscale("log")
    axes[2].set_ylabel("Average Loss")
    axes[2].set_title(f"Loss vs η/B (r={ratio_corr:.3f})")
    axes[2].grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors_map.items()]
    axes[2].legend(handles=legend_elements)

    plt.suptitle(f"Learning Rate / Batch Size Sensitivity — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"exp3_lr_sensitivity_{label}.png", dpi=150)
    plt.close()
    print(f"  Saved: exp3_lr_sensitivity_{label}.png")


# --- Experiment 4: Convergence Speed ---

def experiment4_convergence_speed(results: list[EvalResult], output_dir: Path, label: str):
    """Measure steps to reach loss thresholds by batch size bucket."""
    print(f"\n{'='*60}")
    print(f"Experiment 4: Convergence Speed — {label}")
    print(f"{'='*60}")

    thresholds = [2.0, 1.5, 1.3, 1.2, 1.1]
    groups = group_by_bucket(results)

    print(f"  {'Bucket':<10} | " + " | ".join(f"L<{t}" for t in thresholds))
    print(f"  {'-'*10}-+-" + "-+-".join("-" * 5 for _ in thresholds))

    convergence_data = {}

    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        candidates = [r for r in bucket if len(r.steps) > 5]
        if not candidates:
            continue

        steps_to_threshold = {}
        for thresh in thresholds:
            step_counts = []
            for r in candidates:
                # Use running average of loss over 10-step windows
                window = min(10, len(r.steps))
                for i in range(window - 1, len(r.steps)):
                    avg = np.mean([r.steps[j].loss for j in range(i - window + 1, i + 1)])
                    if avg < thresh:
                        step_counts.append(r.steps[i].step)
                        break
            if step_counts:
                steps_to_threshold[thresh] = (np.median(step_counts), len(step_counts), len(candidates))

        convergence_data[bucket_name] = steps_to_threshold
        row = f"  {bucket_name:<10} | "
        cells = []
        for thresh in thresholds:
            if thresh in steps_to_threshold:
                med, reached, total = steps_to_threshold[thresh]
                cells.append(f"{med:5.0f}")
            else:
                cells.append("  N/A")
        row += " | ".join(cells)
        print(row)

    # Print fraction reaching each threshold
    print(f"\n  Fraction reaching threshold:")
    print(f"  {'Bucket':<10} | " + " | ".join(f"L<{t}" for t in thresholds))
    print(f"  {'-'*10}-+-" + "-+-".join("-" * 5 for _ in thresholds))
    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        data = convergence_data.get(bucket_name, {})
        row = f"  {bucket_name:<10} | "
        cells = []
        for thresh in thresholds:
            if thresh in data:
                _, reached, total = data[thresh]
                cells.append(f"{reached}/{total}")
            else:
                cells.append("  0/0")
        row += " | ".join(cells)
        print(row)

    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(thresholds))
    width = 0.2
    colors = {"bs=1": "C0", "bs=2-4": "C1", "bs=5-16": "C2", "bs=17+": "C3"}

    for i, bucket_name in enumerate(["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]):
        data = convergence_data.get(bucket_name, {})
        medians = [data[t][0] if t in data else 0 for t in thresholds]
        ax.bar(x + i * width, medians, width, label=bucket_name, color=colors[bucket_name])

    ax.set_xlabel("Loss Threshold")
    ax.set_ylabel("Median Steps to Reach")
    ax.set_title(f"Convergence Speed by Batch Size — {label}")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"L < {t}" for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"exp4_convergence_speed_{label}.png", dpi=150)
    plt.close()
    print(f"  Saved: exp4_convergence_speed_{label}.png")


# --- Experiment 5: Loss Curve Shape / Exponential Decay ---

def exp_decay(t, L_inf, A, tau):
    """L(t) = L_inf + A * exp(-t / tau)"""
    return L_inf + A * np.exp(-t / tau)


def experiment5_loss_curve_shape(results: list[EvalResult], output_dir: Path, label: str):
    """Fit exponential decay to loss curves, compare time constants."""
    print(f"\n{'='*60}")
    print(f"Experiment 5: Loss Curve Shape / Exponential Decay — {label}")
    print(f"{'='*60}")

    if not HAS_SCIPY:
        print("  Skipped: scipy required for curve fitting")
        return

    groups = group_by_bucket(results)

    fit_results = {}
    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        candidates = [r for r in bucket if len(r.steps) > 30]
        if not candidates:
            continue

        # Pick top-5 evals and fit each
        best_5 = sorted(candidates, key=lambda r: r.avg_loss)[:5]
        taus = []
        l_infs = []
        fits = []

        for r in best_5:
            steps_arr = np.array([s.step for s in r.steps], dtype=float)
            loss_arr = np.array([s.loss for s in r.steps])

            # Smooth loss for more stable fitting
            window = min(10, len(loss_arr) // 3)
            if window < 2:
                continue
            smooth_loss = np.convolve(loss_arr, np.ones(window)/window, mode='valid')
            smooth_steps = steps_arr[window-1:]

            # Initial guesses
            L_inf_guess = smooth_loss[-1]
            A_guess = smooth_loss[0] - smooth_loss[-1]
            tau_guess = smooth_steps[-1] / 3

            try:
                popt, pcov = curve_fit(
                    exp_decay, smooth_steps, smooth_loss,
                    p0=[L_inf_guess, max(A_guess, 0.1), max(tau_guess, 1)],
                    bounds=([0, 0, 1], [10, 10, 1e6]),
                    maxfev=5000,
                )
                L_inf, A, tau = popt
                taus.append(tau)
                l_infs.append(L_inf)
                fits.append((r, popt))
            except (RuntimeError, ValueError):
                continue

        if taus:
            fit_results[bucket_name] = {
                'tau_mean': np.mean(taus),
                'tau_std': np.std(taus),
                'L_inf_mean': np.mean(l_infs),
                'learning_rate_1_over_tau': 1.0 / np.mean(taus),
                'fits': fits,
            }
            print(f"  {bucket_name}: τ = {np.mean(taus):.1f} ± {np.std(taus):.1f} steps, "
                  f"L∞ = {np.mean(l_infs):.4f}, "
                  f"learning rate (1/τ) = {1.0/np.mean(taus):.6f}")

    if not HAS_MATPLOTLIB or not fit_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"bs=1": "C0", "bs=2-4": "C1", "bs=5-16": "C2", "bs=17+": "C3"}

    # Left: overlay best fit curves
    for bucket_name, data in fit_results.items():
        if not data['fits']:
            continue
        r, popt = data['fits'][0]  # Best eval
        steps_arr = np.array([s.step for s in r.steps], dtype=float)
        loss_arr = np.array([s.loss for s in r.steps])
        color = colors[bucket_name]

        # Plot raw data (faded)
        axes[0].scatter(steps_arr[::5], loss_arr[::5], c=color, alpha=0.15, s=3)
        # Plot fit
        t_fit = np.linspace(steps_arr[0], steps_arr[-1], 200)
        axes[0].plot(t_fit, exp_decay(t_fit, *popt),
                     color=color, linewidth=2,
                     label=f"{bucket_name}: τ={popt[2]:.0f}, L∞={popt[0]:.3f}")

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Exponential Decay Fits")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: bar chart of 1/τ (learning rate)
    bucket_names = list(fit_results.keys())
    rates = [fit_results[b]['learning_rate_1_over_tau'] for b in bucket_names]
    rate_colors = [colors[b] for b in bucket_names]
    axes[1].bar(bucket_names, rates, color=rate_colors)
    axes[1].set_ylabel("Learning Rate (1/τ)")
    axes[1].set_title("Effective Learning Rate by Batch Size")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Loss Curve Shape Analysis — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"exp5_loss_curve_shape_{label}.png", dpi=150)
    plt.close()
    print(f"  Saved: exp5_loss_curve_shape_{label}.png")


# --- Summary statistics ---

def print_summary(results: list[EvalResult], label: str):
    """Print summary statistics table."""
    print(f"\n{'='*60}")
    print(f"Summary Statistics — {label}")
    print(f"{'='*60}")

    groups = group_by_bucket(results)
    print(f"  {'Bucket':<10} | {'n':>4} | {'Mean Loss':>10} | {'Best Loss':>10} | "
          f"{'Median Loss':>11} | {'Avg Steps':>10} | {'Avg Params':>11}")
    print(f"  {'-'*10}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*11}")

    for bucket_name in ["bs=1", "bs=2-4", "bs=5-16", "bs=17+"]:
        bucket = groups.get(bucket_name, [])
        if not bucket:
            continue
        losses = [r.avg_loss for r in bucket]
        steps = [r.total_steps for r in bucket if r.total_steps > 0]
        params = [r.actual_params for r in bucket if r.actual_params > 0]
        print(f"  {bucket_name:<10} | {len(bucket):4d} | {np.mean(losses):10.4f} | "
              f"{min(losses):10.4f} | {np.median(losses):11.4f} | "
              f"{np.mean(steps) if steps else 0:10.0f} | "
              f"{np.mean(params)/1e6 if params else 0:8.1f}M")


# --- Main ---

def run_analysis(base_dir: Path, output_dir: Path, label: str):
    """Run all experiments on one dataset."""
    results = load_all_evals(base_dir)
    if not results:
        print(f"No results found in {base_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(results, label)
    experiment1_loss_trajectory(results, output_dir, label)
    experiment2_gradient_norms(results, output_dir, label)
    experiment3_lr_sensitivity(results, output_dir, label)
    experiment4_convergence_speed(results, output_dir, label)
    experiment5_loss_curve_shape(results, output_dir, label)


def main():
    data_root = Path.home() / "elman" / "benchmark_results" / "cmaes_multicontext"
    output_root = Path(__file__).parent / "output"

    datasets = {
        "e88_n16": data_root / "e88_n16_512",
        "e88_n32": data_root / "e88_n32_512",
    }

    for name, path in datasets.items():
        if path.exists():
            print(f"\n{'#'*60}")
            print(f"# Dataset: {name}")
            print(f"# Path: {path}")
            print(f"{'#'*60}")
            run_analysis(path, output_root / name, name)
        else:
            print(f"Skipping {name}: {path} not found")

    print(f"\nAll outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
