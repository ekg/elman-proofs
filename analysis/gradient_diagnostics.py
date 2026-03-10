#!/usr/bin/env python3
"""Targeted gradient diagnostic experiments for batch size learning dynamics.

These experiments require re-training specific configs with additional logging.
Run when GPU is available, picking 3-5 configs from the CMA-ES Pareto front.

Experiments:
  6. Per-sample gradient cosine similarity
  7. Hidden state continuity ablation
  8. Gradient noise scale (B_noise) measurement
  9. Gradient Agreement Filtering (GAF)

Usage:
  # Run all diagnostics on a specific config
  python gradient_diagnostics.py --experiment all --config best_bs1

  # Run single experiment
  python gradient_diagnostics.py --experiment 6 --config best_bs1

  # List available configs from CMA-ES results
  python gradient_diagnostics.py --list-configs
"""

import argparse
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Add elman project to path
ELMAN_DIR = Path.home() / "elman"
sys.path.insert(0, str(ELMAN_DIR))


@dataclass
class Config:
    """A training configuration from CMA-ES results."""
    dim: int
    n_heads: int
    n_state: int
    depth: int
    lr: float
    batch_size: int
    actual_params: int
    avg_loss: float
    eval_dir: str


def load_pareto_configs(n_state: int = 16, n_configs: int = 5) -> list[Config]:
    """Load Pareto-front configs from CMA-ES results, spanning batch sizes."""
    data_root = Path.home() / "elman" / "benchmark_results" / "cmaes_multicontext"
    search_dir = data_root / f"e88_n{n_state}_512"

    if not search_dir.exists():
        print(f"No data at {search_dir}")
        return []

    # Find most recent search
    subdirs = sorted(search_dir.glob("e88_*"))
    if not subdirs:
        return []
    search = subdirs[-1]

    # Load all successful evals
    configs = []
    for eval_dir in search.glob("eval_*"):
        done = eval_dir / ".done"
        if not done.exists():
            continue
        try:
            data = json.loads(done.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if not data.get("success"):
            continue
        params = data["params"]
        configs.append(Config(
            dim=params["dim"],
            n_heads=params["n_heads"],
            n_state=params["n_state"],
            depth=params["depth"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            actual_params=data.get("actual_params", 0),
            avg_loss=data.get("loss", 999),
            eval_dir=str(eval_dir),
        ))

    # Select Pareto front: best loss at each batch size bucket
    selected = []
    for bs_range, label in [(1, 1), (2, 4), (5, 16), (17, 128)]:
        bucket = [c for c in configs if bs_range <= c.batch_size <= (label if isinstance(label, int) else bs_range)]
        if bucket:
            best = min(bucket, key=lambda c: c.avg_loss)
            selected.append(best)

    # Also include overall best
    overall_best = min(configs, key=lambda c: c.avg_loss)
    if overall_best not in selected:
        selected.insert(0, overall_best)

    return selected[:n_configs]


def build_model(config: Config):
    """Build an E88 model from config parameters."""
    # Import from elman project
    from elman.models.e88_fused import E88FusedLM

    model = E88FusedLM(
        vocab_size=256,
        dim=config.dim,
        depth=config.depth,
        n_heads=config.n_heads,
        n_state=config.n_state,
        expansion=1.0,
        use_gate=True,
        gate_activation='silu',
    )
    return model


def get_data_loader(batch_size: int, chunk_size: int = 512, n_batches: int = 100):
    """Create a data loader from the training data."""
    data_path = Path("/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    # Memory-map the file
    data = np.memmap(str(data_path), dtype=np.uint8, mode='r')
    total_bytes = len(data)

    def sequential_loader():
        """Yield sequential chunks (mimicking bs=1 training)."""
        offset = 0
        for _ in range(n_batches):
            if offset + batch_size * (chunk_size + 1) > total_bytes:
                offset = 0
            batch_input = []
            batch_target = []
            for b in range(batch_size):
                start = offset + b * chunk_size
                chunk = data[start:start + chunk_size + 1].copy()
                batch_input.append(torch.from_numpy(chunk[:-1].astype(np.int64)))
                batch_target.append(torch.from_numpy(chunk[1:].astype(np.int64)))
            offset += batch_size * chunk_size
            yield torch.stack(batch_input), torch.stack(batch_target)

    return sequential_loader


# --- Experiment 6: Per-Sample Gradient Cosine Similarity ---

def experiment6_gradient_cosine(config: Config, n_steps: int = 50, device: str = "cuda"):
    """Measure pairwise cosine similarity of per-sample gradients.

    For a batch of B samples, compute each sample's gradient independently,
    then measure all (B choose 2) pairwise cosine similarities.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 6: Per-Sample Gradient Cosine Similarity")
    print(f"  Config: dim={config.dim}, depth={config.depth}, bs={config.batch_size}")
    print(f"{'='*60}")

    if config.batch_size < 2:
        print("  Need batch_size >= 2 for pairwise comparison. Using bs=8 override.")
        test_bs = 8
    else:
        test_bs = config.batch_size

    model = build_model(config).to(device)
    model.train()

    loader = get_data_loader(batch_size=test_bs, n_batches=n_steps)
    criterion = nn.CrossEntropyLoss()

    all_cosines = []
    all_conflicts = []  # fraction with cos < 0

    for step, (inputs, targets) in enumerate(loader()):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute per-sample gradients
        per_sample_grads = []
        for b in range(test_bs):
            model.zero_grad()
            output = model(inputs[b:b+1])
            loss = criterion(output.view(-1, 256), targets[b].view(-1))
            loss.backward()

            # Flatten all parameter gradients into one vector
            grad_vec = torch.cat([
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ])
            per_sample_grads.append(grad_vec)

        # Compute pairwise cosine similarities
        grads_matrix = torch.stack(per_sample_grads)  # [B, D]
        norms = grads_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = grads_matrix / norms
        cosine_matrix = normalized @ normalized.T  # [B, B]

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(test_bs, test_bs, device=device), diagonal=1).bool()
        pairwise = cosine_matrix[mask]

        all_cosines.extend(pairwise.cpu().tolist())
        conflict_frac = (pairwise < 0).float().mean().item()
        all_conflicts.append(conflict_frac)

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: mean_cos={pairwise.mean():.4f}, "
                  f"conflict_frac={conflict_frac:.3f}")

    cosines = np.array(all_cosines)
    print(f"\n  Overall Results:")
    print(f"    Mean cosine similarity: {cosines.mean():.4f}")
    print(f"    Median cosine similarity: {np.median(cosines):.4f}")
    print(f"    Std: {cosines.std():.4f}")
    print(f"    Fraction negative (conflicting): {(cosines < 0).mean():.4f}")
    print(f"    Fraction near-orthogonal (|cos| < 0.1): {(np.abs(cosines) < 0.1).mean():.4f}")

    return {
        'mean_cosine': cosines.mean(),
        'median_cosine': np.median(cosines),
        'std_cosine': cosines.std(),
        'conflict_fraction': (cosines < 0).mean(),
        'orthogonal_fraction': (np.abs(cosines) < 0.1).mean(),
        'cosines': cosines,
    }


# --- Experiment 7: Hidden State Continuity Ablation ---

def experiment7_hidden_state_ablation(config: Config, n_steps: int = 2000,
                                       device: str = "cuda"):
    """Compare sequential (continuous hidden state) vs shuffled (reset hidden state) training.

    Both use bs=1. Sequential reads consecutive 512-byte windows.
    Shuffled reads random positions, resetting hidden state each chunk.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 7: Hidden State Continuity Ablation")
    print(f"  Config: dim={config.dim}, depth={config.depth}")
    print(f"{'='*60}")

    data_path = Path("/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    data = np.memmap(str(data_path), dtype=np.uint8, mode='r')
    total_bytes = len(data)
    chunk_size = 512

    criterion = nn.CrossEntropyLoss()

    results = {}

    for mode in ["sequential", "shuffled"]:
        print(f"\n  Training mode: {mode}")
        model = build_model(config).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        losses = []
        offset = 0

        for step in range(n_steps):
            if mode == "sequential":
                start = offset
                offset += chunk_size
                if offset + chunk_size + 1 > total_bytes:
                    offset = 0
            else:
                start = np.random.randint(0, total_bytes - chunk_size - 1)

            chunk = data[start:start + chunk_size + 1].copy()
            x = torch.from_numpy(chunk[:-1].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(chunk[1:].astype(np.int64)).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, 256), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if (step + 1) % 200 == 0:
                recent = np.mean(losses[-100:])
                print(f"    Step {step+1}: loss={recent:.4f}")

        results[mode] = np.array(losses)

    # Compare
    seq_final = np.mean(results["sequential"][-200:])
    shuf_final = np.mean(results["shuffled"][-200:])
    print(f"\n  Final loss (last 200 steps):")
    print(f"    Sequential: {seq_final:.4f}")
    print(f"    Shuffled:   {shuf_final:.4f}")
    print(f"    Difference: {shuf_final - seq_final:.4f} "
          f"({'sequential wins' if seq_final < shuf_final else 'shuffled wins'})")

    return results


# --- Experiment 8: Gradient Noise Scale Measurement ---

def experiment8_gradient_noise_scale(config: Config, n_measurements: int = 20,
                                      measurement_bs: int = 16, device: str = "cuda"):
    """Measure B_noise = tr(Σ) / ‖G‖² at various training steps.

    Uses a moderate batch size to estimate the gradient covariance.
    B_noise determines the critical batch size.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 8: Gradient Noise Scale Measurement")
    print(f"  Config: dim={config.dim}, depth={config.depth}")
    print(f"  Measurement batch size: {measurement_bs}")
    print(f"{'='*60}")

    model = build_model(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    loader = get_data_loader(batch_size=1, n_batches=n_measurements * (measurement_bs + 50))
    data_iter = iter(loader())

    b_noise_values = []
    steps_measured = []

    for measurement in range(n_measurements):
        # Train for 50 steps between measurements
        for _ in range(50):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader())
                inputs, targets = next(data_iter)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, 256), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        current_step = (measurement + 1) * 50

        # Compute per-sample gradients for B_noise estimation
        per_sample_grads = []
        for _ in range(measurement_bs):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader())
                inputs, targets = next(data_iter)
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, 256), targets.view(-1))
            loss.backward()

            grad_vec = torch.cat([
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ])
            per_sample_grads.append(grad_vec)

        grads = torch.stack(per_sample_grads)  # [B, D]

        # Mean gradient G
        G = grads.mean(dim=0)  # [D]
        G_norm_sq = G.norm().pow(2).item()

        # Gradient covariance trace: tr(Σ) = (1/B) Σ ‖g_i - G‖²
        diffs = grads - G.unsqueeze(0)  # [B, D]
        tr_sigma = diffs.pow(2).sum().item() / measurement_bs

        # B_noise = tr(Σ) / ‖G‖²
        if G_norm_sq > 1e-10:
            b_noise = tr_sigma / G_norm_sq
        else:
            b_noise = float('inf')

        b_noise_values.append(b_noise)
        steps_measured.append(current_step)

        print(f"  Step {current_step}: B_noise = {b_noise:.2f}, "
              f"‖G‖² = {G_norm_sq:.4f}, tr(Σ) = {tr_sigma:.4f}")

    print(f"\n  Summary:")
    arr = np.array(b_noise_values)
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0:
        print(f"    Mean B_noise: {finite.mean():.2f}")
        print(f"    Median B_noise: {np.median(finite):.2f}")
        print(f"    Min B_noise: {finite.min():.2f}")
        print(f"    Max B_noise: {finite.max():.2f}")
        print(f"    B_noise < 1 (bs=1 optimal): {(finite < 1).sum()}/{len(finite)} measurements")
        print(f"    B_noise < 10: {(finite < 10).sum()}/{len(finite)} measurements")

    return {
        'b_noise': b_noise_values,
        'steps': steps_measured,
    }


# --- Experiment 9: Gradient Agreement Filtering (GAF) ---

def experiment9_gaf(config: Config, n_steps: int = 2000, test_bs: int = 8,
                     cos_threshold: float = 0.0, device: str = "cuda"):
    """Train at bs=8 but filter out micro-gradients with low agreement.

    GAF: Only apply micro-gradients that have positive cosine similarity
    with the mean gradient. Compare to bs=1 baseline.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 9: Gradient Agreement Filtering (GAF)")
    print(f"  Config: dim={config.dim}, depth={config.depth}")
    print(f"  Test batch size: {test_bs}, cos threshold: {cos_threshold}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    results = {}

    for mode in ["bs1_baseline", f"bs{test_bs}_vanilla", f"bs{test_bs}_gaf"]:
        print(f"\n  Mode: {mode}")
        model = build_model(config).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        if mode == "bs1_baseline":
            effective_bs = 1
        else:
            effective_bs = test_bs

        loader = get_data_loader(batch_size=effective_bs, n_batches=n_steps)
        losses = []

        for step, (inputs, targets) in enumerate(loader()):
            if step >= n_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)

            if "gaf" in mode:
                # Compute per-sample gradients
                per_sample_grads = []
                per_sample_losses = []
                for b in range(effective_bs):
                    model.zero_grad()
                    output = model(inputs[b:b+1])
                    loss = criterion(output.view(-1, 256), targets[b].view(-1))
                    loss.backward()
                    per_sample_losses.append(loss.item())

                    grad_vec = torch.cat([
                        p.grad.flatten() for p in model.parameters() if p.grad is not None
                    ])
                    per_sample_grads.append(grad_vec)

                grads = torch.stack(per_sample_grads)
                mean_grad = grads.mean(dim=0)

                # Filter: keep only gradients with cos > threshold relative to mean
                mean_norm = mean_grad.norm().clamp(min=1e-8)
                cosines = (grads @ mean_grad) / (grads.norm(dim=1).clamp(min=1e-8) * mean_norm)
                mask = cosines > cos_threshold
                n_kept = mask.sum().item()

                if n_kept > 0:
                    filtered_grad = grads[mask].mean(dim=0)
                else:
                    filtered_grad = mean_grad  # Fallback

                # Apply filtered gradient
                model.zero_grad()
                idx = 0
                for p in model.parameters():
                    if p.grad is not None:
                        numel = p.numel()
                        p.grad = filtered_grad[idx:idx + numel].reshape(p.shape)
                        idx += numel

                optimizer.step()
                losses.append(np.mean(per_sample_losses))

            else:
                # Standard forward/backward
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output.view(-1, 256), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            if (step + 1) % 200 == 0:
                recent = np.mean(losses[-100:])
                print(f"    Step {step+1}: loss={recent:.4f}")

        results[mode] = np.array(losses)

    # Compare
    print(f"\n  Final loss comparison (last 200 steps):")
    for mode, losses in results.items():
        if len(losses) >= 200:
            final = np.mean(losses[-200:])
        else:
            final = np.mean(losses[-len(losses)//2:])
        print(f"    {mode}: {final:.4f}")

    return results


# --- CLI ---

def list_configs():
    """Print available Pareto-front configs."""
    for n_state in [16, 32]:
        configs = load_pareto_configs(n_state=n_state)
        if configs:
            print(f"\nE88 n_state={n_state}:")
            for i, c in enumerate(configs):
                print(f"  [{i}] dim={c.dim}, depth={c.depth}, n_heads={c.n_heads}, "
                      f"bs={c.batch_size}, lr={c.lr:.6f}, loss={c.avg_loss:.4f}, "
                      f"params={c.actual_params/1e6:.0f}M")


def main():
    parser = argparse.ArgumentParser(description="Gradient diagnostic experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["6", "7", "8", "9", "all"],
                        help="Which experiment to run")
    parser.add_argument("--config", type=str, default="best_bs1",
                        help="Config to use: 'best_bs1', 'best_overall', or index")
    parser.add_argument("--n-state", type=int, default=16, choices=[16, 32])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--n-steps", type=int, default=2000)
    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    # Load configs
    configs = load_pareto_configs(n_state=args.n_state)
    if not configs:
        print("No configs found. Check CMA-ES results directory.")
        return

    # Select config
    if args.config == "best_overall":
        config = min(configs, key=lambda c: c.avg_loss)
    elif args.config == "best_bs1":
        bs1 = [c for c in configs if c.batch_size == 1]
        config = min(bs1, key=lambda c: c.avg_loss) if bs1 else configs[0]
    elif args.config.isdigit():
        idx = int(args.config)
        config = configs[min(idx, len(configs) - 1)]
    else:
        config = configs[0]

    print(f"Selected config: dim={config.dim}, depth={config.depth}, "
          f"n_heads={config.n_heads}, bs={config.batch_size}, "
          f"lr={config.lr:.6f}, loss={config.avg_loss:.4f}")

    experiments = {
        "6": lambda: experiment6_gradient_cosine(config, n_steps=min(50, args.n_steps),
                                                  device=args.device),
        "7": lambda: experiment7_hidden_state_ablation(config, n_steps=args.n_steps,
                                                        device=args.device),
        "8": lambda: experiment8_gradient_noise_scale(config, device=args.device),
        "9": lambda: experiment9_gaf(config, n_steps=args.n_steps, device=args.device),
    }

    if args.experiment == "all":
        for exp_id in ["6", "7", "8", "9"]:
            try:
                experiments[exp_id]()
            except Exception as e:
                print(f"\n  Experiment {exp_id} failed: {e}")
    else:
        experiments[args.experiment]()


if __name__ == "__main__":
    main()
