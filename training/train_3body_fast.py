"""
train_3body_fast.py
-------------------
Fast training for 3-body potential using analytical gradients.

Key optimization: Build the full computation graph so autograd can
compute forces efficiently, avoiding slow numerical gradients.

Usage:
    python train_3body_fast.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.nn_potential_3body import CombinedPotential


# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 1           # one config at a time for autograd
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
ALPHA = 1.0              # energy loss weight
BETA = 0.01              # force loss weight

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "3body_dataset.npz")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "nnp_3body_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data(path):
    """Load 3-body dataset."""
    data = np.load(path)
    return (
        torch.tensor(data['positions'], dtype=torch.float32),
        torch.tensor(data['energy_total'], dtype=torch.float32),
        torch.tensor(data['forces'], dtype=torch.float32)
    )


def compute_pairs_and_angles(positions, box_size=8.0, r_cutoff=2.5):
    """
    Compute pairwise indices and angle indices for a configuration.
    Returns tensors that can be used for autograd.
    """
    n = len(positions)
    
    # Pairwise distances with autograd support
    pairs_i = []
    pairs_j = []
    
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            r_vec = r_vec - box_size * torch.round(r_vec / box_size)
            r = torch.norm(r_vec)
            if r < r_cutoff and r > 0.5:
                pairs_i.append(i)
                pairs_j.append(j)
    
    # Angles
    angle_i = []
    angle_j = []
    angle_k = []
    
    # Build neighbor list
    neighbors = {i: [] for i in range(n)}
    for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    for i in range(n):
        neigh = neighbors[i]
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                j = neigh[a]
                k = neigh[b]
                if j == k:
                    continue
                angle_i.append(i)
                angle_j.append(j)
                angle_k.append(k)
    
    return pairs_i, pairs_j, angle_i, angle_j, angle_k


def compute_energy_with_grad(model, positions, pairs_i, pairs_j, angle_i, angle_j, angle_k,
                              box_size, norm_params):
    """
    Compute energy with full autograd graph for force computation.
    """
    n = len(positions)
    r_mean = norm_params['r_mean']
    r_std = norm_params['r_std']
    
    # Compute pairwise distances (autograd-friendly)
    r_pairs = []
    for i, j in zip(pairs_i, pairs_j):
        r_vec = positions[j] - positions[i]
        r_vec = r_vec - box_size * torch.round(r_vec / box_size)
        r = torch.norm(r_vec)
        r_pairs.append(r)
    
    if len(r_pairs) > 0:
        r_pairs = torch.stack(r_pairs)
        r_pairs_norm = (r_pairs - r_mean) / r_std
        e_2b = model.net_2b(r_pairs_norm).sum()
    else:
        e_2b = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    
    # Compute angles
    cos_angles = []
    for i, j, k in zip(angle_i, angle_j, angle_k):
        r_ij = positions[j] - positions[i]
        r_ik = positions[k] - positions[i]
        r_ij = r_ij - box_size * torch.round(r_ij / box_size)
        r_ik = r_ik - box_size * torch.round(r_ik / box_size)
        
        r_ij_mag = torch.norm(r_ij)
        r_ik_mag = torch.norm(r_ik)
        
        cos_t = torch.dot(r_ij, r_ik) / (r_ij_mag * r_ik_mag + 1e-10)
        cos_t = torch.clamp(cos_t, -0.999, 0.999)
        cos_angles.append(cos_t)
    
    if len(cos_angles) > 0:
        cos_angles = torch.stack(cos_angles)
        e_3b = model.net_3b(cos_angles).sum()
    else:
        e_3b = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    
    return e_2b + e_3b


def main():
    print("=" * 60)
    print("  Fast 3-body Training with Autograd Forces")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Loss weights: α={ALPHA} (energy), β={BETA} (force)")
    
    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    positions, energies, forces = load_data(DATA_PATH)
    n_configs = len(positions)
    n_atoms = positions.shape[1]
    
    print(f"    Configs: {n_configs}")
    print(f"    Atoms per config: {n_atoms}")
    
    # Normalize params
    norm_params = {'r_mean': 1.5, 'r_std': 0.6}  # typical values
    
    # Precompute pair/angle indices
    print("    Precomputing topology...")
    all_topology = []
    for idx in range(n_configs):
        topology = compute_pairs_and_angles(positions[idx])
        all_topology.append(topology)
    
    # ── 2. Initialize Model ───────────────────────────────────────────────
    print("\n[2/5] Initializing model...")
    model = CombinedPotential(hidden_2b=[64, 64, 32], hidden_3b=[32, 32, 16]).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    print(f"    Total params: {sum(p.numel() for p in model.parameters())}")
    
    # ── 3. Training Loop ──────────────────────────────────────────────────
    print(f"\n[3/5] Training for {EPOCHS} epochs...")
    
    train_losses = []
    energy_losses = []
    force_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        
        epoch_loss = 0.0
        epoch_e_loss = 0.0
        epoch_f_loss = 0.0
        
        # Shuffle indices
        perm = np.random.permutation(n_configs)
        
        for count, idx in enumerate(perm):
            pos = positions[idx].clone().to(DEVICE).requires_grad_(True)
            e_true = energies[idx].to(DEVICE)
            f_true = forces[idx].to(DEVICE)
            topology = all_topology[idx]
            
            # Forward: compute energy
            e_pred = compute_energy_with_grad(
                model, pos, *topology, box_size=8.0, norm_params=norm_params
            )
            
            # Energy loss
            loss_e = (e_pred - e_true) ** 2
            
            # Force loss via autograd
            f_pred = torch.autograd.grad(
                outputs=e_pred,
                inputs=pos,
                grad_outputs=torch.ones_like(e_pred),
                create_graph=True,
                retain_graph=True
            )[0]
            
            loss_f = ((f_pred - f_true) ** 2).mean()
            
            # Combined
            loss = ALPHA * loss_e + BETA * loss_f
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_e_loss += loss_e.item()
            epoch_f_loss += loss_f.item()
            
            if (count + 1) % 500 == 0:
                print(f"      Processed {count + 1}/{n_configs} configs")
        
        avg_loss = epoch_loss / n_configs
        avg_e_loss = epoch_e_loss / n_configs
        avg_f_loss = epoch_f_loss / n_configs
        
        train_losses.append(avg_loss)
        energy_losses.append(avg_e_loss)
        force_losses.append(avg_f_loss)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"    Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} (E: {avg_e_loss:.4f}, F: {avg_f_loss:.4f}) | "
              f"LR: {current_lr:.2e}")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'energy_losses': energy_losses,
        'force_losses': force_losses
    }, MODEL_SAVE_PATH)
    print(f"\n    Model saved to: {MODEL_SAVE_PATH}")
    
    # ── 4. Evaluation ─────────────────────────────────────────────────────
    print("\n[4/5] Evaluating...")
    
    model.eval()
    all_e_pred = []
    all_e_true = []
    all_f_pred = []
    all_f_true = []
    
    with torch.no_grad():
        for idx in range(n_configs):
            pos = positions[idx].to(DEVICE)
            e_true = energies[idx].item()
            f_true = forces[idx].numpy()
            topology = all_topology[idx]
            
            e_pred = compute_energy_with_grad(
                model, pos, *topology, box_size=8.0, norm_params=norm_params
            )
            
            all_e_pred.append(e_pred.item())
            all_e_true.append(e_true)
    
    all_e_pred = np.array(all_e_pred)
    all_e_true = np.array(all_e_true)
    
    e_mae = np.abs(all_e_pred - all_e_true).mean()
    e_r2 = 1 - np.sum((all_e_true - all_e_pred)**2) / np.sum((all_e_true - all_e_true.mean())**2)
    
    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].scatter(all_e_true, all_e_pred, alpha=0.5, s=10)
    lims = [all_e_true.min(), all_e_true.max()]
    axes[0].plot(lims, lims, 'k--', linewidth=1)
    axes[0].set_xlabel('True Energy (ε)')
    axes[0].set_ylabel('Predicted Energy (ε)')
    axes[0].set_title(f'Energy: MAE={e_mae:.4f}, R²={e_r2:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(energy_losses, label='Energy Loss', linewidth=1)
    axes[1].plot(force_losses, label='Force Loss', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Training Losses')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_3body_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Saved: {plot_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"  Energy MAE: {e_mae:.4f}")
    print(f"  Energy R²:  {e_r2:.4f}")
    print(f"{'='*60}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
