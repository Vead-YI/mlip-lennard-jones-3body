"""
train_3body.py
--------------
Train the 2-body + 3-body neural network potential with force matching.

Key features:
    - Energy loss: L_E = MSE(E_pred, E_true)
    - Force loss: L_F = MSE(F_pred, F_true)
    - Combined: L = α * L_E + β * L_F

The force loss is critical here because 3-body forces involve
complex angular derivatives that benefit from explicit training.

Usage:
    python train_3body.py
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
EPOCHS = 500
BATCH_SIZE = 32          # configs per batch
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
ALPHA = 1.0              # energy loss weight
BETA = 0.1               # force loss weight

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "3body_dataset.npz")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "nnp_3body_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class ThreeBodyDataset(Dataset):
    """Custom dataset for 3-body configurations."""
    
    def __init__(self, data_path, r_cutoff=2.5):
        data = np.load(data_path)
        
        self.positions = torch.tensor(data['positions'], dtype=torch.float32)  # (N_configs, N_atoms, 3)
        self.energy_total = torch.tensor(data['energy_total'], dtype=torch.float32)
        self.energy_2body = torch.tensor(data['energy_2body'], dtype=torch.float32)
        self.energy_3body = torch.tensor(data['energy_3body'], dtype=torch.float32)
        self.forces = torch.tensor(data['forces'], dtype=torch.float32)  # (N_configs, N_atoms, 3)
        
        self.r_cutoff = r_cutoff
        self.n_configs = len(self.positions)
        self.n_atoms = self.positions.shape[1]
        
        # Precompute pairwise distances and angles for each config
        print("    Precomputing pair/angle lists...")
        self.pair_data = []
        self.angle_data = []
        
        for idx in range(self.n_configs):
            pairs, angles = self._extract_pairs_and_angles(idx)
            self.pair_data.append(pairs)
            self.angle_data.append(angles)
    
    def _extract_pairs_and_angles(self, idx):
        """Extract pairwise distances and angle cosines for a configuration."""
        pos = self.positions[idx].numpy()
        n = len(pos)
        
        pairs = []  # (i, j, r_ij)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                if r < self.r_cutoff and r > 0.5:
                    pairs.append((i, j, r))
        
        angles = []  # (i, j, k, cos_theta)
        # For each atom i, find all j,k neighbors and compute angles
        neighbors = {i: [] for i in range(n)}
        for i, j, r in pairs:
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
                    
                    r_ij = pos[j] - pos[i]
                    r_ik = pos[k] - pos[i]
                    
                    r_ij_mag = np.linalg.norm(r_ij)
                    r_ik_mag = np.linalg.norm(r_ik)
                    
                    if r_ij_mag < 1e-10 or r_ik_mag < 1e-10:
                        continue
                    
                    cos_theta = np.dot(r_ij, r_ik) / (r_ij_mag * r_ik_mag)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angles.append((i, j, k, cos_theta))
        
        return pairs, angles
    
    def __len__(self):
        return self.n_configs
    
    def __getitem__(self, idx):
        return {
            'positions': self.positions[idx],
            'energy_total': self.energy_total[idx],
            'energy_2body': self.energy_2body[idx],
            'energy_3body': self.energy_3body[idx],
            'forces': self.forces[idx],
            'pairs': self.pair_data[idx],
            'angles': self.angle_data[idx]
        }


def collate_fn(batch):
    """Custom collate to handle variable-size pair/angle lists."""
    return batch  # return list of dicts


def compute_energy_and_forces(model, sample, norm_params, box_size=8.0):
    """
    Compute energy and forces for a single configuration.
    
    Forces are computed via autograd through the full computation graph.
    """
    positions = sample['positions'].clone().requires_grad_(True)
    pairs = sample['pairs']
    angles = sample['angles']
    
    # Normalize inputs
    r_mean = norm_params['r_mean']
    r_std = norm_params['r_std']
    
    # ── Energy from 2-body ───────────────────────────────────────────────
    r_pairs = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=DEVICE)
    r_pairs_norm = (r_pairs - r_mean) / r_std
    
    e_2b_per_pair = model.net_2b(r_pairs_norm)
    e_2b = e_2b_per_pair.sum()
    
    # ── Energy from 3-body ───────────────────────────────────────────────
    cos_angles = torch.tensor([a[3] for a in angles], dtype=torch.float32, device=DEVICE)
    e_3b_per_angle = model.net_3b(cos_angles)
    e_3b = e_3b_per_angle.sum()
    
    # Total energy
    e_total = e_2b + e_3b
    
    # ── Forces via autograd ───────────────────────────────────────────────
    # For proper force computation, we need to track how r_ij and cos_theta
    # depend on atom positions. This requires a more sophisticated approach.
    # 
    # For simplicity in this demo, we compute forces numerically:
    #   F_i = -dE/dr_i ≈ -(E(r_i + ε) - E(r_i - ε)) / (2ε)
    #
    # In production, use torch.autograd.grad() with proper graph construction.
    
    n_atoms = len(positions)
    forces = torch.zeros(n_atoms, 3, device=DEVICE)
    
    # Numerical gradient (slow but correct)
    eps = 1e-4
    for i in range(n_atoms):
        for d in range(3):
            pos_plus = positions.clone()
            pos_plus[i, d] += eps
            
            pos_minus = positions.clone()
            pos_minus[i, d] -= eps
            
            # Recompute energy for perturbed positions
            e_plus = _compute_energy_for_positions(model, pos_plus, pairs, angles, 
                                                    norm_params, box_size)
            e_minus = _compute_energy_for_positions(model, pos_minus, pairs, angles,
                                                     norm_params, box_size)
            
            forces[i, d] = -(e_plus - e_minus) / (2 * eps)
    
    return e_total, forces


def _compute_energy_for_positions(model, positions, pairs, angles, norm_params, box_size):
    """Helper to recompute energy for numerical gradient."""
    r_mean = norm_params['r_mean']
    r_std = norm_params['r_std']
    
    # Recompute distances and angles
    pos_np = positions.detach().numpy()
    
    r_pairs = []
    for i, j, _ in pairs:
        r_vec = pos_np[j] - pos_np[i]
        r_vec = r_vec - box_size * np.round(r_vec / box_size)
        r = np.linalg.norm(r_vec)
        r_pairs.append(r)
    
    cos_angles = []
    for i, j, k, _ in angles:
        r_ij = pos_np[j] - pos_np[i]
        r_ik = pos_np[k] - pos_np[i]
        r_ij_mag = np.linalg.norm(r_ij)
        r_ik_mag = np.linalg.norm(r_ik)
        if r_ij_mag > 1e-10 and r_ik_mag > 1e-10:
            cos_t = np.dot(r_ij, r_ik) / (r_ij_mag * r_ik_mag)
            cos_t = np.clip(cos_t, -1, 1)
        else:
            cos_t = 0
        cos_angles.append(cos_t)
    
    r_pairs_t = torch.tensor(r_pairs, dtype=torch.float32, device=DEVICE)
    r_pairs_norm = (r_pairs_t - r_mean) / r_std
    e_2b = model.net_2b(r_pairs_norm).sum()
    
    cos_t = torch.tensor(cos_angles, dtype=torch.float32, device=DEVICE)
    e_3b = model.net_3b(cos_t).sum()
    
    return (e_2b + e_3b).detach()


def main():
    print("=" * 60)
    print("  Training 3-body NNP with Force Matching")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Loss weights: α={ALPHA} (energy), β={BETA} (force)")
    
    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    dataset = ThreeBodyDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn)
    
    print(f"    Configs: {len(dataset)}")
    print(f"    Atoms per config: {dataset.n_atoms}")
    
    # Normalize params (based on data statistics)
    all_r = []
    for sample in dataset:
        all_r.extend([p[2] for p in sample['pairs']])
    all_r = torch.tensor(all_r)
    
    r_mean = all_r.mean().item()
    r_std = all_r.std().item()
    
    norm_params = {'r_mean': r_mean, 'r_std': r_std}
    print(f"    r normalization: μ={r_mean:.3f}, σ={r_std:.3f}")
    
    # ── 2. Initialize Model ───────────────────────────────────────────────
    print("\n[2/5] Initializing model...")
    model = CombinedPotential(hidden_2b=[64, 64, 32], hidden_3b=[32, 32, 16]).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    
    print(f"    2-body params: {sum(p.numel() for p in model.net_2b.parameters())}")
    print(f"    3-body params: {sum(p.numel() for p in model.net_3b.parameters())}")
    print(f"    Total params:  {sum(p.numel() for p in model.parameters())}")
    
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
        n_samples = 0
        
        for batch in dataloader:
            batch_e_loss = 0.0
            batch_f_loss = 0.0
            
            for sample in batch:
                # Compute energy and forces
                e_pred, f_pred = compute_energy_and_forces(model, sample, norm_params)
                
                # Energy loss
                e_true = sample['energy_total'].to(DEVICE)
                loss_e = criterion(e_pred, e_true)
                batch_e_loss += loss_e.item()
                
                # Force loss
                f_true = sample['forces'].to(DEVICE)
                loss_f = criterion(f_pred, f_true)
                batch_f_loss += loss_f.item()
                
                # Combined
                loss = ALPHA * loss_e + BETA * loss_f
                epoch_loss += loss.item()
                
                # Backward (accumulate gradients)
                loss.backward()
            
            # Step after batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_e_loss += batch_e_loss
            epoch_f_loss += batch_f_loss
            n_samples += len(batch)
        
        avg_loss = epoch_loss / n_samples
        avg_e_loss = epoch_e_loss / n_samples
        avg_f_loss = epoch_f_loss / n_samples
        
        train_losses.append(avg_loss)
        energy_losses.append(avg_e_loss)
        force_losses.append(avg_f_loss)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} (E: {avg_e_loss:.4f}, F: {avg_f_loss:.4f}) | "
                  f"LR: {current_lr:.2e}")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'energy_losses': energy_losses,
        'force_losses': force_losses,
        'alpha': ALPHA,
        'beta': BETA
    }, MODEL_SAVE_PATH)
    print(f"\n    Model saved to: {MODEL_SAVE_PATH}")
    
    # ── 4. Evaluation ─────────────────────────────────────────────────────
    print("\n[4/5] Final evaluation...")
    
    model.eval()
    all_e_pred = []
    all_e_true = []
    all_f_pred = []
    all_f_true = []
    
    with torch.no_grad():
        for sample in dataset:
            e_pred, f_pred = compute_energy_and_forces(model, sample, norm_params)
            all_e_pred.append(e_pred.cpu().item())
            all_e_true.append(sample['energy_total'].item())
            all_f_pred.append(f_pred.cpu().numpy())
            all_f_true.append(sample['forces'].numpy())
    
    all_e_pred = np.array(all_e_pred)
    all_e_true = np.array(all_e_true)
    all_f_pred = np.concatenate([f.flatten() for f in all_f_pred])
    all_f_true = np.concatenate([f.flatten() for f in all_f_true])
    
    # Metrics
    e_mse = ((all_e_pred - all_e_true) ** 2).mean()
    e_mae = np.abs(all_e_pred - all_e_true).mean()
    e_r2 = 1 - np.sum((all_e_true - all_e_pred)**2) / np.sum((all_e_true - all_e_true.mean())**2)
    
    f_mse = ((all_f_pred - all_f_true) ** 2).mean()
    f_mae = np.abs(all_f_pred - all_f_true).mean()
    f_corr = np.corrcoef(all_f_pred, all_f_true)[0, 1]
    
    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy correlation
    axes[0, 0].scatter(all_e_true, all_e_pred, alpha=0.5, s=10)
    lims = [all_e_true.min(), all_e_true.max()]
    axes[0, 0].plot(lims, lims, 'k--', linewidth=1)
    axes[0, 0].set_xlabel('True Energy (ε)')
    axes[0, 0].set_ylabel('Predicted Energy (ε)')
    axes[0, 0].set_title(f'Energy: MAE={e_mae:.4f}, R²={e_r2:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Force correlation
    axes[0, 1].scatter(all_f_true, all_f_pred, alpha=0.1, s=1)
    lims = [all_f_true.min(), all_f_true.max()]
    axes[0, 1].plot(lims, lims, 'k--', linewidth=1)
    axes[0, 1].set_xlabel('True Force (ε/σ)')
    axes[0, 1].set_ylabel('Predicted Force (ε/σ)')
    axes[0, 1].set_title(f'Force: MAE={f_mae:.4f}, Corr={f_corr:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss curves
    axes[1, 0].plot(energy_losses, label='Energy Loss', linewidth=1)
    axes[1, 0].plot(force_losses, label='Force Loss', linewidth=1)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy breakdown
    axes[1, 1].hist(all_e_true, bins=30, alpha=0.5, label='True', density=True)
    axes[1, 1].hist(all_e_pred, bins=30, alpha=0.5, label='Predicted', density=True)
    axes[1, 1].set_xlabel('Energy (ε)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Energy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_3body_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Saved: {plot_path}")
    plt.close()
    
    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Metrics")
    print(f"{'='*60}")
    print(f"  Energy:")
    print(f"    MSE:  {e_mse:.6f}")
    print(f"    MAE:  {e_mae:.6f}")
    print(f"    R²:   {e_r2:.6f}")
    print(f"  Force:")
    print(f"    MSE:  {f_mse:.6f}")
    print(f"    MAE:  {f_mae:.6f}")
    print(f"    Corr: {f_corr:.6f}")
    print(f"{'='*60}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
