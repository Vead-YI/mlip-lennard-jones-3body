"""
train_3body_minimal.py
----------------------
Minimal training script for 3-body potential (energy-only, fast).

This version focuses on getting results quickly for demonstration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cpu")  # force CPU for stability
EPOCHS = 50
LR = 1e-3

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "3body_dataset.npz")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    print("Loading data...")
    data = np.load(DATA_PATH)
    
    # Use subsample for speed
    n_samples = 500
    indices = np.random.choice(len(data['positions']), n_samples, replace=False)
    
    # Precompute simple descriptors
    BOX = 8.0
    CUTOFF = 2.5
    
    all_r_pairs = []
    all_energies = []
    
    print("Computing descriptors...")
    for idx in indices:
        pos = data['positions'][idx]
        n = len(pos)
        e = data['energy_total'][idx]
        
        # Just pairwise distances
        r_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                r_vec = pos[j] - pos[i]
                r_vec = r_vec - BOX * np.round(r_vec / BOX)
                r = np.linalg.norm(r_vec)
                if r < CUTOFF and r > 0.5:
                    r_pairs.append(r)
        
        if len(r_pairs) > 0:
            all_r_pairs.append(torch.tensor(r_pairs, dtype=torch.float32))
            all_energies.append(torch.tensor(e, dtype=torch.float32))
    
    # Normalize
    all_r = torch.cat(all_r_pairs)
    r_mean, r_std = all_r.mean().item(), all_r.std().item()
    
    print(f"Dataset: {len(all_r_pairs)} configs")
    print(f"r: μ={r_mean:.3f}, σ={r_std:.3f}")
    
    # Simple model
    print("\nInitializing model...")
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.Linear(32, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    
    losses = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        perm = np.random.permutation(len(all_r_pairs))
        for i in perm:
            r = all_r_pairs[i]
            e_true = all_energies[i]
            
            # Normalize
            r_norm = (r - r_mean) / r_std
            r_input = r_norm.unsqueeze(-1)
            
            e_pred = model(r_input).sum()
            loss = (e_pred - e_true) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(all_r_pairs)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    
    all_e_pred = []
    all_e_true = []
    
    with torch.no_grad():
        for i in range(len(all_r_pairs)):
            r = all_r_pairs[i]
            r_norm = (r - r_mean) / r_std
            e_pred = model(r_norm.unsqueeze(-1)).sum().item()
            
            all_e_pred.append(e_pred)
            all_e_true.append(all_energies[i].item())
    
    all_e_pred = np.array(all_e_pred)
    all_e_true = np.array(all_e_true)
    
    mae = np.abs(all_e_pred - all_e_true).mean()
    r2 = 1 - np.sum((all_e_true - all_e_pred)**2) / np.sum((all_e_true - all_e_true.mean())**2)
    
    print(f"\nResults:")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}")
    
    # Plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].scatter(all_e_true, all_e_pred, alpha=0.5, s=20)
    lims = [min(all_e_true.min(), all_e_pred.min()), max(all_e_true.max(), all_e_pred.max())]
    axes[0].plot(lims, lims, 'k--', label='y=x')
    axes[0].set_xlabel('True Energy')
    axes[0].set_ylabel('Predicted Energy')
    axes[0].set_title(f'Energy Prediction (MAE={mae:.3f}, R²={r2:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "3body_minimal_results.png"), dpi=150)
    print(f"\nSaved plot to: {RESULTS_DIR}/3body_minimal_results.png")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
