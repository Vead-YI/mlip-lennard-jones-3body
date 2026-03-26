"""
generate_data_3body.py
----------------------
Generate training data for 3-body interatomic potentials.

This includes:
  - 2-body: LJ potential V_2(r_ij)
  - 3-body: Angle potential V_3(r_ij, r_ik, theta_ijk)

For simplicity, we use a toy 3-body potential:
    V_3(theta) = K * (cos(theta) - cos(theta_0))^2

This captures the physics of bond angle bending.

Output:
    data/3body_dataset.npz
"""

import numpy as np
import os

# ── Parameters ───────────────────────────────────────────────────────────
SIGMA = 1.0
EPSILON = 1.0
K_ANGLE = 0.5           # 3-body angle strength
THETA_0 = np.pi / 3     # equilibrium angle (60 degrees)
R_MIN = 0.85
R_MAX = 3.5
R_CUTOFF = 2.5

N_CONFIGS = 3000        # number of atomic configurations
N_ATOMS = 10            # atoms per configuration


def lj_potential(r, epsilon=EPSILON, sigma=SIGMA):
    """2-body Lennard-Jones potential."""
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def lj_force_magnitude(r, epsilon=EPSILON, sigma=SIGMA):
    """2-body LJ force magnitude F = -dV/dr."""
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return 24.0 * epsilon / r * (2.0 * sr12 - sr6)


def angle_potential(theta, k=K_ANGLE, theta_0=THETA_0):
    """3-body angle bending potential."""
    return k * (np.cos(theta) - np.cos(theta_0)) ** 2


def angle_potential_gradient(theta, k=K_ANGLE, theta_0=THETA_0):
    """dV/d(theta) for angle potential."""
    return 2.0 * k * (np.cos(theta) - np.cos(theta_0)) * (-np.sin(theta))


def compute_angle(r_ij, r_ik):
    """
    Compute angle theta_ijk between vectors r_ij and r_ik.
    r_ij = r_j - r_i, r_ik = r_k - r_i
    """
    # Normalize
    r_ij_norm = np.linalg.norm(r_ij)
    r_ik_norm = np.linalg.norm(r_ik)
    
    if r_ij_norm < 1e-10 or r_ik_norm < 1e-10:
        return np.pi  # degenerate
    
    cos_theta = np.dot(r_ij, r_ik) / (r_ij_norm * r_ik_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return np.arccos(cos_theta)


def generate_random_config(n_atoms, box_size, min_dist=0.9, seed=None):
    """Generate a random atomic configuration avoiding overlaps."""
    rng = np.random.default_rng(seed)
    positions = []
    
    for _ in range(n_atoms):
        attempts = 0
        while attempts < 500:
            candidate = rng.uniform(1.0, box_size - 1.0, size=3)
            
            overlap = False
            for existing in positions:
                dist = np.linalg.norm(candidate - existing)
                if dist < min_dist:
                    overlap = True
                    break
            
            if not overlap:
                positions.append(candidate)
                break
            attempts += 1
        
        if attempts == 500:
            positions.append(rng.uniform(1.0, box_size - 1.0, size=3))
    
    return np.array(positions)


def compute_energy_and_forces(positions, box_size):
    """
    Compute total energy and forces for a configuration.
    
    Returns:
        energy_2body: float
        energy_3body: float
        forces: (N, 3) array
    """
    n = len(positions)
    forces = np.zeros((n, 3))
    energy_2body = 0.0
    energy_3body = 0.0
    
    # ── 2-body interactions ────────────────────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            # PBC
            r_vec = r_vec - box_size * np.round(r_vec / box_size)
            r = np.linalg.norm(r_vec)
            
            if r > R_CUTOFF or r < 0.5:
                continue
            
            # Energy
            energy_2body += lj_potential(r)
            
            # Force on i from j
            f_mag = lj_force_magnitude(r)
            f_vec = f_mag * r_vec / r
            forces[i] += f_vec
            forces[j] -= f_vec
    
    # ── 3-body angle interactions ───────────────────────────────────────
    for i in range(n):
        neighbors = []
        for j in range(n):
            if j == i:
                continue
            r_vec = positions[j] - positions[i]
            r_vec = r_vec - box_size * np.round(r_vec / box_size)
            r = np.linalg.norm(r_vec)
            if r < R_CUTOFF and r > 0.5:
                neighbors.append((j, r_vec, r))
        
        # All pairs of neighbors form angles
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                j, r_ij, r_ij_mag = neighbors[a]
                k, r_ik, r_ik_mag = neighbors[b]
                
                # Compute angle
                theta = compute_angle(r_ij, r_ik)
                
                # Energy
                energy_3body += angle_potential(theta)
                
                # Force: chain rule dV/d(theta) * d(theta)/dr
                # This is complex; we use numerical gradient for simplicity
                # F_i = -dV/dr_i, F_j = -dV/dr_j, F_k = -dV/dr_k
                
                dV_dtheta = angle_potential_gradient(theta)
                
                # d(theta)/dr_ij and d(theta)/dr_ik (analytical)
                # theta = arccos(cos_theta)
                # d(theta)/dr_ij = -1/sin(theta) * d(cos_theta)/dr_ij
                
                sin_theta = np.sin(theta)
                if abs(sin_theta) < 1e-10:
                    continue
                
                r_ji = -r_ij  # i -> j vector
                r_ki = -r_ik  # i -> k vector
                
                # d(cos_theta)/dr_ij = (r_ik - cos_theta * r_ij) / (|r_ij| * |r_ik|)
                cos_theta = np.dot(r_ij, r_ik) / (r_ij_mag * r_ik_mag)
                
                # Gradient with respect to r_ij
                dcos_drij = (r_ik / (r_ij_mag * r_ik_mag) - 
                            cos_theta * r_ij / (r_ij_mag ** 2))
                dcos_drik = (r_ij / (r_ij_mag * r_ik_mag) - 
                            cos_theta * r_ik / (r_ik_mag ** 2))
                
                dtheta_drij = -dV_dtheta / sin_theta * dcos_drij
                dtheta_drik = -dV_dtheta / sin_theta * dcos_drik
                
                # Forces
                # F_i = -dV/dr_i = -dV/d(theta) * d(theta)/dr_i
                # But r_ij = r_j - r_i, so d(theta)/dr_i = -dtheta_drij - dtheta_drik
                forces[i] += dtheta_drij + dtheta_drik
                forces[j] -= dtheta_drij
                forces[k] -= dtheta_drik
    
    return energy_2body, energy_3body, forces


def generate_dataset(n_configs, n_atoms, box_size, seed_start=0):
    """Generate full dataset."""
    all_positions = []
    all_energy_2body = []
    all_energy_3body = []
    all_forces = []
    
    for idx in range(n_configs):
        positions = generate_random_config(n_atoms, box_size, seed=seed_start + idx)
        e2, e3, forces = compute_energy_and_forces(positions, box_size)
        
        all_positions.append(positions)
        all_energy_2body.append(e2)
        all_energy_3body.append(e3)
        all_forces.append(forces)
        
        if (idx + 1) % 500 == 0:
            print(f"    Generated {idx + 1}/{n_configs} configurations")
    
    return (np.array(all_positions), 
            np.array(all_energy_2body), 
            np.array(all_energy_3body),
            np.array(all_forces))


def main():
    print("=" * 60)
    print("  Generating 3-body Potential Dataset")
    print("=" * 60)
    
    BOX_SIZE = 8.0
    
    print(f"\nConfig: {N_CONFIGS} configs × {N_ATOMS} atoms")
    print(f"Box size: {BOX_SIZE}×{BOX_SIZE}×{BOX_SIZE}")
    print(f"Angle strength K = {K_ANGLE}, equilibrium θ₀ = {np.degrees(THETA_0):.1f}°")
    
    print("\nGenerating configurations...")
    positions, e2, e3, forces = generate_dataset(N_CONFIGS, N_ATOMS, BOX_SIZE)
    
    print(f"\nDataset statistics:")
    print(f"  2-body energy range: [{e2.min():.3f}, {e2.max():.3f}]")
    print(f"  3-body energy range: [{e3.min():.3f}, {e3.max():.3f}]")
    print(f"  Total energy range:  [{(e2+e3).min():.3f}, {(e2+e3).max():.3f}]")
    print(f"  Force range:         [{forces.min():.3f}, {forces.max():.3f}]")
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), "3body_dataset.npz")
    np.savez(out_path, 
             positions=positions,
             energy_2body=e2,
             energy_3body=e3,
             energy_total=e2 + e3,
             forces=forces)
    print(f"\nDataset saved to: {out_path}")
    
    # Quick visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        
        axes[0].hist(e2, bins=50, alpha=0.7, color='steelblue', edgecolor='k')
        axes[0].set_xlabel('2-body Energy (ε)')
        axes[0].set_title('2-body Energy Distribution')
        
        axes[1].hist(e3, bins=50, alpha=0.7, color='tomato', edgecolor='k')
        axes[1].set_xlabel('3-body Energy (ε)')
        axes[1].set_title('3-body Energy Distribution')
        
        axes[2].hist(forces.flatten(), bins=50, alpha=0.7, color='forestgreen', edgecolor='k')
        axes[2].set_xlabel('Force Component (ε/σ)')
        axes[2].set_title('Force Distribution')
        
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "3body_data_dist.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"Data distribution plot saved to: {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot.")
    
    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()
