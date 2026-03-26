"""
nn_potential_3body.py
---------------------
Simplified 3-body neural network potential.

Architecture:
    - Input: atomic positions (N_atoms, 3)
    - Descriptor: extract radial + angular features
    - Output: total energy (scalar)

This is a simplified version for demonstration.
"""

import torch
import torch.nn as nn


class RadialDescriptor(nn.Module):
    """
    Radial (2-body) descriptor: pairwise distances.
    
    For each atom i, compute distances to neighbors within cutoff.
    Then apply a neural network to each distance and sum.
    """
    
    def __init__(self, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        in_dim = 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())  # smooth activation
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, r):
        """Input: (N,) distances, Output: (N,) energies"""
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        return self.net(r).squeeze(-1)


class AngularDescriptor(nn.Module):
    """
    Angular (3-body) descriptor: bond angles.
    
    For each triplet (i, j, k), compute angle θ_ijk.
    Apply neural network to cos(θ) and sum.
    """
    
    def __init__(self, hidden_dims=[32, 16]):
        super().__init__()
        
        layers = []
        in_dim = 1  # cos(theta)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, cos_theta):
        """Input: (M,) cos(theta), Output: (M,) energies"""
        if cos_theta.dim() == 1:
            cos_theta = cos_theta.unsqueeze(-1)
        return self.net(cos_theta).squeeze(-1)


class ThreeBodyPotential(nn.Module):
    """
    Combined 2-body + 3-body potential.
    
    E_total = Σ_ij E_2b(r_ij) + Σ_ijk E_3b(θ_ijk)
    """
    
    def __init__(self, hidden_2b=[64, 32], hidden_3b=[32, 16]):
        super().__init__()
        self.radial = RadialDescriptor(hidden_2b)
        self.angular = AngularDescriptor(hidden_3b)
    
    def forward(self, r_pairs, cos_angles):
        """
        Args:
            r_pairs: (N_pairs,) pairwise distances
            cos_angles: (M_angles,) cos(theta) for each angle
        
        Returns:
            total_energy: scalar
        """
        e_2b = self.radial(r_pairs).sum() if len(r_pairs) > 0 else 0.0
        e_3b = self.angular(cos_angles).sum() if len(cos_angles) > 0 else 0.0
        return e_2b + e_3b
    
    def get_params_count(self):
        n_2b = sum(p.numel() for p in self.radial.parameters())
        n_3b = sum(p.numel() for p in self.angular.parameters())
        return {'2-body': n_2b, '3-body': n_3b, 'total': n_2b + n_3b}


if __name__ == "__main__":
    model = ThreeBodyPotential()
    print("3-body Potential Model")
    print(model.get_params_count())
