import torch
import torch.nn as nn
from . import create_convblock1d


class GTIGU(nn.Module):
    """
    Geometry-Texture Interaction Gating Unit (GT-IGU)
    
    - Bayesian Evidence Vector: E = Concat(F_g ⊙ F_t, (F_g - F_t)^2)
    - Gate: Λ = σ(φ(E))
    - Fusion: Z_opt = Λ * F_g + (1 - Λ) * F_t
    
    Inverse Variance Weighting (Eq. 8):
    Λ* = diag(Σ_t) / (diag(Σ_g) + diag(Σ_t))
    When texture noise dominates (Σ_t >> Σ_g), Λ* → 1, favoring geometry.
    """

    def __init__(self, in_channels, return_gate=False):
        """
        Args:
            in_channels: Feature dimension
            return_gate: If True, return gate values for visualization
        """
        super().__init__()
        self.in_channels = in_channels
        self.return_gate = return_gate

        self.gate_mlp = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._cached_gate = None
        self._cached_s_align = None
        self._cached_s_diff = None

    def forward(self, f_geo, f_tex, return_gate=None):
        """
        Args:
            f_geo: Geometry features (B, C, N)
            f_tex: Texture features (B, C, N)
            return_gate: Whether to return gate values (overrides init setting)
        Returns:
            f_fused: Fused features (B, C, N)
            gate_info (optional): Gate info dict for visualization
        """
        s_align = f_geo * f_tex

        s_diff = torch.pow(f_geo - f_tex, 2)

        combined = torch.cat([s_align, s_diff], dim=1)  # (B, 2C, N)
        alpha = self.gate_mlp(combined)  # (B, C, N), Λ = σ(φ(E))

        f_fused = alpha * f_geo + (1 - alpha) * f_tex
        
        if not self.training:
            self._cached_gate = alpha.detach()
            self._cached_s_align = s_align.detach()
            self._cached_s_diff = s_diff.detach()
        
        should_return_gate = return_gate if return_gate is not None else self.return_gate
        if should_return_gate:
            gate_info = {
                'gate': alpha,
                's_align': s_align,
                's_diff': s_diff,
                'f_geo': f_geo,
                'f_tex': f_tex,
            }
            return f_fused, gate_info

        return f_fused
    
    def get_cached_gate(self):
        """Get cached gate values for visualization."""
        if self._cached_gate is None:
            return None
        return {
            'gate': self._cached_gate,
            's_align': self._cached_s_align,
            's_diff': self._cached_s_diff,
        }
    
    def clear_cache(self):
        """Clear cached gate values to free memory."""
        self._cached_gate = None
        self._cached_s_align = None
        self._cached_s_diff = None
