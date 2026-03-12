from torch import nn
from einops import rearrange  # Essential for "patchifying" and video tensor reshapes


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size

        # A Linear layer is the most agnostic way to project if you
        # handle the "patch gathering" logic in the forward pass.
        # Calculation: (channels * patch_h * patch_w) or (channels * patch_t * patch_h * patch_w)
        # For simplicity, we'll assume a fixed patch volume for the linear input.
        self.proj = nn.Linear(in_chans * (patch_size ** 2), embed_dim)

    def forward(self, x):
        # x shape: (B, C, H, W) or (B, C, F, H, W)
        p = self.patch_size

        if x.ndim == 5:  # Video (B, C, F, H, W)
            # Assuming temporal patches of size 1 for now (standard in many DiT-video models)
            # If you want temporal patching (e.g. 2x2x2), change 'f' to '(f p_t)'
            x = rearrange(x, 'b c f (h p1) (w p2) -> b (f h w) (c p1 p2)', p1=p, p2=p)
        else:  # Image (B, C, H, W)
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)

        return self.proj(x)


class AdaLNZeroStrategy(nn.Module):
    def __init__(self, hidden_size, c):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(c, 6 * hidden_size)

        # Initialize to zero for identity behavior at start
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, condition):
        # Generate the 6 parameters
        params = self.linear(self.silu(condition)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params

        # This is where 'modulate' lived. Now it's internal logic.
        # We return the two modulated versions of x (for Attn and MLP) and the gates
        res_msa = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        res_mlp = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        return res_msa, gate_msa, res_mlp, gate_mlp


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, learn_variance=True):
        super().__init__()
        self.learn_sigma = learn_variance
        # If learning sigma, we need 2x the channels (mean + variance)
        self.output_dim = out_channels * 2 if learn_variance else out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * self.output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.norm_final(x)
        x = self.linear(x)
        return x
