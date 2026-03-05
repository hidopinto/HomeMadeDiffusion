from torch import nn
from timm.models.vision_transformer import Attention, Mlp # Don't reinvent the wheel for standard ViT parts
from einops import rearrange, repeat  # Essential for "patchifying" and video tensor reshapes


class PatchEmbed(nn.Module):
    def forward(self, x):
        # x shape could be (B, C, H, W) for images or (B, C, F, H, W) for video
        # Logic: flatten all spatial/temporal dims into one sequence dim 'n'
        if x.ndim == 5:  # Video
            x = rearrange(x, 'b c f h w -> b (f h w) c')
        else:  # Image
            x = rearrange(x, 'b c h w -> b (h w) c')
        return self.proj(x)


class FinalLayer(nn.Module):
    pass
