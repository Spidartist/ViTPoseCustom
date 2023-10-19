import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from ..builder import BACKBONES
import numpy as np
import einops


def patchify(images, patch_size=4):
    """Splitting images into patches.
    Args:
        images: Input tensor with size (batch, channels, height, width)
            We can assume that image is square where height == width.
    Returns:
        A batch of image patches with size (
          batch, (height / patch_size) * (width / patch_size),
        channels * patch_size * patch_size)
    """
    # BEGIN YOUR CODE
    _, _, height, width = images.shape
    new_height = height / patch_size
    new_width = width / patch_size
    images = einops.rearrange(images, 'b c (h ps1) (w ps2) -> b (h w) (c ps1 ps2)', h=int(new_height), w=int(new_width))
    return images
    # END YOUR CODE


def unpatchify(patches, patch_size=4):
    """Combining patches into images.
    Args:
        patches: Input tensor with size (
        batch, (height / patch_size) * (width / patch_size),
        channels * patch_size * patch_size)
    Returns:
        A batch of images with size (batch, channels, height, width)
    """
    # BEGIN YOUR CODE
    _, x, y = patches.shape
    height = int(np.sqrt(x))
    width = height
    patches = einops.rearrange(patches, 'b (h w) (c ps1 ps2) -> b c (h ps1) (w ps2)', h=height, w=width, ps1=patch_size,
                               ps2=patch_size)
    return patches


class Transformer(nn.Module):
    """Transformer Encoder
    Args:
        embedding_dim: dimension of embedding
        n_heads: number of attention heads
        n_layers: number of attention layers
        feedforward_dim: hidden dimension of MLP layer
    Returns:
        Transformer embedding of input
    """

    def __init__(self, embedding_dim=768, n_heads=12, n_layers=12, feedforward_dim=3072):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feedforward_dim,
                activation=F.gelu,
                batch_first=True,
                dropout=0.1,
                norm_first=True
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim, eps=1e-6)
        )

    def forward(self, x):
        return self.transformer(x)


@BACKBONES.register_module()
class ViT_T(BaseBackbone):
    def __init__(self, embedding_dim=768, n_layers=12, n_heads=12, feedforward_dim=768 * 4, patch_size=4,
                 num_patches=16):
        super(ViT_T, self).__init__()
        self.encoder = Transformer(embedding_dim=embedding_dim, n_layers=n_layers, n_heads=n_heads,
                                   feedforward_dim=feedforward_dim)

    def forward_features(self, images):
        patches = patchify(images)

        projection = self.encoder_input_projection(patches)

        b, n, l = projection.shape
        projection += self.encoder_position_encoding[:, :(n + 1)]
        output = self.encoder(projection)
        return output

    def forward(self, x):
        x = self.forward_features(x)
        return x
