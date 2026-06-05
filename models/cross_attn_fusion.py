import torch
import torch.nn as nn

from typing import Optional


class CrossAttentionFusion(nn.Module):
    """Fuse SAM3 text tokens with external CLIP/SigLIP token embeddings."""

    def __init__(
            self, 
            sam_dim: int = 256, 
            text_dim: int = 768, 
            num_heads: int = 8
        ) -> None:
        """Create the projection, attention, and residual gating layers."""

        super().__init__()
        self.sam_norm = nn.LayerNorm(sam_dim)
        self.text_norm = nn.LayerNorm(sam_dim)
        self.text_proj = nn.Linear(text_dim, sam_dim)
        self.cross_attn = nn.MultiheadAttention(sam_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(sam_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        sam_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return SAM token features after cross-attending to external text tokens."""

        batch_first_sam = sam_tokens.transpose(0, 1)
        projected_text = self.text_proj(text_tokens)
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.bool()
        fused, _ = self.cross_attn(
            query=self.sam_norm(batch_first_sam),
            key=self.text_norm(projected_text),
            value=projected_text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = batch_first_sam + torch.tanh(self.gate) * fused
        return self.out_norm(fused).transpose(0, 1)