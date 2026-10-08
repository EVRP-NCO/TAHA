from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.models.nn.attention import (
    MultiHeadAttention,
    ChannelEnhancedContextualAttention,
    PrefixTunedMultiHeadAttention,
    KNNLocalAttention,
    HeterogeneousAttention,
)
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger
from typing import Callable, Optional

import torch
import torch.nn as nn

from torch import Tensor

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        bias: whether to use bias in MHA linear layers
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE FFN
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
        self.skip_mha = SkipConnection(self.mha)
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, return_attention: bool = False):
        """ Forward pass for the MHA layer.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_dim).
            attn_mask (Optional[Tensor]): Attention mask passed to MHA. Shape (batch, seq_len, seq_len) or broadcastable.
            return_attention (bool): Whether to return attention weights.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output tensor of shape (batch, seq_len, embed_dim), and optionally attention weights.
        """
        
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)

        # Call MHA and handle its output
        mha_output = self.mha(x, attn_mask=attn_mask, return_attention=return_attention)
        
        attn_weights = None
        if return_attention:
            mha_output, attn_weights = mha_output

        # Manually apply skip connection
        x_residual = x + mha_output
        
        # Apply normalization and FFN
        x = self.norm1(x_residual)
        x = self.norm2(self.ffn(x))
        
        if return_attention:
            return x, attn_weights
        return x

class PrefixTunedMultiHeadAttentionLayer(nn.Module):
    """PrefixTunedMultiHeadAttention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        bias: whether to use bias in MHA linear layers
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE FFN
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.mha = PrefixTunedMultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
        self.skip_mha = SkipConnection(self.mha)
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, return_attention: bool = False):
        """ Forward pass for the MHA layer.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_dim).
            attn_mask (Optional[Tensor]): Attention mask passed to MHA. Shape (batch, seq_len, seq_len) or broadcastable.
            return_attention (bool): Whether to return attention weights.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output tensor of shape (batch, seq_len, embed_dim), and optionally attention weights.
        """
        
        # Call MHA and handle its output
        mha_output = self.mha(x, attn_mask=attn_mask, return_attention=return_attention)
        
        attn_weights = None
        if return_attention:
            mha_output, attn_weights = mha_output

        # Manually apply skip connection
        x_residual = x + mha_output
        
        # Apply normalization and FFN
        x = self.norm1(x_residual)
        x = self.norm2(self.ffn(x))
        
        if return_attention:
            return x, attn_weights
        return x
class ChannelEnhancedContextualAttentionLayer(nn.Module):
    """
    Layer combining Channel Enhanced Contextual Self-Attention with normalization and FFN.
    Mirrors the structure of MultiHeadAttentionLayer.

    Args:
        embed_dim: dimension of the embeddings (dk)
        num_heads: number of attention heads
        channels: number of channels (c) for the internal Conv2d/ChannelAttention
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        moe_kwargs: Keyword arguments for MoE in FFN
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int=8,
        channels: int=4,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.ceca = SkipConnection(
            ChannelEnhancedContextualAttention(embed_dim, num_heads, channels)
        )
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, kv: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """ Forward pass for the CECA layer.

        Args:
            x (Tensor): Input tensor (B, N, D) or (B, 4, N, D)
            kv (Optional[Tensor]): Key/Value input tensor. If None, uses x.
            attn_mask (Optional[Tensor])
        """
        # If input still has 4 constraint channels, fuse them first so that
        # residual shapes match inside SkipConnection
        # if x.dim() == 4:
        #     # Use the same fusion logic as ChannelEnhancedContextualAttention
        #     ceca_module = self.ceca.module  # Access wrapped module
        #     x = ceca_module._process_multi_channel_input(x)  # (B,N,D)

        # Apply CECA with skip connection, then normalization
        x = self.norm1(self.ceca(x, kv=kv, attn_mask=attn_mask))
        # Apply FFN with skip connection, then normalization
        x = self.norm2(self.ffn(x))
        return x


class KNNLocalAttentionLayer(nn.Module):
    """KNN-based Local Attention layer with normalization and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        k_neighbors: Optional[int] = None,
        k_ratio: Optional[float] = 1 / 3,
        include_self: bool = True,
        distance_bias_on_logits: bool = False,
        distance_bias: bool = False,
        distance_bias_dropout: float = 0.0,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        moe_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.knn_attn = KNNLocalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(
        self,
        x: Tensor,
        distance_matrix: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ):
        if distance_matrix is None:
            raise ValueError("distance_matrix must be provided for KNNLocalAttentionLayer")

        attn_output = self.knn_attn(
            x,
            distance_matrix=distance_matrix,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )

        attn_weights = None
        if return_attention:
            attn_output, attn_weights = attn_output

        x = self.norm1(x + attn_output)
        x = self.norm2(self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x


class HeterogeneousAttentionLayer(nn.Module):
    """
    Heterogeneous Attention Layer with normalization and feed-forward layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.het_attn = HeterogeneousAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons,lora_r=4, lora_alpha=1.0, lora_dropout=0.0, **moe_kwargs)
        else:
            ffn_module = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
            )
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x, attn_mask=None, split_sizes=None):
        # x can be tuple (d, s, c) or tensor
        if isinstance(x, tuple):
            x_cat = torch.cat(x, dim=1)
            residual = x_cat
        else:
            residual = x
            
        h = self.het_attn(x, split_sizes=split_sizes, attn_mask=attn_mask)
        
        h = residual + h
        h = self.norm1(h)
        h = self.norm2(self.ffn(h))
        return h


class GraphAttentionNetwork(nn.Module):
    """Stacked attention encoder supporting multiple attention primitives.

    Args:
        num_heads: number of heads (used when the attention type relies on heads)
        embed_dim: embedding dimensionality
        num_layers: number of stacked attention blocks
        normalization: normalization strategy inside each block
        feedforward_hidden: hidden dimension of the FFN sub-layer
        attention_type: 'mha', 'ceca', 'prefix_mha', or 'knn_local'
        ceca_channels: number of channels for CECA (required when attention_type='ceca')
        knn_k: fixed number of neighbors for KNN attention (optional when ratio provided)
        knn_ratio: ratio relative to sequence length to derive neighbor count
        knn_include_self: whether KNN masks always keep self connections
        knn_distance_bias_on_logits: add distance-derived bias to logits before masking
        knn_distance_bias: enable adding distance-derived bias after attention
        sdpa_fn/moe_kwargs: forwarded to the respective submodules when applicable
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        attention_type: str = "ceca",
        ceca_channels: int = 4,
        knn_k: Optional[int] = None,
        knn_ratio: Optional[float] = 1 / 3,
        knn_include_self: bool = True,
        knn_distance_bias_on_logits: bool = False,
        knn_distance_bias: bool = False,
        knn_distance_bias_dropout: float = 0.0,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            moe_tag = dict(moe_kwargs or {})
            if attention_type == "mha":
                layer_cls = MultiHeadAttentionLayer
                layer_kwargs = {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "feedforward_hidden": feedforward_hidden,
                    "normalization": normalization,
                    "sdpa_fn": sdpa_fn,
                    "moe_kwargs": moe_tag,
                }
            elif attention_type == "heterogeneous":
                layer_cls = HeterogeneousAttentionLayer
                layer_kwargs = {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "feedforward_hidden": feedforward_hidden,
                    "normalization": normalization,
                    "sdpa_fn": sdpa_fn,
                    "moe_kwargs": moe_tag,
                }
            elif attention_type == "ceca":
                layer_cls = ChannelEnhancedContextualAttentionLayer
                layer_kwargs = {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "channels": ceca_channels,
                    "feedforward_hidden": feedforward_hidden,
                    "normalization": normalization,
                    "moe_kwargs": moe_tag,
                }
            elif attention_type == "prefix_mha":
                layer_cls = PrefixTunedMultiHeadAttentionLayer
                layer_kwargs = {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "feedforward_hidden": feedforward_hidden,
                    "normalization": normalization,
                    "sdpa_fn": sdpa_fn,
                    "moe_kwargs": moe_tag,
                }
            elif attention_type == "knn_local":
                layer_cls = KNNLocalAttentionLayer
                layer_kwargs = {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "k_neighbors": knn_k,
                    "k_ratio": knn_ratio,
                    "include_self": knn_include_self,
                    "distance_bias_on_logits": knn_distance_bias_on_logits,
                    "distance_bias": knn_distance_bias,
                    "distance_bias_dropout": knn_distance_bias_dropout,
                    "feedforward_hidden": feedforward_hidden,
                    "normalization": normalization,
                    "moe_kwargs": moe_tag,
                }
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            self.layers.append(layer_cls(**layer_kwargs))

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        distance_matrix: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        
        split_sizes = None
        if isinstance(x, tuple):
            # Assuming x is (depot, station, customer)
            split_sizes = [t.size(1) for t in x]

        if return_attention:
            # We only return the attention weights of the last layer
            # Note: this is a bit hacky, we should probably return all attention weights
            # or let the user specify which layer to return
            # Also, we only support MHA for now
            for layer in self.layers[:-1]:
                if isinstance(layer, KNNLocalAttentionLayer):
                    x = layer(x, distance_matrix, attn_mask=mask)
                elif isinstance(layer, HeterogeneousAttentionLayer):
                    x = layer(x, attn_mask=mask, split_sizes=split_sizes)
                elif isinstance(layer, ChannelEnhancedContextualAttentionLayer):
                    x = layer(x, kv=kv, attn_mask=mask)
                else:
                    x = layer(x, attn_mask=mask)
            
            layer = self.layers[-1]
            if isinstance(layer, KNNLocalAttentionLayer):
                x, attn = layer(x, distance_matrix, attn_mask=mask, return_attention=True)
            elif isinstance(layer, HeterogeneousAttentionLayer):
                # HeterogeneousAttention doesn't support return_attention yet
                x = layer(x, attn_mask=mask, split_sizes=split_sizes)
                attn = None
            elif isinstance(layer, ChannelEnhancedContextualAttentionLayer):
                # CECA doesn't support return_attention yet
                x = layer(x, kv=kv, attn_mask=mask)
                attn = None
            else:
                x, attn = layer(x, attn_mask=mask, return_attention=True)
            return x, attn

        for layer in self.layers:
            if isinstance(layer, KNNLocalAttentionLayer):
                x = layer(x, distance_matrix, attn_mask=mask)
            elif isinstance(layer, HeterogeneousAttentionLayer):
                x = layer(x, attn_mask=mask, split_sizes=split_sizes)
            elif isinstance(layer, ChannelEnhancedContextualAttentionLayer):
                x = layer(x, kv=kv, attn_mask=mask)
            else:
                x = layer(x, attn_mask=mask)
        return x
