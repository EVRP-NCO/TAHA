from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
# from rl4co.models.common.constructive import AutoregressiveEncoder # Keep base class import if needed
from rl4co.models.common.constructive.base import ConstructiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork # Correct import
# Remove incorrect/unused imports
# from rl4co.models.nn.attention import MultiHeadAttention, SparseMultiHeadAttention
# from rl4co.models.nn.graph import GraphAttentionEncoder
# from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AttentionModelEncoder(ConstructiveEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.
    Modified to support separate complete and constraint embeddings for EVRPTW.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
        collect_attention: bool = False,  # 添加注意力采集控制参数
    ):
        super(AttentionModelEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.collect_attention = collect_attention  # 保存注意力采集设置


        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )
        
        self.net = (
            GraphAttentionNetwork(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_layers,
                normalization=normalization, 
                feedforward_hidden=feedforward_hidden,
                attention_type="heterogeneous",#mha/heterogeneous  # 切换到MHA进行对比
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None, return_attention: bool = False
    ) -> tuple:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention
            return_attention: Whether to return attention weights for visualization

        Returns:
            Tuple containing:
                - h: Latent representation
                - init_h: Initial embedding
                - attention_weights (optional): List of attention weights
        """
        init_h = self.init_embedding(td)

        # Handle different embedding return formats
        constraint_embed = None
        distance_matrix = None
        if isinstance(init_h, tuple):
            if len(init_h) == 3:
                # Unified VRP returns (complete_embedding, constraint_embedding, distance_matrix)
                node_embed, constraint_embed, distance_matrix = init_h
            elif len(init_h) == 2:
                # EVRPTW returns (complete_embedding, constraint_embedding)
                node_embed, constraint_embed = init_h
            else:
                raise ValueError(
                    "Unexpected init embedding tuple length. Expected 2 or 3 elements."
                )
        else:
            # For other problems that return single embedding
            node_embed = init_h

        # Set constraint_embedding and distance_matrix to td if available
        if constraint_embed is not None:
            td.set("constraint_embedding", constraint_embed)
        if distance_matrix is not None:
            td.set("init_distance_matrix", distance_matrix)

        if return_attention:
            h, attention_weights = self.net(
                node_embed,
                mask=mask,
                distance_matrix=distance_matrix,
                return_attention=True,
            )
            return h, init_h, attention_weights

        h = self.net(
            node_embed,
            mask=mask,
            distance_matrix=distance_matrix,
            return_attention=False,
        )
        return h, init_h
