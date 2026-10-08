import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Any
from torch import Tensor
from tensordict import TensorDict
from rl4co.envs import RL4COEnvBase


class Adapter(nn.Module):
    """Independent middle layer that sits between encoder and decoder.
    
    This adapter enhances node embeddings by injecting:
    1. Route context from current trajectory
    2. Energy/constraint feasibility information
    3. Dynamic state information
    
    The enhanced embeddings are then used by the decoder for attention and logit computation.
    
    Data flow: Encoder → node_embeddings → Adapter (enhance) → enhanced_embeddings → Decoder
    
    Note: The adapter holds a reference to base_decoder but does NOT register it
    as a submodule. This ensures encoder, adapter, and decoder appear as three
    independent components in the model structure, not nested.
    
    Args:
        base_decoder: Reference to the decoder (not registered as submodule)
        bias_scale: Scaling factor for constraint enhancement
        temperature: Temperature for feasibility computation
        embed_dim: Embedding dimension (default: 128)
    """

    def __init__(
        self, 
        base_decoder: nn.Module, 
        bias_scale: float = 1.0, 
        temperature: float = 1.0, 
        embed_dim: int = 128,
        use_logits_bias: bool = True,
        bias_decay_epochs: Optional[int] = None,
        min_bias_scale: float = 0.0,
        max_bias_epochs: int = 5,  # 只在前 N 个 epoch 中使用 Logits 偏置
        use_learnable_bias: bool = True,  # 是否使用可学习的 MLP 生成 bias
        use_adaptive_bias: bool = False,  # 是否根据 infeasible 比例自适应调整
        adaptive_bias_sensitivity: float = 1.0,  # 自适应调整的敏感度
    ):
        super().__init__()
        # Store decoder reference without registering it as a submodule
        # This prevents decoder from appearing under adapter in model structure
        object.__setattr__(self, '_base_decoder_ref', base_decoder)
        self._env: Optional[RL4COEnvBase] = None
        self.tour_actions: Optional[list] = None
        self.bias_scale = bias_scale
        self.initial_bias_scale = bias_scale  # 保存初始值用于衰减
        temperature = max(temperature, 1e-3)
        self.temperature = temperature
        self.eps = 1e-9
        self.embed_dim = embed_dim
        self.use_logits_bias = use_logits_bias  # 是否使用 Logits 偏置
        self.bias_decay_epochs = bias_decay_epochs  # 衰减周期（None = 不衰减）
        self.min_bias_scale = min_bias_scale  # 最小偏置强度
        self.max_bias_epochs = max_bias_epochs  # 只在前 N 个 epoch 中使用 Logits 偏置
        self.current_epoch = 0  # 当前训练轮次
        
        # 自适应调整相关
        self.use_adaptive_bias = use_adaptive_bias
        self.adaptive_bias_sensitivity = adaptive_bias_sensitivity
        self.infeasible_ratio_history = []  # 记录 infeasible 比例历史
        
        # 可学习的 MLP bias 生成器
        self.use_learnable_bias = use_learnable_bias
        if use_learnable_bias:
            # 输入：feasibility, remaining_energy_ratio, distance_ratio, constraint_flag
            # 输出：bias value
            self.learnable_bias_mlp = nn.Sequential(
                nn.Linear(4, embed_dim // 2),  # 输入特征维度
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 1),  # 输出单个 bias 值
                nn.Tanh()  # 限制输出范围在 [-1, 1]
            )
            
        # 约束特征编码器
        self.constraint_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 4),  # feasibility + constraint_flag
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # 能量可行性编码器（将 feasibility 编码为嵌入维度）
        self.energy_feasibility_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),  # feasibility 维度为 1
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)  # 输出维度为 embed_dim
        )
        
        # Constraint-aware Cross-Attention
        # 让约束信息通过注意力机制动态地增强节点嵌入
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads
        
        # Multi-head attention components
        self.constraint_to_query = nn.Linear(embed_dim, embed_dim)
        self.node_to_key = nn.Linear(embed_dim, embed_dim)
        self.node_to_value = nn.Linear(embed_dim, embed_dim)
        self.attention_out = nn.Linear(embed_dim, embed_dim)
        
        # 嵌入融合层（只融合原始嵌入和注意力特征，路径上下文直接加）
        self.embedding_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),  # original + attended
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 约束投影层（可选，用于 logits 增强）
        self.constraint_proj: Optional[nn.Linear] = None
    
    @property
    def base_decoder(self):
        """Access decoder without it being a registered submodule."""
        return object.__getattribute__(self, '_base_decoder_ref')
    
    def set_epoch(self, epoch: int):
        """Update current epoch and adjust bias_scale dynamically.
        
        This allows the logits bias to decay over training, reducing interference
        with learned strategies in later epochs.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # 动态衰减 bias_scale
        if self.bias_decay_epochs is not None and self.bias_decay_epochs > 0:
            # 线性衰减：从 initial_bias_scale 衰减到 min_bias_scale
            if epoch >= self.bias_decay_epochs:
                # 训练后期：使用最小偏置
                self.bias_scale = self.min_bias_scale
            else:
                # 训练中期：线性衰减
                decay_ratio = epoch / self.bias_decay_epochs
                self.bias_scale = self.initial_bias_scale * (1 - decay_ratio) + self.min_bias_scale * decay_ratio
        # 如果不设置 decay_epochs，则保持初始值不变

    def set_env(self, env: RL4COEnvBase):
        self._env = env

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:
        """Pre-decoder hook - enhance embeddings once before decoding starts."""
        self._env = env
        self._num_starts = num_starts
        
        # Call base decoder's pre_decoder_hook to get the cache
        td, env, hidden = self.base_decoder.pre_decoder_hook(td, env, hidden, num_starts)
        
        # Enhance embeddings ONCE before decoding starts (no POMO expansion yet)
        # The decoder's internal batchify will handle copying enhanced embeddings
        if hidden is not None and hasattr(hidden, "node_embeddings"):
            from rl4co.models.zoo.am.decoder import PrecomputedCache
            
            # Enhance embeddings (td is not yet batchified, so dimensions match)
            enhanced_embeddings = self._enhance_embeddings_static(hidden.node_embeddings, td)
            
            # Update cache with enhanced embeddings
            if isinstance(hidden, PrecomputedCache):
                glimpse_key, glimpse_val, logit_key = \
                    self.base_decoder.project_node_embeddings(enhanced_embeddings).chunk(3, dim=-1)
                
                hidden = PrecomputedCache(
                    node_embeddings=enhanced_embeddings,
                    graph_context=hidden.graph_context,
                    glimpse_key=glimpse_key,
                    glimpse_val=glimpse_val,
                    logit_key=logit_key,
                )
        
        return td, env, hidden

    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> dict:
        """Forward pass combining embedding enhancement and logits bias.
        
        Strategy:
        1. Enhance embeddings (let Decoder "learn" constraints)
        2. Decoder computes logits
        3. Add logits bias (ensure feasibility)
        
        Args:
            td: TensorDict with environment state
            hidden: Hidden state (embeddings may be pre-enhanced in pre_decoder_hook)
            num_starts: Number of starts for multi-start decoding
            
        Returns:
            Dict with keys: 'logits', 'mask', 'pip_aux_loss'
        """
        # Step 1: Enhance embeddings (if not already enhanced or need dynamic update)
        # Note: embeddings are already enhanced in pre_decoder_hook, but we can
        # optionally re-enhance here if we want to use current state (e.g., tour_actions)
        if hidden is not None and hasattr(hidden, "node_embeddings"):
            # Optionally re-enhance with current state (e.g., if tour_actions changed)
            # For now, use pre-enhanced embeddings from pre_decoder_hook
            pass
        
        # Step 2: Decoder computes logits with enhanced embeddings
        decoder_output = self.base_decoder(td, hidden, num_starts)

        # Standardize output format
        if isinstance(decoder_output, dict):
            logits = decoder_output["logits"]
            mask = decoder_output["mask"]
            step_aux_loss = decoder_output.get("pip_aux_loss", torch.tensor(0.0, device=logits.device))
        elif isinstance(decoder_output, tuple) and len(decoder_output) == 2:
            logits, mask = decoder_output
            step_aux_loss = torch.tensor(0.0, device=logits.device)
        else:
            raise ValueError(
                f"Decoder output must be a dict or tuple(logits, mask); got {type(decoder_output)}"
            )
        
        # Step 3: Add logits bias (ensure feasibility) - 可选
        # 只在前 max_bias_epochs 个 epoch 中使用 Logits 偏置
        # 如果 use_logits_bias=False 或 bias_scale=0 或 epoch >= max_bias_epochs，则跳过 Logits 偏置
        use_bias_now = self.use_logits_bias and self.bias_scale > 1e-6 and self.current_epoch < self.max_bias_epochs
        if use_bias_now:
            if mask is None:
                mask = td.get("action_mask", None)
            if mask is None:
                raise ValueError("Adapter requires a valid action mask")
            
            if self._env is None:
                raise RuntimeError("Adapter requires env; call set_env() or pass via pre_decoder_hook")
            
            # Compute energy/constraint bias
            energy_bias, energy_weights = self._compute_energy_bias(td, mask)
            td.set("energy_attention_weights", energy_weights)
            
            # Enhance logits with bias
            enhanced_logits = logits + energy_bias.to(logits.dtype)
        else:
            # 不使用 Logits 偏置，只使用 Embedding 增强
            enhanced_logits = logits
        
        # Return enhanced output
        return {
            "logits": enhanced_logits,
            "mask": mask,
            "pip_aux_loss": step_aux_loss
        }

    def _enhance_embeddings_static(self, node_embeddings: torch.Tensor, td: TensorDict) -> torch.Tensor:
        """Enhance node embeddings with constraint_embedding (through attention), route_context, and energy feasibility.
        
        Args:
            node_embeddings: [batch_size, num_nodes, embed_dim] - original embeddings
            td: TensorDict with initial state (NOT yet batchified by POMO)
            
        Returns:
            enhanced_embeddings: [batch_size, num_nodes, embed_dim] - enhanced embeddings
        """
        batch_size, num_nodes, embed_dim = node_embeddings.shape
        
        # 1. 获取 constraint_embedding（从 encoder 来的，包含约束 flag）
 
        constraint_embedding = td.get("constraint_embedding")
 
              
        # 4. 计算能量可行性（feasibility）
        energy_feasibility = self._compute_energy_feasibility(td)  # [batch, num_nodes]
        
        # 5. 将能量可行性编码为嵌入维度
        energy_feasibility_emb = self.energy_feasibility_encoder(
            energy_feasibility.unsqueeze(-1)  # [batch, num_nodes, 1]
        )  # [batch, num_nodes, embed_dim]
        
        # 6. Multi-head Attention: 使用 constraint_embedding 作为 query（复杂的注意力计算）
        # 对原始节点嵌入做 attention
        attended_features = self._constraint_aware_attention(
            constraint_embedding,  # query: constraint_embedding（包含约束 flag）
            node_embeddings,       # key & value: 原始节点嵌入
            td                     # for masking
        )  # [batch, num_nodes, embed_dim]
        
        # 7. 融合: 原始嵌入 + 注意力增强特征
        combined = torch.cat([
            node_embeddings,       # 原始节点嵌入
            attended_features,     # 注意力增强特征（constraint_embedding 作为 query）
        ], dim=-1)  # [batch, num_nodes, embed_dim * 2]
        
        # 8. 通过融合网络生成中间嵌入
        intermediate_embeddings = self.embedding_fusion(combined)  # [batch, num_nodes, embed_dim]
        
        # 9. 路径上下文和能量可行性直接加到增强嵌入中（简单相加）
        enhanced_embeddings = intermediate_embeddings + energy_feasibility_emb  # [batch, num_nodes, embed_dim]
        
        return enhanced_embeddings
    
    def _constraint_aware_attention(
        self, 
        constraint_embedding: torch.Tensor,  # [batch, num_nodes, embed_dim] - 约束嵌入（包含约束 flag）
        node_embeddings: torch.Tensor,       # [batch, num_nodes, embed_dim] - 原始节点嵌入
        td: TensorDict
    ) -> torch.Tensor:
        """Constraint-aware multi-head attention.
        
        使用 constraint_embedding 作为 query，对原始节点嵌入做 attention。
        这样可以让节点关注约束信息（包含约束 flag），生成增强嵌入。
        
        Args:
            constraint_embedding: 约束嵌入 [batch, num_nodes, embed_dim] - 包含约束 flag
            node_embeddings: 原始节点嵌入 [batch, num_nodes, embed_dim]
            td: TensorDict with state
            
        Returns:
            attended_features: 注意力增强的特征 [batch, num_nodes, embed_dim]
        """
        batch_size, num_nodes, embed_dim = node_embeddings.shape
        
        # 1. 生成 Q, K, V
        Q = self.constraint_to_query(constraint_embedding)  # [batch, num_nodes, embed_dim] - 从 constraint_embedding 生成
        K = self.node_to_key(node_embeddings)               # [batch, num_nodes, embed_dim] - 从原始节点嵌入生成
        V = self.node_to_value(node_embeddings)             # [batch, num_nodes, embed_dim] - 从原始节点嵌入生成
        
        # 2. Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, nodes, head_dim]
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, nodes, nodes]
        
        # 4. 应用注意力
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, heads, nodes, nodes]
        
        # 5. 加权求和
        attended = torch.matmul(attn_weights, V)  # [batch, heads, nodes, head_dim]
        
        # 6. Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, embed_dim)
        
        # 7. 输出投影
        attended_features = self.attention_out(attended)  # [batch, num_nodes, embed_dim]
        
        return attended_features
    
    def _compute_energy_feasibility(self, td: TensorDict) -> torch.Tensor:
        """计算能量可行性（feasibility）用于嵌入增强.
        
        根据当前能量状态动态计算能量可达性，考虑：
        1. 当前剩余能量
        2. 到目标节点的距离
        3. 如果是充电站，考虑充电后的能量
        
        Args:
            td: TensorDict with current state (可能已经解码了几步)
            
        Returns:
            feasibility: [batch, num_nodes] - 能量可行性分数
        """
        batch_size = td["locs"].shape[0]
        num_nodes = td["locs"].shape[1]
        
        # 获取距离信息
        dist_matrix = td.get("dist_matrix")
        if dist_matrix is None:
            # 如果没有距离矩阵，计算欧氏距离
            locs = td["locs"]
            dist_matrix = torch.cdist(locs, locs, p=2)
        
        # 获取当前节点（在解码过程中会动态变化）
        current_node = td.get("current_node", torch.zeros(batch_size, 1, dtype=torch.long, device=td.device))
        current_node = current_node.long().squeeze(-1)
        if current_node.dim() == 0:
            current_node = current_node.unsqueeze(0).expand(batch_size)
        elif current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        
        batch_idx = torch.arange(batch_size, device=dist_matrix.device)
        distances = dist_matrix[batch_idx, current_node]  # [batch, num_nodes]
        
        # 计算能量可行性
        max_energy = float(getattr(self._env, "MAX_ENERGY", 3.0)) if self._env is not None else 3.0
        used_length = td["used_length"]
        
 
        # 当前剩余能量
        remaining_energy = torch.clamp(max_energy - used_length, min=0.0)  # [batch, 1]
        
        # 计算能量可达性：考虑当前能量和距离
        # 如果剩余能量 >= 距离，则可达
        energy_feasible = remaining_energy.expand(-1, num_nodes) >= distances  # [batch, num_nodes]
        
        # 对于充电站，考虑充电后的能量
        # 获取充电站信息（假设节点 1 到 num_station 是充电站）
        num_station = 0
        if hasattr(self._env, 'generator') and hasattr(self._env.generator, 'num_station'):
            num_station = self._env.generator.num_station
        elif "stations" in td:
            num_station = td["stations"].shape[-2] if td["stations"].dim() > 0 else 0
        
        if num_station > 0:
            # 充电站节点索引：1 到 num_station
            station_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=distances.device)
            for i in range(1, min(1 + num_station, num_nodes)):
                station_mask[:, i] = True
            
            # 对于充电站，考虑充电后的能量（假设可以充到满）
            # 到达充电站后，能量可以恢复到 max_energy
            station_energy_after_charge = max_energy  # 充电后能量
            station_feasible = distances <= station_energy_after_charge  # [batch, num_nodes]
            
            # 合并：普通节点用当前能量判断，充电站用充电后能量判断
            energy_feasible = torch.where(
                station_mask,
                station_feasible,
                energy_feasible
            )
        
        # 计算可行性分数：剩余能量 - 距离（越大越好）
        feasibility_raw = remaining_energy.expand(-1, num_nodes) - distances  # [batch, num_nodes]
        
        # 对于充电站，使用充电后的能量计算
        if num_station > 0:
            station_feasibility_raw = torch.full_like(feasibility_raw, max_energy) - distances
            feasibility_raw = torch.where(station_mask, station_feasibility_raw, feasibility_raw)
        
        # 使用 sigmoid 归一化到 [0, 1]
        feasibility = torch.sigmoid(feasibility_raw / (self.temperature + 1e-6))  # [batch, num_nodes]
        
        return feasibility
    
    
    def _compute_node_constraint_features(self, td: TensorDict) -> torch.Tensor:
        """计算每个节点的约束/可行性特征.
        
        Args:
            td: TensorDict with state
            
        Returns:
            constraint_features: [batch, num_nodes, embed_dim]
        """
        # 获取距离信息
        dist_matrix = td["dist_matrix"]
        batch_size = dist_matrix.size(0)
        current_node = td["current_node"].long().squeeze(-1)
        batch_idx = torch.arange(batch_size, device=dist_matrix.device)
        distances = dist_matrix[batch_idx, current_node]  # [batch, num_nodes]
        
        # 计算能量可行性
        max_energy = float(getattr(self._env, "MAX_ENERGY", 3.0)) if self._env is not None else 3.0
        used_length = td["used_length"]
        # 确保 used_length 是正确的形状 [batch, 1]
        if used_length.dim() > 2:
            used_length = used_length.view(batch_size, -1)[:, :1]
        elif used_length.dim() == 1:
            used_length = used_length.unsqueeze(-1)
        elif used_length.size(-1) > 1:
            used_length = used_length[:, :1]
            
        remaining_energy = torch.clamp(max_energy - used_length, min=0.0)
        remaining_energy = remaining_energy.view(batch_size, 1).expand(-1, distances.size(1))
        
        feasibility_raw = remaining_energy - distances
        feasibility = torch.sigmoid(feasibility_raw / (self.temperature + 1e-6))  # [batch, num_nodes]
        
        # 获取约束标志
        constraint_flag = td.get("constraint_energy", torch.ones(batch_size, 1, device=distances.device))
        if constraint_flag.dim() == 0:
            constraint_flag = constraint_flag.view(1, 1).expand(batch_size, distances.size(1))
        elif constraint_flag.dim() == 1:
            constraint_flag = constraint_flag.view(batch_size, 1).expand(-1, distances.size(1))
        else:
            while constraint_flag.dim() > 2:
                constraint_flag = constraint_flag.squeeze(-1)
            if constraint_flag.size(1) == 1:
                constraint_flag = constraint_flag.expand(-1, distances.size(1))
        
        # 合并为特征向量
        constraint_input = torch.stack([feasibility, constraint_flag.float()], dim=-1)  # [batch, num_nodes, 2]
        
        # 编码为约束特征
        constraint_features = self.constraint_encoder(constraint_input)  # [batch, num_nodes, embed_dim]
        
        return constraint_features
    
    def _compute_energy_bias(self, td: TensorDict, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy/constraint-based bias for action logits.
        
        根据当前能量状态动态计算能量距离矩阵，考虑：
        1. 当前剩余能量（每到一个点位会变化）
        2. 到目标节点的距离
        3. 如果是充电站，考虑充电后的能量
        
        This adds a bias to logits based on energy feasibility and constraints.
        Nodes that are more feasible (energy-sufficient) get higher bias.
        
        Args:
            td: TensorDict with current state (在解码过程中会动态更新)
            mask: Action mask [batch, num_nodes]
            
        Returns:
            energy_bias: Bias to add to logits [batch, num_nodes]
            energy_weights: Attention weights for visualization [batch, num_nodes]
        """
        if mask is None or not isinstance(mask, torch.Tensor):
            raise ValueError("Adapter requires a valid torch.Tensor mask")
        mask_bool = mask.bool()
        batch_size, num_nodes = mask_bool.shape

        dist_matrix = td["dist_matrix"]
        current_node = td["current_node"].long().squeeze(-1)
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        batch_idx = torch.arange(batch_size, device=dist_matrix.device)
        distances = dist_matrix[batch_idx, current_node]  # [batch, num_nodes]

        # 计算能量可行性（动态更新）
        max_energy = float(getattr(self._env, "MAX_ENERGY", 3.0)) if self._env is not None else 3.0
        used_length = td["used_length"]
        # 确保 used_length 是正确的形状 [batch, 1]
        if used_length.dim() > 2:
            used_length = used_length.view(batch_size, -1)[:, :1]
        elif used_length.dim() == 1:
            used_length = used_length.unsqueeze(-1)
        elif used_length.size(-1) > 1:
            used_length = used_length[:, :1]
            
        # 当前剩余能量（动态变化）
        remaining_energy = torch.clamp(max_energy - used_length, min=0.0)  # [batch, 1]
        
        # 获取充电站信息
        num_station = 0
        if hasattr(self._env, 'generator') and hasattr(self._env.generator, 'num_station'):
            num_station = self._env.generator.num_station
        elif "stations" in td:
            num_station = td["stations"].shape[-2] if td["stations"].dim() > 0 else 0
        
        # 计算能量可行性分数
        # 对于普通节点：剩余能量 - 距离
        feasibility_raw = remaining_energy.expand(-1, num_nodes) - distances  # [batch, num_nodes]
        
        # 对于充电站，考虑充电后的能量
        if num_station > 0:
            # 充电站节点索引：1 到 num_station
            station_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=distances.device)
            for i in range(1, min(1 + num_station, num_nodes)):
                station_mask[:, i] = True
            
            # 对于充电站，使用充电后的能量计算（假设可以充到满）
            station_feasibility_raw = torch.full_like(feasibility_raw, max_energy) - distances
            feasibility_raw = torch.where(station_mask, station_feasibility_raw, feasibility_raw)
        
        feasibility = torch.sigmoid(feasibility_raw / (self.temperature + 1e-6))  # [batch, num_nodes]

        # 获取约束标志
        flag = td.get("constraint_energy")
        if flag is None:
            constraint_energy = torch.zeros_like(mask_bool, dtype=feasibility.dtype)
        else:
            flag = flag.to(dtype=feasibility.dtype, device=feasibility.device)
            if flag.dim() == 0:
                constraint_energy = flag.view(1, 1).expand(batch_size, num_nodes)
            elif flag.dim() == 1:
                constraint_energy = flag.view(batch_size, 1).expand(-1, num_nodes)
            elif flag.dim() >= 2:
                base = flag
                while base.dim() > 2:
                    base = base.squeeze(-1)
                if base.shape == (batch_size, 1):
                    constraint_energy = base.expand(-1, num_nodes)
                elif base.shape == (batch_size, num_nodes):
                    constraint_energy = base
                else:
                    while flag.dim() < mask_bool.dim():
                        flag = flag.unsqueeze(-1)
                    constraint_energy = flag.expand_as(mask_bool).to(dtype=feasibility.dtype)
            else:
                constraint_energy = torch.zeros_like(mask_bool, dtype=feasibility.dtype)
        constraint_energy = torch.where(constraint_energy > 0.5, 1.0, 0.0)

        # 计算能量偏置
        if self.use_learnable_bias:
            # 使用可学习的 MLP 生成 bias
            # 准备输入特征：[feasibility, remaining_energy_ratio, distance_ratio, constraint_flag]
            remaining_energy_ratio = (remaining_energy / max_energy).expand(-1, num_nodes)  # [batch, num_nodes]
            
            # 计算距离比例（归一化到 [0, 1]）
            max_distance = distances.max(dim=-1, keepdim=True)[0]  # [batch, 1]
            distance_ratio = distances / (max_distance + 1e-6)  # [batch, num_nodes]
            
            # 约束标志（已经是 [batch, num_nodes]）
            constraint_flag = constraint_energy.float()  # [batch, num_nodes]
            
            # 组合特征：[batch, num_nodes, 4]
            bias_features = torch.stack([
                feasibility,                    # 能量可行性
                remaining_energy_ratio,         # 剩余能量比例
                distance_ratio,                 # 距离比例
                constraint_flag                 # 约束标志
            ], dim=-1)  # [batch, num_nodes, 4]
            
            # 使用 MLP 生成 bias
            # MLP 输出形状: [batch, num_nodes, 1]
            bias_raw = self.learnable_bias_mlp(bias_features)  # [batch, num_nodes, 1]
            bias_raw = bias_raw.squeeze(-1)  # [batch, num_nodes]
            
            # 应用 bias_scale 进行缩放
            energy_bias = bias_raw * self.bias_scale
        else:
            # 使用原始的规则函数
            energy_bias = feasibility * self.bias_scale
        
        # 应用 mask
        energy_bias = energy_bias.masked_fill(~mask_bool, 0.0)
        
        # 自适应调整：根据 infeasible 比例动态调整 bias_scale
        if self.use_adaptive_bias and self.training:
            # 计算当前 infeasible 比例
            infeasible_mask = ~mask_bool  # 被 mask 掉的节点是不可行的
            infeasible_ratio = infeasible_mask.float().mean().item()
            
            # 记录历史
            self.infeasible_ratio_history.append(infeasible_ratio)
            
            # 如果 infeasible 比例高，增加 bias_scale
            # 如果 infeasible 比例低，减少 bias_scale
            if len(self.infeasible_ratio_history) > 10:
                # 使用最近的平均值
                recent_ratio = sum(self.infeasible_ratio_history[-10:]) / 10
                # 调整 bias_scale：infeasible 比例越高，bias_scale 越大
                adjustment = 1.0 + (recent_ratio - 0.5) * self.adaptive_bias_sensitivity
                self.bias_scale = self.initial_bias_scale * adjustment
                # 限制在合理范围内
                self.bias_scale = torch.clamp(
                    torch.tensor(self.bias_scale),
                    min=self.min_bias_scale,
                    max=self.initial_bias_scale * 2.0
                ).item()

        # 计算注意力权重（用于可视化）
        attention_proxy = torch.clamp(feasibility, min=self.eps)
        attention_proxy = attention_proxy.masked_fill(~mask_bool, 0.0)
        return energy_bias, attention_proxy
