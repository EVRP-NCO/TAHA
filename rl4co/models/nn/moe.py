import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------------------------------------------------------
# Part 1: LoRA 专家 (轻量化核心)
# -------------------------------------------------------------------------

class LoRAExpert(nn.Module):
    """
    一个 LoRA 专家就是一对低秩矩阵 (A, B)。
    它不包含原始权重，专门用于学习“增量知识”。
    """
    def __init__(self, in_dim, out_dim, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        
        # LoRA 的精髓：A 降维，B 升维
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        # A 使用 Kaiming 初始化保证信号强度
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B 使用零初始化：保证初始状态下，LoRA专家输出为0 (不干扰Base模型)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # x: [Batch, In]
        # Result: (x @ A @ B) * scale
        x = self.lora_A(self.dropout(x))
        x = self.lora_B(x)
        return x * self.scaling


# -------------------------------------------------------------------------
# Part 2: 鲁棒的 Switch Router (无需 Loss)
# -------------------------------------------------------------------------

class RobustSwitchRouter(nn.Module):
    """
    专为 No-Loss 设计的 Top-1 Router。
    依赖 Input Norm 和 Jitter Noise 维持自然平衡。
    """
    def __init__(self, input_dim, num_experts):
        super().__init__()
        # [关键] 路由前的归一化，防止向量模长影响决策
        self.norm = nn.LayerNorm(input_dim)
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(self, x, training=True):
        x_norm = self.norm(x)
        logits = self.gate(x_norm)
        
        # [平衡机制] 在训练时注入高斯噪声 (Jitter)
        # 这在 Switch Transformer 原文中被证明对 Top-1 路由至关重要
        # 足够大的噪声让模型被迫探索所有专家，从而不需要 Aux Loss
        if training:
            logits = logits + torch.randn_like(logits) * 0.01 
        
        probs = F.softmax(logits, dim=-1)
        
        # Switch Core: 只选 Top-1
        top1_val, top1_idx = torch.max(probs, dim=-1)
        
        return top1_val, top1_idx, probs


# -------------------------------------------------------------------------
# Part 3: 主架构 (Base + Switch LoRA)
# -------------------------------------------------------------------------

class MoE(nn.Module):
    """
    Decoupled-Synergistic MoE (DS MoE) — Switch-LoRA architecture (loss-free).
    
    Logic: 
    y = Base(x) + Switch(LoRA_1...LoRA_N)(x)
    
    无需任何 load balancing loss，因为：
    1. Base 层 (Dense) 处理了所有通用逻辑。
    2. Experts 初始化为 0，起步互不干扰。
    3. Router 的 Noise 和 Norm 保证了自然分布。
    """
    def __init__(self, input_size, output_size, num_neurons=None, hidden_act="ReLU", out_bias=True, 
                 num_experts=8, k=1, noisy_gating=True, **kwargs):
        super(MoE, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        # 强制 Top-1
        self.k = 1 
        
        # LoRA Config
        self.lora_r = kwargs.get("lora_r", 8)  # 默认 r=8
        self.lora_alpha = kwargs.get("lora_alpha", 16.0)
        self.lora_dropout = kwargs.get("lora_dropout", 0.05)

        # 1. Base Layer (Shared Backbone)
        # 这是一个普通的线性层/MLP，所有数据都经过这里
        # 它保证了模型的“下限”，即使 LoRA 专家乱选，输出也不会崩坏。
        self.base_layer = nn.Linear(input_size, output_size, bias=out_bias)

        # 2. Router
        self.router = RobustSwitchRouter(input_size, num_experts)

        # 3. LoRA Experts
        self.experts = nn.ModuleList([
            LoRAExpert(
                input_size, output_size, 
                r=self.lora_r, 
                alpha=self.lora_alpha, 
                dropout=self.lora_dropout
            )
            for _ in range(num_experts)
        ])

    def forward(self, x, **kwargs): # 接口干净，不接收 loss_coef
        # x: [Batch, Seq_len, Dim]
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_size)
        
        # --- A. Base Path Calculation ---
        # 所有 token 都享用的通用计算
        base_out = self.base_layer(x_flat)
        
        # --- B. Switch LoRA Routing ---
        top1_val, top1_idx, probs = self.router(x_flat, training=self.training)
        
        # --- C. Sparse Computation ---
        # 初始化 LoRA 的输出缓冲 (和输入 dtype 一致)
        lora_out = torch.zeros_like(base_out)
        
        # 遍历所有专家 (标准 Switch 做法)
        for i, expert in enumerate(self.experts):
            # 1. 找出谁选了专家 i
            indices = (top1_idx == i).nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                continue
            
            # 2. 取出这部分数据
            inp_subset = x_flat[indices]
            
            # 3. 专家计算 (低秩矩阵运算，极快)
            expert_result = expert(inp_subset)
            
            # 4. 路由权重加权
            # y = expert(x) * prob
            # unsqueeze(1) 用于广播: [N] * [N, 1]
            scaling = top1_val[indices].unsqueeze(1).to(expert_result.dtype)
            weighted_res = expert_result * scaling
            
            # 5. 填回缓冲区 (加上类型转换保险，防止 float16/32 冲突)
            lora_out[indices] = weighted_res.to(lora_out.dtype)
        
        # --- D. Final Fusion ---
        # 最终结果 = 基础能力 + 个性化微调
        final_output = base_out + lora_out
        
        return final_output.reshape(original_shape[:-1] + (self.output_size,))