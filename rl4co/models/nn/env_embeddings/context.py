import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.utils.ops import gather_by_index


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPContext,
        "atsp": TSPContext,
        "cvrp": VRPContext,
        "evrp": EVRPContext,
        "evrptw": EVRPTWContext,
        "cvrptw": VRPTWContext,
        "ffsp": FFSPContext,
        "svrp": SVRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "spctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "mdpp": DPPContext,
        "pdp": PDPContext,
        "mtsp": MTSPContext,
        "smtwtp": SMTWTPContext,
        "mdcpdp": MDCPDPContext,
        "mtvrp": MTVRPContext,
        "unified_vrp": EVRPTWContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class FFSPContext(EnvContext):
    def __init__(self, embed_dim, stage_cnt=None):
        self.has_stage_emb = stage_cnt is not None
        step_context_dim = (1 + int(self.has_stage_emb)) * embed_dim
        super().__init__(embed_dim=embed_dim, step_context_dim=step_context_dim)
        if self.has_stage_emb:
            self.stage_emb = nn.Parameter(torch.rand(stage_cnt, embed_dim))

    def _cur_node_embedding(self, embeddings: TensorDict, td):
        cur_node_embedding = gather_by_index(
            embeddings["machine_embeddings"], td["stage_machine_idx"]
        )
        return cur_node_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        if self.has_stage_emb:
            state_embedding = self._state_embedding(embeddings, td)
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
            return self.project_context(context_embedding)
        else:
            return self.project_context(cur_node_embedding)

    def _state_embedding(self, _, td):
        cur_stage_emb = self.stage_emb[td["stage_idx"]]
        return cur_stage_emb


class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(TSPContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding


class EVRPContext(EnvContext):
    def __init__(self, embed_dim):
        super(EVRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        cap = 1 - td["used_capacity"]
        # base_pomo
        leng = 3.0- td["used_length"]
        # num_station=td["locs"].size(-2)-td["demand"].size(-1)-1
        # size = (
        #     td["locs"].size(-2)
        #     - td["visited"].sum(-1, keepdim=True)
        #     - (
        #         (td["current_node"] < num_station) & (td["current_node"] > 0)
        #     ).float()
        #     * (num_station)
        #     - (td["current_node"] == 0).float() * (num_station + 1)
        # ) / embeddings.size(-2)
        # state_embedding = torch.cat((cap, leng,size), dim=-1)
        state_embedding = torch.cat((cap, leng), dim=-1)
        return state_embedding


class EVRPTWContext(EnvContext):
    def __init__(self, embed_dim):
        super(EVRPTWContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )

    def _state_embedding(self, embeddings, td):
        cap = 1 - td["used_capacity"]
        leng = 3.0 - td["used_length"]
        # num_station=td["locs"].size(-2)-td["demand"].size(-1)-1
        # size = (
        #     td["locs"].size(-2)-num_station
        #     - td["visited"].sum(-1, keepdim=True)
        #     - (td["current_node"] == 0).float()
        # ) / (embeddings.size(-2)-num_station)
        # state_embedding = torch.cat((leng, cap, td["current_time"],size), dim=-1)
        state_embedding = torch.cat((leng, cap, td["current_time"]), dim=-1)#
        return state_embedding


class UnifiedVRPContext(EnvContext):
    def __init__(self, embed_dim):
        super(UnifiedVRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        cap = 1 - td["used_capacity"]
        current_time = td["current_time"]

        # If current_time has far more elements than cap, it's likely misshapen.
        # This can happen if it's being incorrectly broadcasted in the env.
        # As a defensive measure, we take the mean to restore it to the expected shape.
        if current_time.numel() > cap.numel() and cap.numel() > 0:
            # We expect current_time to be [batch, pomo], same as cap
            expected_shape = cap.shape
            # Let's average the extra dimensions
            while current_time.dim() > len(expected_shape):
                current_time = current_time.mean(dim=-1)
            # If it's still not right, reshape
            if current_time.shape != expected_shape:
                current_time = current_time.reshape(expected_shape)

        # Reshape current_time to match cap's dimensions before unsqueezing
        if current_time.dim() < cap.dim():
            current_time = current_time.unsqueeze(-1).expand_as(cap)
        
        state_embedding = torch.cat((cap.unsqueeze(-1), current_time.unsqueeze(-1)), dim=-1)
        
        # Unsqueeze to match cur_node_embedding shape (batch, num_starts, embed_dim)
        if state_embedding.dim() == 2 and embeddings.dim() == 3:
            state_embedding = state_embedding.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            
        return state_embedding


class VRPTWContext(VRPContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
        - current time
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        capacity = super()._state_embedding(embeddings, td)
        current_time = td["current_time"]
        return torch.cat([capacity, current_time], -1)


class SVRPContext(EnvContext):
    """Context embedding for the Skill Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - current technician
    """

    def __init__(self, embed_dim):
        super(SVRPContext, self).__init__(embed_dim=embed_dim, step_context_dim=embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class PCTSPContext(EnvContext):
    """Context embedding for the Prize Collecting TSP (PCTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining prize (prize_required - cur_total_prize)
    """

    def __init__(self, embed_dim):
        super(PCTSPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = torch.clamp(
            td["prize_required"] - td["cur_total_prize"], min=0
        )[..., None]
        return state_embedding


class OPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embed_dim):
        super(OPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["max_length"][..., 0] - td["tour_length"]
        return state_embedding[..., None]


class DPPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embed_dim):
        super(DPPContext, self).__init__(embed_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class PDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(PDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embed_dim, 2 * embed_dim)
        proj_in_dim = (
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)


class SMTWTPContext(EnvContext):
    """Context embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Project the following to the embedding space:
        - current node embedding
        - current time
    """

    def __init__(self, embed_dim):
        super(SMTWTPContext, self).__init__(embed_dim, embed_dim + 1)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_job"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["current_time"]
        return state_embedding


class MDCPDPContext(EnvContext):
    """Context embedding for the MDCPDP.
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(MDCPDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class SchedulingContext(nn.Module):
    def __init__(self, embed_dim: int, scaling_factor: int = 1000):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.proj_busy = nn.Linear(1, embed_dim, bias=False)

    def forward(self, h, td):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        busy_proj = self.proj_busy(busy_for.unsqueeze(-1))
        # (b m e)
        return h + busy_proj


class MTVRPContext(VRPContext):
    """Context embedding for Multi-Task VRPEnv.
    Project the following to the embedding space:
        - current node embedding
        - remaining_linehaul_capacity (vehicle_capacity - used_capacity_linehaul)
        - remaining_backhaul_capacity (vehicle_capacity - used_capacity_backhaul)
        - current time
        - current_route_length
        - open route indicator
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 5
        )

    def _state_embedding(self, embeddings, td):
        remaining_linehaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]
        current_route_length = td["current_route_length"]
        open_route = td["open_route"]
        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                current_route_length,
                open_route,
            ],
            -1,
        )


class ConstraintAwareEVRPTWContext(EnvContext):
    """Constraint-aware context embedding for EVRPTW environment.
    Dynamically adapts to active constraints by masking irrelevant features.
    """
    
    def __init__(self, embed_dim):
        super(ConstraintAwareEVRPTWContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        
        # Learnable constraint masks
        self.energy_mask_proj = nn.Linear(1, embed_dim, bias=False)
        self.time_mask_proj = nn.Linear(1, embed_dim, bias=False)
        
        # Constraint-aware projection
        self.constraint_aware_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        
    def _state_embedding(self, embeddings, td):
        # Get base state features
        cap = 1 - td["used_capacity"]
        leng = 3.0 - td["used_length"]
        current_time = td["current_time"]
        
        # Extract constraint flags
        constraint_energy = td.get("constraint_energy", torch.ones_like(td["current_node"], dtype=torch.bool))
        constraint_time_windows = td.get("constraint_time_windows", torch.ones_like(td["current_node"], dtype=torch.bool))
        
        batch_size = cap.shape[0]
        
        # Apply constraint masks to features
        # When energy constraint is disabled, mask energy-related features
        if not constraint_energy.any():
            leng = torch.zeros_like(leng)  # Mask energy-related features
        
        # When time window constraint is disabled, mask time-related features
        if not constraint_time_windows.any():
            current_time = torch.zeros_like(current_time)  # Mask time-related features
        
        state_embedding = torch.cat((leng, cap, current_time), dim=-1)
        
        # Add constraint information to the embedding
        energy_mask = constraint_energy.float().unsqueeze(-1)
        time_mask = constraint_time_windows.float().unsqueeze(-1)
        
        energy_mask_emb = self.energy_mask_proj(energy_mask)
        time_mask_emb = self.time_mask_proj(time_mask)
        
        constraint_info = energy_mask_emb + time_mask_emb
        
        # If state_embedding needs to be expanded for multi-start decoding
        if state_embedding.dim() == 2 and embeddings.dim() == 3:
            state_embedding = state_embedding.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            constraint_info = constraint_info.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        
        return state_embedding, constraint_info
        
    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding, constraint_info = self._state_embedding(embeddings, td)
        
        # Combine node embedding with state embedding
        base_context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        base_context = self.project_context(base_context_embedding)
        
        # Combine with constraint information
        combined_context = torch.cat([base_context, constraint_info], dim=-1)
        adapted_context = self.constraint_aware_proj(combined_context)
        
        return adapted_context


class ConstraintAwareUnifiedVRPContext(EnvContext):
    """Constraint-aware context embedding for Unified VRP environment.
    Handles energy, time window, and backhaul constraints dynamically.
    """
    
    def __init__(self, embed_dim):
        super(ConstraintAwareUnifiedVRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )
        
        # Learnable constraint masks
        self.energy_mask_proj = nn.Linear(1, embed_dim, bias=False)
        self.time_mask_proj = nn.Linear(1, embed_dim, bias=False)
        self.backhaul_mask_proj = nn.Linear(1, embed_dim, bias=False)
        
        # Constraint-aware projection
        self.constraint_aware_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        
    def _state_embedding(self, embeddings, td):
        cap = 1 - td["used_capacity"]
        current_time = td["current_time"]
        
        # Extract constraint flags
        constraint_energy = td.get("constraint_energy", torch.ones_like(td["current_node"], dtype=torch.bool))
        constraint_time_windows = td.get("constraint_time_windows", torch.ones_like(td["current_node"], dtype=torch.bool))
        constraint_backhaul = td.get("constraint_backhaul", torch.zeros_like(td["current_node"], dtype=torch.bool))
        
        # Apply constraint masks to features
        # When energy constraint is disabled, we don't need to mask cap since it's capacity-related
        # When time window constraint is disabled, mask time-related features
        if not constraint_time_windows.any():
            current_time = torch.zeros_like(current_time)
        
        state_embedding = torch.cat((cap.unsqueeze(-1), current_time.unsqueeze(-1)), dim=-1)
        
        # Create constraint masks
        energy_mask = constraint_energy.float().unsqueeze(-1)
        time_mask = constraint_time_windows.float().unsqueeze(-1)
        backhaul_mask = constraint_backhaul.float().unsqueeze(-1)
        
        energy_mask_emb = self.energy_mask_proj(energy_mask)
        time_mask_emb = self.time_mask_proj(time_mask)
        backhaul_mask_emb = self.backhaul_mask_proj(backhaul_mask)
        
        constraint_info = energy_mask_emb + time_mask_emb + backhaul_mask_emb
        
        # Unsqueeze to match cur_node_embedding shape (batch, num_starts, embed_dim)
        if state_embedding.dim() == 2 and embeddings.dim() == 3:
            state_embedding = state_embedding.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            constraint_info = constraint_info.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            
        return state_embedding, constraint_info
        
    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding, constraint_info = self._state_embedding(embeddings, td)
        
        # Combine node embedding with state embedding
        base_context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        base_context = self.project_context(base_context_embedding)
        
        # Combine with constraint information
        combined_context = torch.cat([base_context, constraint_info], dim=-1)
        adapted_context = self.constraint_aware_proj(combined_context)
        
