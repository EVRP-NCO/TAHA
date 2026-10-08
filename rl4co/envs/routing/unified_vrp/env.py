from typing import Optional
import os
import logging

import numpy as np
import torch

from tensordict.tensordict import TensorDict

from rl4co.data.utils import (
    load_txt_to_tensordict,
    load_npz_to_tensordict,
    load_evrp_to_tensordict,
    load_evrpbtw_csv_to_tensordict,
)
from rl4co.envs.routing.evrp.env import EVRPEnv
from rl4co.utils.ops import gather_by_index
from ..evrp.generator import EVRPGenerator
from .generator import EVRPTWGenerator, UnifiedVRPGenerator
from ..evrp.render import render

# Initialize logger
log = logging.getLogger(__name__)
# Set logging level (if needed)
# logging.basicConfig(level=logging.INFO) # If not already configured elsewhere

# Define TIME_EPSILON for time window checks
TIME_EPSILON = 1e-6
MAX_ENERGY = 3.0 # Module-level constant for maximum energy capacity
BACKHAUL_CAPACITY_THRESHOLD = 0.7
CAPACITY_EPS = 1e-6

# Add these near the top of the class or function
class UnifiedVRPEnv(EVRPEnv):

    name = "unified_vrp"
    # MAX_ENERGY = 3.0 removed, will use module-level constant

    def __init__(
        self,
        generator: UnifiedVRPGenerator = None,
        generator_params: dict = {},
        use_dynamic_charging: bool = True, # Whether to use dynamic charging time
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            # Note: The generator should be UnifiedVRPGenerator for mixed problems
            generator = UnifiedVRPGenerator(**generator_params)
        self.generator = generator
        self.use_dynamic_charging = use_dynamic_charging


    def calculate_dynamic_charging_time(self, td: TensorDict, current_node: torch.Tensor) -> torch.Tensor:
        """Calculate dynamic charging time based on current energy consumption (only called at stations)"""
        batch_size = td["locs"].shape[0]
        device = td.device
        
        # Calculate travel distance from previous node to current node
        previous_node = td["current_node"].squeeze(-1)
        current_node_idx = current_node.squeeze(-1)
        batch_index = torch.arange(batch_size, device=device)
        
        # Distance from previous node to current station
        dist = td["dist_matrix"][batch_index, previous_node, current_node_idx].unsqueeze(-1)
        
        # Energy consumed upon arrival
        energy_consumed = td["used_length"] + dist
        current_energy = torch.clamp(MAX_ENERGY - energy_consumed, min=0.0)
        
        # Determine Target Energy
        # Default is full charge (MAX_ENERGY)
        target_energy = torch.full_like(current_energy, MAX_ENERGY)
        
        # Apply Partial Charging Constraint (Charge up to 70%)
        if "constraint_partial_charging" in td.keys():
            flag_partial = td["constraint_partial_charging"]
            if flag_partial.ndim == 1: flag_partial = flag_partial.unsqueeze(-1)
            # If constraint is active, target is 70%
            target_energy = torch.where(flag_partial, torch.tensor(0.7 * MAX_ENERGY, device=device), target_energy)
            
        # Ensure target >= current (cannot discharge)
        target_energy = torch.max(target_energy, current_energy)
        
        # Calculate Energy to Charge
        energy_to_charge = target_energy - current_energy
        
        # Base Inverse Refueling Rate (Time per unit Energy)
        inverse_refueling_rate = td["inverse_refueling_rate"].squeeze(-1).unsqueeze(-1)
        
        # Apply Nonlinear Charging Constraint
        if "constraint_nonlinear_charging" in td.keys():
            flag_nonlinear = td["constraint_nonlinear_charging"]
            if flag_nonlinear.ndim == 1: flag_nonlinear = flag_nonlinear.unsqueeze(-1)
            
            # Segments: 0-20%, 20-80%, 80-100%
            # Time Multipliers: 1x, 2x, 4x (since rate decays by half: 1, 0.5, 0.25)
            E_20 = torch.tensor(0.2 * MAX_ENERGY, device=device)
            E_80 = torch.tensor(0.8 * MAX_ENERGY, device=device)
            E_100 = torch.tensor(MAX_ENERGY, device=device)
            
            def get_overlap(low, high, start, end):
                return torch.clamp(torch.min(high, end) - torch.max(low, start), min=0.0)
            
            # Energy charged in each segment
            e_seg1 = get_overlap(torch.tensor(0.0, device=device), E_20, current_energy, target_energy)
            e_seg2 = get_overlap(E_20, E_80, current_energy, target_energy)
            e_seg3 = get_overlap(E_80, E_100, current_energy, target_energy)
            
            nonlinear_time = (e_seg1 * 1.0 + e_seg2 * 2.0 + e_seg3 * 4.0) * inverse_refueling_rate
            linear_time = energy_to_charge * inverse_refueling_rate
            
            dynamic_charging_time = torch.where(flag_nonlinear, nonlinear_time, linear_time)
        else:
            dynamic_charging_time = energy_to_charge * inverse_refueling_rate
            
        return dynamic_charging_time

    def _step(self, td: TensorDict) -> TensorDict:
        device = td.device
        batch_size = td.batch_size[0]
        current_node = td["action"].unsqueeze(-1)
        previous_node = td["current_node"].squeeze(-1)
        num_station = td["locs"].size(-2) - td["demand"].size(-1) - 1

        batch_index = torch.arange(batch_size, device=device)
        next_index = current_node.squeeze(-1)
        travel_time = td["dist_matrix"][batch_index, previous_node, next_index].unsqueeze(-1)

        static_duration = gather_by_index(td["durations"], td["action"]).reshape(batch_size, 1)
        is_station = ((current_node > 0) & (current_node <= num_station)).float()
        if self.use_dynamic_charging and "inverse_refueling_rate" in td and is_station.any():
            dynamic_time = self.calculate_dynamic_charging_time(td, current_node)
            duration = static_duration * (1 - is_station) + dynamic_time * is_station
        else:
            duration = static_duration

        time_windows = gather_by_index(td["time_windows"], td["action"])
        start_times = time_windows[..., 0].reshape(batch_size, 1)
        end_times = time_windows[..., 1].reshape(batch_size, 1)
        arrival_time = td["current_time"] + travel_time
        service_start = torch.max(arrival_time, start_times)
        departure_time = service_start + duration

        time_window_flags = td.get("constraint_time_windows", None)
        active_time_windows = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        if time_window_flags is not None:
            tw_tensor = time_window_flags.to(device=device, dtype=torch.bool).reshape(batch_size, -1)
            active_time_windows = tw_tensor[:, [0]]

        late_arrival = arrival_time > end_times
        penalty = torch.zeros_like(td["penalty_time"])
        if late_arrival.any():
            penalty_amount = torch.where(
                active_time_windows,
                arrival_time - end_times,
                torch.zeros_like(arrival_time),
            )
            penalty_mask = late_arrival & active_time_windows
            penalty[penalty_mask] = penalty_amount[penalty_mask]
        td["penalty_time"] = td["penalty_time"] + penalty

        td["current_time"] = torch.where(current_node != 0, departure_time, torch.zeros_like(departure_time))

        num_customers = td["demand"].size(-1)
        padded_demand = torch.nn.functional.pad(td["demand"], (1, 0), value=0.0)
        demand_indices = torch.clamp(current_node - num_station, min=0, max=num_customers)
        selected_demand = gather_by_index(padded_demand, demand_indices, squeeze=False)

        vehicle_capacity = td.get("vehicle_capacity", torch.ones_like(td["used_capacity"]))

        constraint_backhaul_flag = td.get(
            "constraint_backhaul",
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        )
        if not torch.is_tensor(constraint_backhaul_flag):
            constraint_backhaul_flag = torch.as_tensor(
                constraint_backhaul_flag, dtype=torch.bool, device=device
            )
        constraint_backhaul_flag = constraint_backhaul_flag.to(device=device, dtype=torch.bool)
        constraint_backhaul_flag = constraint_backhaul_flag.reshape(batch_size, -1)[:, :1]

        capacity_candidate = td["used_capacity"] + selected_demand
        linehaul_mask = selected_demand >= 0
        normalized_cap = torch.minimum(vehicle_capacity, torch.ones_like(vehicle_capacity))
        cap_limit = torch.where(constraint_backhaul_flag, normalized_cap, vehicle_capacity)
        updated_capacity_linehaul = torch.minimum(torch.clamp(capacity_candidate, min=0.0), cap_limit)
        updated_capacity_backhaul = torch.clamp(capacity_candidate, min=0.0)
        used_capacity = torch.where(linehaul_mask, updated_capacity_linehaul, updated_capacity_backhaul)
        used_capacity = torch.where(current_node != 0, used_capacity, torch.zeros_like(used_capacity))

        index_helper = torch.arange(batch_size, device=device)[:, None]
        travel_vector = (
            td["locs"][index_helper, td["current_node"]]
            - td["locs"][index_helper, current_node]
        ).norm(p=2, dim=-1)
        
        # Update used_length (energy consumption)
        # If customer: accumulate
        # If depot: reset to 0
        # If station: reset to (MAX_ENERGY - target_energy)
        
        accumulated_length = travel_vector + td["used_length"]
        
        target_energy_at_station = torch.full((batch_size, 1), MAX_ENERGY, device=device)
        if "constraint_partial_charging" in td.keys():
             flag = td["constraint_partial_charging"]
             if flag.ndim == 1: flag = flag.unsqueeze(-1)
             target_energy_at_station = torch.where(flag, torch.tensor(0.7 * MAX_ENERGY, device=device), target_energy_at_station)
        
        reset_used_length = MAX_ENERGY - target_energy_at_station
        
        used_length = torch.where(
            current_node > num_station,
            accumulated_length, # Customer
            torch.where(
                current_node == 0,
                torch.zeros_like(accumulated_length), # Depot
                reset_used_length # Station
            )
        )

        visited = td["visited"].clone()
        visit = visited.scatter(-1, current_node, 1)
        if num_station > 0:
            visit[..., 1 : 1 + num_station][(current_node > num_station).expand(batch_size, num_station)] = 0

        backhaul_phase = td.get("backhaul_phase", torch.zeros_like(current_node, dtype=torch.bool))
        backhaul_phase = backhaul_phase.to(device=device, dtype=torch.bool)
        is_depot = current_node == 0
        is_customer = current_node > num_station
        selected_is_backhaul = (selected_demand.squeeze(-1) < 0).unsqueeze(-1) & is_customer
        backhaul_phase = torch.where(
            is_depot,
            torch.zeros_like(backhaul_phase),
            backhaul_phase | selected_is_backhaul,
        )

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visit,
                "used_length": used_length,
                "current_time": td["current_time"],
                "penalty_time": td["penalty_time"],
                "penalty_energy": td["penalty_energy"],
                "backhaul_phase": backhaul_phase,
            }
        )

        action_mask = self.get_action_mask(td)
        td.set("action_mask", action_mask)

        td.update({"i": td["i"] + 1})

        all_customers_visited = td["visited"][..., 1 + num_station :].all(dim=-1)
        depot_condition = (current_node.squeeze(-1) == 0).unsqueeze(-1)
        done = depot_condition & all_customers_visited.unsqueeze(-1)
        td.update({"done": done})

        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if td is not None:
            device = td.device
        else:
            device = getattr(self.generator, "device", torch.device("cpu"))
            if batch_size is None:
                batch_size = self.generator.batch_size
            td = self.generator(batch_size=batch_size).to(device)

        locs_input = td["locs"].to(device)
        depot_input = td["depot"].to(device)
        stations_input = td["stations"].to(device)
        demand_input = td["demand"].to(device)
        durations_input = td["durations"].to(device)
        time_windows_input = td["time_windows"].to(device)

        actual_batch_size = locs_input.shape[:-2]
        num_instances = int(np.prod(actual_batch_size)) if len(actual_batch_size) > 0 else 1

        def _extract_flag(name: str) -> torch.Tensor:
            if name in td.keys():
                value = td[name]
            if value.numel() == 1 and num_instances > 1:
                value = value.expand(num_instances)
            elif value.numel() != num_instances:
                repeats = max(1, (num_instances + value.numel() - 1) // max(value.numel(), 1))
                value = value.repeat(repeats)[:num_instances]
            return value.view(*actual_batch_size)

        constraint_time_windows = _extract_flag("constraint_time_windows")
        constraint_energy = _extract_flag("constraint_energy")
        constraint_backhaul = _extract_flag("constraint_backhaul")
        constraint_nonlinear_charging = _extract_flag("constraint_nonlinear_charging")
        constraint_partial_charging = _extract_flag("constraint_partial_charging")

        # Calculate task_id based on constraints
        task_id = (
            constraint_time_windows.long() * 1 + 
            constraint_energy.long() * 2 + 
            constraint_backhaul.long() * 4 +
            constraint_nonlinear_charging.long() * 8 + 
            constraint_partial_charging.long() * 16
        )

        all_locs_for_reset = torch.cat((depot_input[..., None, :], stations_input, locs_input), -2)
        if "capacity" in td.keys():
            capacity_tensor = td["capacity"].to(device=device, dtype=torch.float32)
            if capacity_tensor.ndim == len(actual_batch_size):
                capacity_tensor = capacity_tensor.unsqueeze(-1)
            vehicle_capacity = capacity_tensor.reshape(*actual_batch_size, 1)
        elif "vehicle_capacity" in td.keys():
            capacity_tensor = td["vehicle_capacity"].to(device=device, dtype=torch.float32)
            vehicle_capacity = capacity_tensor.reshape(*actual_batch_size, 1)
        else:
            vehicle_capacity = torch.full(
                (*actual_batch_size, 1),
                float(getattr(self.generator, "vehicle_capacity", 1.0)),
                dtype=torch.float32,
                device=device,
            )

        td_reset = TensorDict(
            {
                "locs": all_locs_for_reset,
                "demand": demand_input,
                "current_node": torch.zeros(*actual_batch_size, 1, dtype=torch.long, device=device),
                "current_time": torch.zeros(*actual_batch_size, 1, dtype=torch.float32, device=device),
                "penalty_time": torch.zeros(*actual_batch_size, 1, dtype=torch.float32, device=device),
                "penalty_energy": torch.zeros(*actual_batch_size, 1, dtype=torch.float32, device=device),
                "used_length": torch.zeros((*actual_batch_size, 1), device=device),
                "used_capacity": torch.zeros((*actual_batch_size, 1), device=device),
                "vehicle_capacity": vehicle_capacity,
                "visited": torch.zeros(
                    (*actual_batch_size, locs_input.shape[-2] + stations_input.size(-2) + 1),
                    dtype=torch.bool,
                    device=device,
                ).scatter_(-1, torch.zeros(*actual_batch_size, 1, dtype=torch.long, device=device), 1),
                "durations": durations_input,
                "time_windows": time_windows_input,
                "constraint_time_windows": constraint_time_windows,
                "constraint_energy": constraint_energy,
                "constraint_backhaul": constraint_backhaul,
                "constraint_nonlinear_charging":constraint_nonlinear_charging,
                "constraint_partial_charging":constraint_partial_charging,
                "task_id": task_id,
                "i": torch.zeros(*actual_batch_size, 1, dtype=torch.int64, device=device),
            },
            batch_size=actual_batch_size,
            device=device,
        )
        # Note: Setting device=device for TensorDict helps, but explicit .to(device) for inputs is safer.

        # Pre-calculate distance matrix
        dist_matrix = torch.cdist(all_locs_for_reset, all_locs_for_reset, p=2.0)
        td_reset.set("dist_matrix", dist_matrix)

        # Carry over inverse_refueling_rate if it exists
        if "inverse_refueling_rate" in td.keys():
            td_reset.set("inverse_refueling_rate", td["inverse_refueling_rate"])

        action_mask = self.get_action_mask(td_reset)
        td_reset.set("action_mask", action_mask)
        # Copy factor if present
        if "factor" in td.keys():
            td_reset.set(
                "factor",
                td["factor"].to(device), # Ensure factor is also on the correct device
            )
        td_reset.set(
            "backhaul_phase",
            torch.zeros(*actual_batch_size, 1, dtype=torch.bool, device=device),
        )
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Create dedicated action masks for each problem type without shared fallbacks."""

        locs = td["locs"]
        current_node = td["current_node"]
        demand = td["demand"]
        used_capacity = td["used_capacity"]
        vehicle_capacity = td.get("vehicle_capacity", torch.ones_like(used_capacity))
        used_length = td["used_length"]
        visited = td["visited"]
        current_time = td["current_time"]
        time_windows = td["time_windows"]

        batch_shape = td.batch_size
        batch_size = int(batch_shape[0]) if len(batch_shape) > 0 else 1
        device = td.device

        num_nodes = locs.size(-2)
        num_customers = demand.size(-1)
        num_station = num_nodes - num_customers - 1

        visited_customers = visited[..., num_station + 1 :]
        visited_stations = visited[..., 1 : num_station + 1]
        all_customers_visited = visited_customers.all(-1, keepdim=True)

        mask_depot = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        mask_station = torch.zeros(batch_size, num_station, dtype=torch.bool, device=device)
        mask_loc = torch.zeros(batch_size, num_customers, dtype=torch.bool, device=device)

        mask_loc |= visited_customers
        if num_station > 0:
            mask_station |= visited_stations

        depot_current = (current_node == 0)
        mask_depot |= depot_current & ~all_customers_visited
        if num_station > 0:
            mask_station |= (depot_current & all_customers_visited).expand(-1, num_station)

        capacity_violation = demand + used_capacity > vehicle_capacity
        mask_loc |= capacity_violation

        def _flag(name: str, default: bool) -> torch.Tensor:
            if name in td.keys():
                value = td[name]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, dtype=torch.bool, device=device)
                else:
                    value = value.to(device=device, dtype=torch.bool)
                value = value.reshape(-1)
            else:
                value = torch.full((1,), default, dtype=torch.bool, device=device)
            if value.numel() == 1 and batch_size > 1:
                value = value.expand(batch_size)
            elif value.numel() != batch_size:
                repeats = max(1, (batch_size + value.numel() - 1) // max(value.numel(), 1))
                value = value.repeat(repeats)[:batch_size]
            return value

        constraint_time_windows = _flag("constraint_time_windows", False)
        constraint_energy = _flag("constraint_energy", False)
        constraint_backhaul = _flag("constraint_backhaul", False)

        depot_energy_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        tw_violations = None
        if constraint_time_windows.any():
            tw_violations = UnifiedVRPEnv._compute_time_window_violations(
                td, current_node, current_time, time_windows, num_station
            )
            mask_loc[constraint_time_windows] |= tw_violations[constraint_time_windows]

        if num_station > 0:
            mask_station[~constraint_energy] = True

        if constraint_energy.any():
            energy_masks = UnifiedVRPEnv._compute_energy_violations(td, current_node, used_length, num_station)
            energy_subset = constraint_energy
            mask_loc[energy_subset] |= energy_masks["customer"][energy_subset]
            if num_station > 0:
                mask_station[energy_subset] |= energy_masks["station"][energy_subset]
                mask_station[energy_subset] |= energy_masks["prefer_depot"][energy_subset].expand(-1, num_station)
            mask_depot[energy_subset] |= energy_masks["depot"][energy_subset]
            depot_energy_mask[energy_subset] = energy_masks["depot"][energy_subset]

        if constraint_backhaul.any():
            zero_pad = torch.zeros(batch_size, 1 + num_station, dtype=demand.dtype, device=device)
            demand_full = torch.cat((zero_pad, demand), dim=-1)
            demand_index = torch.clamp(current_node, min=0, max=demand_full.size(-1) - 1)
            current_demand = gather_by_index(demand_full, demand_index, squeeze=False)
            current_is_backhaul = (current_demand < 0)

            backhaul_phase = td.get("backhaul_phase", torch.zeros_like(current_node, dtype=torch.bool))
            backhaul_phase = backhaul_phase.to(device=device, dtype=torch.bool).squeeze(-1)

            enforce_batches = constraint_backhaul & (backhaul_phase | current_is_backhaul.squeeze(-1))
            if enforce_batches.any():
                positive_customers = demand > 0
                mask_loc[enforce_batches] |= positive_customers[enforce_batches]

        final_mask_bool = torch.cat((mask_depot, mask_station, mask_loc), -1)
        return (~final_mask_bool).bool()

    @staticmethod
    def _compute_time_window_violations(
        td: TensorDict,
        current_node: torch.Tensor,
        current_time: torch.Tensor,
        time_windows: torch.Tensor,
        num_station: int,
    ) -> torch.Tensor:
        batch_size = current_node.shape[0]
        device = current_node.device
        batch_index = torch.arange(batch_size, device=device)
        current_node_idx = current_node.squeeze(-1)
        travel_times = td["dist_matrix"][batch_index, current_node_idx, :]
        arrival_times = current_time + travel_times
        end_times = time_windows[..., 1]
        violation = arrival_times > (end_times + TIME_EPSILON)
        return violation[:, 1 + num_station :]

    @staticmethod
    def _compute_energy_violations(
        td: TensorDict,
        current_node: torch.Tensor,
        used_length: torch.Tensor,
        num_station: int,
    ) -> dict:
        batch_size = current_node.shape[0]
        device = current_node.device
        num_nodes = td["locs"].size(-2)
        customer_slice = slice(1 + num_station, num_nodes)
        depot_station_slice = slice(0, 1 + num_station)

        dist_matrix = td["dist_matrix"]
        batch_index = torch.arange(batch_size, device=device)
        current_node_idx = current_node.squeeze(-1)

        dist_curr_to_customer = dist_matrix[batch_index, current_node_idx, customer_slice]
        dist_customer_to_refuel = dist_matrix[:, customer_slice, :][:, :, depot_station_slice]
        total_energy_to_customer = dist_curr_to_customer.unsqueeze(-1) + dist_customer_to_refuel

        energy_violation_customer = (
            total_energy_to_customer + used_length.unsqueeze(-1)
        ) > MAX_ENERGY*0.9
        energy_violation_customer = energy_violation_customer.all(-1)

        dist_curr_to_refuel = dist_matrix[batch_index, current_node_idx, depot_station_slice]
        energy_to_refuel = dist_curr_to_refuel + used_length

        energy_violation_station = torch.zeros(batch_size, num_station, dtype=torch.bool, device=device)
        energy_violation_station = energy_to_refuel[:, 1:] > MAX_ENERGY
        energy_violation_depot = energy_to_refuel[:, :1] > MAX_ENERGY

        current_energy = MAX_ENERGY - used_length
        high_energy_threshold = 0.5* MAX_ENERGY  # Align station mask with EVRPTW behaviour
        prefer_depot = (current_energy > high_energy_threshold) & (~energy_violation_depot)

        return {
            "customer": energy_violation_customer,
            "station": energy_violation_station,
            "depot": energy_violation_depot,
            "prefer_depot": prefer_depot,
        }

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        total_penalty = td["penalty_time"].squeeze(-1)
        td.set("path", total_penalty)

        reward = super()._get_reward(td, actions)

        return reward - total_penalty

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".txt":
            td_load = load_txt_to_tensordict(fpath)
        elif ext == ".npz":
            td_load = load_npz_to_tensordict(fpath)
        elif ext == ".evrp":
            td_load = load_evrp_to_tensordict(fpath)
        elif ext == ".csv":
            td_load = load_evrpbtw_csv_to_tensordict(fpath)
        else:
            raise ValueError(f"Unsupported data extension '{ext}' for UnifiedVRPEnv")
        return td_load

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)