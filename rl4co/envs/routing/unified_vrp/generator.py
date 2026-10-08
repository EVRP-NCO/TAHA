import logging
from typing import Callable, Dict, Tuple, Union

import torch
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.routing.evrptw.generator import EVRPTWGenerator


logger = logging.getLogger(__name__)


class UnifiedVRPGenerator(EVRPTWGenerator):
    """
    Unified VRP (MTEVRP) generator using stochastic task composition:
    MBT-style Bernoulli sampling.
    
    Each constraint ν ∈ V is independently activated using:
        1_ν = 1 if rand < p_ν  else 0
    Default: p_ν = 1/2 (maximum-entropy sampling across variants)

    Supported constraints:
    - energy
    - time_windows
    - backhaul
    - nonlinear_charging
    - partial_charging
    """

    def __init__(
        self,
        num_loc: int = 20,
        num_station: int = 4,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = Uniform,
        station_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_demand: float = 0,
        max_demand: float = 1,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        vehicle_capacity: float = 1.0,
        max_time: float = 9,
        charge_time: float = 0.1,
        constraint_probs: Dict[str, float] | None = None,  # NEW
        **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            num_station=num_station,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_distribution=depot_distribution,
            **kwargs,
        )

        # Default: uniform activation probability for each constraint (p = 1/2)
        if constraint_probs is None:
            constraint_probs = {
                "energy": 0.5,
                "time_windows": 0.5,
                "backhaul": 0.5,
                "nonlinear_charging": 0.5,
                "partial_charging": 0.5,
            }

        self.constraint_probs = constraint_probs


    # -----------------------------
    # MBT-style Bernoulli sampling
    # -----------------------------
    def _sample_flag(self, batch_size: int, key: str, device="cpu"):
        """Return a 0/1 vector using Bernoulli(p)."""
        p = self.constraint_probs[key]
        return (torch.rand(batch_size, device=device) < p)


    def _generate(self, batch_size, device="cpu") -> TensorDict:
        """Generate a batch of unified VRP instances."""
        td = super()._generate(batch_size)

        if isinstance(batch_size, int):
            bs = batch_size
        else:
            bs = batch_size[0]

        num_station = td["stations"].size(-2)
        num_customers = td["demand"].size(-1)

        # ---- MBT/Bernoulli sampling for each constraint ----
        constraint_energy = self._sample_flag(bs, "energy", device)
        constraint_time_windows = self._sample_flag(bs, "time_windows", device)
        constraint_backhaul = self._sample_flag(bs, "backhaul", device)
        constraint_nonlinear_charging = self._sample_flag(bs, "nonlinear_charging", device)
        constraint_partial_charging = self._sample_flag(bs, "partial_charging", device)


        # ---- Apply sampled constraints ----

        # 1) Remove time windows
        inactive_tw = ~constraint_time_windows
        if inactive_tw.any():
            idx = inactive_tw.nonzero(as_tuple=True)[0]
            td["time_windows"][idx, :, 0] = 0.0
            td["time_windows"][idx, :, 1] = self.max_time
            td["durations"][idx, 1 + num_station :] = 0.0

        # 2) Backhaul (classical linehaul/backhaul split)
        if constraint_backhaul.any():
            idx = constraint_backhaul.nonzero(as_tuple=True)[0]
            linehaul = max(1, int(0.8 * num_customers))
            td["demand"][idx, linehaul:] = -td["demand"][idx, linehaul:]

        # ---- Store constraint flags ----
        td["constraint_time_windows"] = constraint_time_windows.bool()
        td["constraint_energy"] = constraint_energy.bool()
        td["constraint_backhaul"] = constraint_backhaul.bool()
        td["constraint_nonlinear_charging"] = constraint_nonlinear_charging.bool()
        td["constraint_partial_charging"] = constraint_partial_charging.bool()

        return td