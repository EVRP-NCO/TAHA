from typing import Dict, Tuple
import csv
import os
import re
import numpy as np
import torch
from tensordict.tensordict import TensorDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))


def _env_flag(name: str, default: bool) -> torch.Tensor:
    val = os.environ.get(name)
    if val is None:
        return torch.tensor([default], dtype=torch.bool)
    val = val.strip().lower()
    return torch.tensor([val not in {"0", "false", "no", ""}], dtype=torch.bool)


def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = {}
    for key in x.files:
        tensor = torch.from_numpy(x[key])
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        x_dict[key] = tensor

    if not x_dict:
        raise ValueError(f"No arrays found in npz file: {filename}")

    first_key = next(iter(x_dict))
    batch_size = x_dict[first_key].shape[0]

    if "locs" not in x_dict or "demand" not in x_dict:
        raise KeyError("npz file must contain 'locs' and 'demand' arrays for EVRP datasets")

    locs = x_dict["locs"].float()
    stations = x_dict.get("stations", torch.zeros(batch_size, 0, locs.size(-1), dtype=locs.dtype))
    demand = x_dict["demand"].float().clamp_(-1.0, 1.0)

    num_station = stations.size(-2)
    num_customers = locs.size(-2)
    node_count = 1 + num_station + num_customers

    durations = torch.zeros(batch_size, node_count, dtype=locs.dtype)
    time_windows = torch.zeros(batch_size, node_count, 2, dtype=locs.dtype)
    time_windows[..., 1] = 1.0

    constraint_time_windows = torch.zeros(batch_size, dtype=torch.bool)
    constraint_energy = torch.ones(batch_size, dtype=torch.bool)
    constraint_backhaul = torch.zeros(batch_size, dtype=torch.bool)

    x_dict.update(
        {
            "locs": locs,
            "stations": stations.float(),
            "demand": demand,
            "durations": durations,
            "time_windows": time_windows,
            "constraint_time_windows": constraint_time_windows,
            "constraint_energy": constraint_energy,
            "constraint_backhaul": constraint_backhaul,
        }
    )

    if "depot" in x_dict:
        x_dict["depot"] = x_dict["depot"].float()

    return TensorDict(x_dict, batch_size=[batch_size])


def load_evrp_to_tensordict(filename):
    f = open(filename, "r")
    content = f.read()
    # vehicles = torch.Tensor(
    #     [int(re.search("VEHICLES: (\d+)", content, re.MULTILINE).group(1))]
    # ).unsqueeze(0)
    # optimalValue = float(
    #     re.search("OPTIMAL_VALUE: (\d+)", content, re.MULTILINE).group(1)
    # )
    capacity = float(re.search(r"CAPACITY: (\d+)", content, re.MULTILINE).group(1))
    # dimension = int(re.search("DIMENSION: (\d+)", content, re.MULTILINE).group(1))
    station_number = int(re.search(r"STATIONS: (\d+)", content, re.MULTILINE).group(1))
    energy_capacity = float(
        re.search(r"ENERGY_CAPACITY: (\d+)", content, re.MULTILINE).group(1)
    )
    energy_consumption = float(
        re.search(r"ENERGY_CONSUMPTION: (\d+\.?\d*)", content, re.MULTILINE).group(1)
    )
    max_length=energy_capacity / energy_consumption/3
    demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    demand = torch.Tensor([float(b) for a, b in demand][1:]).unsqueeze(0)
    nodes = re.findall(r"^(\d+)( +)([+-]?\d+(?:\.\d+)?)( +)([+-]?\d+(?:\.\d+)?)", content, re.MULTILINE)
    nodes = torch.Tensor([[float(c), float(e)] for a, b, c,d,e in nodes])
    bias = torch.min(nodes, dim=0).values.unsqueeze(0)
    nodes = (nodes - bias) / max_length
    depot = nodes[0].unsqueeze(0)
    stations = nodes[-station_number:].unsqueeze(0)
    locs = nodes[1:-station_number].unsqueeze(0)
    min_times = torch.full((1, nodes.size(-2)), 0.0)
    max_times = torch.full((1, nodes.size(-2)), 2000)
    time_windows = torch.stack((min_times, max_times), dim=-1)
    durations = torch.full((1, nodes.size(-2)), 0)
    # f = open(filename, "r")
    # content = f.read()
    # vehicles = torch.Tensor(
    #     [int(re.search("VEHICLES: (\d+)", content, re.MULTILINE).group(1))]
    # ).unsqueeze(0)
    # optimalValue = float(
    #     re.search("OPTIMAL_VALUE: (\d+)", content, re.MULTILINE).group(1)
    # )
    # capacity = float(re.search("CAPACITY: (\d+)", content, re.MULTILINE).group(1))
    # dimension = int(re.search("DIMENSION: (\d+)", content, re.MULTILINE).group(1))
    # station_number = int(re.search("STATIONS: (\d+)", content, re.MULTILINE).group(1))
    # energy_capacity = float(
    #     re.search("ENERGY_CAPACITY: (\d+)", content, re.MULTILINE).group(1)
    # )
    # energy_consumption = float(
    #     re.search("ENERGY_CONSUMPTION: (\d+\.?\d*)", content, re.MULTILINE).group(1)
    # )
    # max_length = energy_capacity / energy_consumption / 3
    # demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    # demand = torch.Tensor([float(b) for a, b in demand][1:]).unsqueeze(0)
    # nodes = re.findall(r"^(\d+)( +)([+-]?\d+(?:\.\d+)?)( +)([+-]?\d+(?:\.\d+)?)", content, re.MULTILINE)
    # nodes = torch.Tensor([[float(c), float(e)] for a, b, c,d,e in nodes])
    # bias = torch.min(nodes, dim=0).values.unsqueeze(0)
    # nodes = (nodes - bias) / max_length
    # depot = nodes[0].unsqueeze(0)
    # stations = nodes[-station_number:].unsqueeze(0)
    # locs = nodes[1:-station_number].unsqueeze(0)
    # min_times = torch.full((1, nodes.size(-2)), 0.0)
    # max_times = torch.full((1, nodes.size(-2)), 3000)
    # time_windows = torch.stack((min_times, max_times), dim=-1)
    # durations = torch.full((1, nodes.size(-2)), 0)
    constraint_time_windows = torch.zeros(1, dtype=torch.bool)
    constraint_energy = torch.ones(1, dtype=torch.bool)
    constraint_backhaul = torch.zeros(1, dtype=torch.bool)
    constraint_nonlinear_charging = _env_flag("EVRP_FORCE_NC", False)
    constraint_partial_charging = _env_flag("EVRP_FORCE_PC", False)
    td = TensorDict(
        {
            "locs": locs,
            "depot": depot,
            "stations": stations,
            "demand": demand / capacity,
            "factor": torch.Tensor([max_length]),
            "durations": durations,
            "time_windows": time_windows,
            "inverse_refueling_rate": torch.Tensor([3.47 / max_length]),  # Default value for EVRP, normalized
            "constraint_time_windows": constraint_time_windows,
            "constraint_energy": constraint_energy,
            "constraint_backhaul": constraint_backhaul,
            "constraint_nonlinear_charging": constraint_nonlinear_charging,
            "constraint_partial_charging": constraint_partial_charging,
        },
        batch_size=[1],
    )
    return td

def load_txt_to_tensordict(filename):
    with open(filename, "r") as f:
        content = f.read()

    upper_content = content.upper()

    # 判断 EVRP with Backhauls & Time Windows 格式
    # 判断 EVRPTW 格式（包含燃料容量定义）
    if "FUEL TANK CAPACITY" in upper_content:
        return _load_evrptw_from_text(content)

    raise ValueError(
        f"Unsupported TXT format for {filename}. Supported: EVRP, EVRPTW, or Solomon CVRPTW."
    )
def _load_evrptw_from_text(content: str) -> TensorDict:
    capacity = float(
        re.search("C Vehicle load capacity /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    energy_capacity = float(
        re.search(
            "Q Vehicle fuel tank capacity /(\d+\.?\d*)/", content, re.MULTILINE
        ).group(1)
    )
    energy_consumption = float(
        re.search("r fuel consumption rate /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    velocity = float(
        re.search("v average Velocity /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    recharge = float(
        re.search("g inverse refueling rate /(\d+\.?\d*)/", content, re.MULTILINE).group(
            1
        )
    )
    depot_add = re.findall(
        r"d          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)       (\d+\.?\d* ?)       (\d+\.?\d*[ ]{0,3})     (\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    depot_add = torch.Tensor(
        [[float(a), float(b), float(e)] for a, b, c, d, e in depot_add]
    )
    stations = re.findall(
        r"f          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    stations = torch.Tensor([[float(a), float(b)] for a, b in stations])
    customs = re.findall(
        r"c          (-?\d+\.?\d* ?)       (-?\d+\.?\d* ?)       (\d+\.?\d* ?)       (\d+\.?\d*[ ]{0,3})     (\d+\.?\d*[ ]{0,2})     (\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    customs = torch.Tensor(
        [
            [float(a), float(b), float(c), float(d), float(e), float(f)]
            for a, b, c, d, e, f in customs
        ]
    )
    demand = torch.clamp(customs[:, 2] / capacity, min=0.0, max=1.0).unsqueeze(0)
    customs_loc = customs[:, :2]
    max_length = energy_capacity / energy_consumption / velocity / 3.0
    charge_time = energy_capacity * recharge / max_length
    customs_start = customs[:, 3] / max_length
    customs_end = customs[:, 4] / max_length
    service = customs[:, 5] / max_length
    max_time = depot_add[0, 2] / max_length
    depot = depot_add[:, :2]
    nodes = torch.cat((depot, stations, customs_loc), dim=-2)
    bias = torch.min(nodes, dim=0).values.unsqueeze(0)
    depot = ((depot - bias) / max_length).cuda()
    stations = ((stations - bias) / max_length).unsqueeze(0).cuda()
    customs = ((customs_loc - bias) / max_length).unsqueeze(0).cuda()
    depot_duration = torch.zeros((1, 1)).cuda()
    station_durations = torch.full((1, stations.size(-2)), charge_time).cuda()
    customer_durations = service.unsqueeze(0).cuda()
    durations = torch.cat((depot_duration, station_durations, customer_durations), dim=-1)
    min_times = torch.full((1, 1 + stations.size(-2) + customs.size(-2)), 0.0).cuda()
    max_times = torch.full((1, 1 + stations.size(-2) + customs.size(-2)), max_time).cuda()
    min_times[:, 1 + stations.size(-2) :] = customs_start.unsqueeze(0).cuda()
    max_times[:, 1 + stations.size(-2) :] = customs_end.unsqueeze(0).cuda()
    time_windows = torch.stack((min_times, max_times), dim=-1)
    constraint_time_windows = torch.ones(1, dtype=torch.bool)
    constraint_energy = torch.ones(1, dtype=torch.bool)
    constraint_backhaul = torch.zeros(1, dtype=torch.bool)
    constraint_nonlinear_charging = _env_flag("EVRP_FORCE_NC", True)
    constraint_partial_charging = _env_flag("EVRP_FORCE_PC", True)

    td = TensorDict(
        {
            "locs": customs,
            "depot": depot,
            "stations": stations,
            "demand": demand,
            "factor": torch.Tensor([max_length]),
            "durations": durations,
            "time_windows": time_windows,
            "inverse_refueling_rate": torch.Tensor([recharge / max_length]),
            "constraint_time_windows": constraint_time_windows,
            "constraint_energy": constraint_energy,
            "constraint_backhaul": constraint_backhaul,
            "constraint_nonlinear_charging": constraint_nonlinear_charging,
            "constraint_partial_charging": constraint_partial_charging,

        },
        batch_size=[1],
    )
    return td


def load_evrpbtw_csv_to_tensordict(filename: str) -> TensorDict:
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in reader
        ]

    if not rows:
        raise ValueError(f"CSV file {filename} is empty")

    def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
        value = row.get(key, "") if row else ""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value '{value}' for column '{key}'") from exc

    depot_row = next((row for row in rows if row.get("Type", "").strip().upper() == "D"), None)
    if depot_row is None:
        raise ValueError(f"CSV file {filename} must contain a depot row (Type 'D')")

    capacity = max(_as_float(depot_row, "C", 1.0), 1e-6)
    energy_capacity = max(_as_float(depot_row, "Q", 1.0) / 1.5, 1e-6)
    energy_consumption = max(_as_float(depot_row, "r", 1.0), 1e-6)
    velocity = max(_as_float(depot_row, "v", 1.0), 1e-6)
    recharge_rate = _as_float(depot_row, "g", 0.0)
    max_length = energy_capacity / (energy_consumption * velocity * 3.0)
    if max_length <= 0:
        raise ValueError("Computed max_length must be positive; please validate instance parameters")

    station_rows = [
        row for row in rows if row.get("Type", "").strip().upper() in {"S", "C"}
    ]
    customer_rows = [
        row for row in rows if row.get("Type", "").strip().upper() in {"L", "B"}
    ]
    if not customer_rows:
        raise ValueError(f"CSV file {filename} must contain at least one customer row")

    def _coord_tensor(row_list):
        if not row_list:
            return torch.zeros(0, 2, dtype=torch.float32)
        coords = torch.tensor(
            [[_as_float(row, "x", 0.0), _as_float(row, "y", 0.0)] for row in row_list],
            dtype=torch.float32,
        )
        return coords

    depot_coord = _coord_tensor([depot_row])
    station_coords = _coord_tensor(station_rows)
    customer_coords = _coord_tensor(customer_rows)
    nodes = torch.cat((depot_coord, station_coords, customer_coords), dim=0)
    bias = nodes.min(dim=0, keepdim=True).values
    scaled_nodes = (nodes - bias) / max_length

    num_station = station_coords.size(0)
    num_customers = customer_coords.size(0)

    depot_scaled = scaled_nodes[:1]
    station_scaled = (
        scaled_nodes[1 : 1 + num_station].unsqueeze(0)
        if num_station > 0
        else torch.zeros(1, 0, 2, dtype=torch.float32)
    )
    customer_scaled = scaled_nodes[1 + num_station :].unsqueeze(0)

    raw_demand = torch.tensor(
        [_as_float(row, "demand", 0.0) / capacity for row in customer_rows], dtype=torch.float32
    )
    positive_part = torch.clamp(raw_demand, min=0.0, max=1.0)
    negative_part = torch.clamp(raw_demand, min=-1.0, max=0.0)
    demand = torch.where(raw_demand >= 0, positive_part, negative_part).unsqueeze(0)

    def _time_pair(row: Dict[str, str], default_due: float) -> Tuple[float, float]:
        ready = _as_float(row, "ReadyTime", 0.0)
        due = _as_float(row, "DueTime", default_due)
        return ready, due

    depot_due = _as_float(depot_row, "DueTime", 0.0)
    ready_values = [0.0]
    due_values = [depot_due]
    for row in station_rows:
        ready, due = _time_pair(row, depot_due)
        ready_values.append(ready)
        due_values.append(due if due > 0 else depot_due)
    for row in customer_rows:
        ready, due = _time_pair(row, depot_due)
        ready_values.append(ready)
        due_values.append(due if due > 0 else depot_due)

    ready_tensor = torch.tensor(ready_values, dtype=torch.float32).unsqueeze(0) / max_length
    due_tensor = torch.tensor(due_values, dtype=torch.float32).unsqueeze(0) / max_length
    time_windows = torch.stack((ready_tensor, due_tensor), dim=-1)

    customer_service = torch.tensor(
        [_as_float(row, "ServiceTime", 0.0) for row in customer_rows], dtype=torch.float32
    ).unsqueeze(0) / max_length
    charge_time = energy_capacity * recharge_rate / max_length if recharge_rate > 0 else 0.0
    depot_duration = torch.zeros(1, 1, dtype=torch.float32)
    station_duration = (
        torch.full((1, num_station), charge_time, dtype=torch.float32)
        if num_station > 0
        else torch.zeros(1, 0, dtype=torch.float32)
    )
    durations = torch.cat((depot_duration, station_duration, customer_service), dim=-1)

    inverse_refueling_rate = torch.tensor([recharge_rate / max_length], dtype=torch.float32)

    node_type_map = {"D": 0, "S": 1, "C": 1, "L": 2, "B": 3}
    customer_type_map = {"L": 0, "B": 1}

    node_types = [node_type_map.get(depot_row.get("Type", "D").upper(), 0)]
    node_types.extend(node_type_map.get(row.get("Type", "S").upper(), 1) for row in station_rows)
    node_types.extend(node_type_map.get(row.get("Type", "L").upper(), 2) for row in customer_rows)
    node_types_tensor = torch.tensor(node_types, dtype=torch.int64).unsqueeze(0)

    customer_types_tensor = torch.tensor(
        [customer_type_map.get(row.get("Type", "L").upper(), 0) for row in customer_rows],
        dtype=torch.int64,
    ).unsqueeze(0)

    td = TensorDict(
        {
            "locs": customer_scaled,
            "depot": depot_scaled,
            "stations": station_scaled,
            "demand": demand,
            "durations": durations,
            "time_windows": time_windows,
            "factor": torch.tensor([max_length], dtype=torch.float32),
            "inverse_refueling_rate": inverse_refueling_rate,
            "constraint_time_windows": torch.ones(1, dtype=torch.bool),
            "constraint_energy": torch.ones(1, dtype=torch.bool),
            "constraint_backhaul": torch.ones(1, dtype=torch.bool),
            "constraint_nonlinear_charging": _env_flag("EVRP_FORCE_NC", True),
            "constraint_partial_charging": _env_flag("EVRP_FORCE_PC", True),
            "vehicle_capacity": torch.ones(1, 1, dtype=torch.float32),
            "node_types": node_types_tensor,
            "customer_types": customer_types_tensor,
        },
        batch_size=[1],
    )
    return td



def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def load_solomon_instance(name, path=None, edge_weights=False):
    """Load solomon instance from a file"""
    import vrplib

    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.txt"
    if not os.path.isfile(file_path):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(
        path=file_path,
        instance_format="solomon",
        compute_edge_weights=edge_weights,
    )


def load_solomon_solution(name, path=None):
    """Load solomon solution from a file"""
    import vrplib

    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.sol"
    if not os.path.isfile(file_path):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=file_path)





