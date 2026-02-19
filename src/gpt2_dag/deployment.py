from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LAYER_PAT = re.compile(r"_(\d{2})")
SHARD_PAT = re.compile(r"_(\d{2})_(\d+)$")


@dataclass
class DeploymentEstimate:
    makespan_ms: float
    cross_node_edges: int
    cross_node_bytes: float
    total_comm_ms: float


def load_aggregated_profile(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _task_layer(task: str) -> int | None:
    if task in {"embed", "ln_f", "lm_head"}:
        return None
    m = LAYER_PAT.search(task)
    if not m:
        return None
    return int(m.group(1))


def _task_shard(task: str) -> int | None:
    m = SHARD_PAT.search(task)
    if not m:
        return None
    return int(m.group(2))


def infer_num_layers(tasks: list[str]) -> int:
    layers = [_task_layer(t) for t in tasks]
    layers = [x for x in layers if x is not None]
    if not layers:
        raise ValueError("Could not infer number of layers from task names")
    return max(layers) + 1


def _stage_for_layer(layer_idx: int, num_layers: int, num_nodes: int) -> int:
    if num_layers <= 0:
        return 0
    return min(num_nodes - 1, int(layer_idx * num_nodes / num_layers))


def assign_tasks(tasks: list[str], num_nodes: int, strategy: str = "tensor") -> dict[str, int]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be >= 1")
    if strategy not in {"tensor", "pipeline"}:
        raise ValueError("strategy must be one of: tensor, pipeline")

    n_layer = infer_num_layers(tasks)
    assignment: dict[str, int] = {}
    for task in tasks:
        if task == "embed":
            assignment[task] = 0
            continue
        if task in {"ln_f", "lm_head"}:
            assignment[task] = num_nodes - 1
            continue

        layer_idx = _task_layer(task)
        if layer_idx is None:
            assignment[task] = 0
            continue
        stage_node = _stage_for_layer(layer_idx, n_layer, num_nodes)

        if strategy == "pipeline":
            assignment[task] = stage_node
            continue

        # tensor strategy: distribute shard tasks across nodes, keep merge tasks on stage node.
        if task.startswith("attn_shard_") or task.startswith("mlp_shard_"):
            shard_idx = _task_shard(task)
            assignment[task] = stage_node if shard_idx is None else int(shard_idx % num_nodes)
        else:
            assignment[task] = stage_node
    return assignment


def estimate_schedule(
    dag_structure: dict,
    task_cost_ms: dict[str, float],
    edge_size_bytes: dict[str, float],
    assignment: dict[str, int],
    bandwidth_mbps: float = 100.0,
    link_latency_ms: float = 0.2,
) -> DeploymentEstimate:
    order = dag_structure["execution_order"]
    edges = dag_structure["edges"]
    preds: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        preds[e["target"]].append(e["source"])

    node_ready: dict[int, float] = defaultdict(float)
    finish: dict[str, float] = {}
    total_comm_ms = 0.0
    cross_edges = 0
    cross_bytes = 0.0
    bps = bandwidth_mbps * 1_000_000.0

    for task in order:
        node = assignment[task]
        est = node_ready[node]
        for p in preds.get(task, []):
            comm_ms = 0.0
            if assignment[p] != node:
                edge_key = f"{p}->{task}"
                size_bytes = float(edge_size_bytes.get(edge_key, 0.0))
                comm_ms = (size_bytes * 8.0 / bps) * 1000.0 + link_latency_ms
                total_comm_ms += comm_ms
                cross_edges += 1
                cross_bytes += size_bytes
            est = max(est, finish[p] + comm_ms)
        dur = float(task_cost_ms[task])
        finish_t = est + dur
        finish[task] = finish_t
        node_ready[node] = finish_t

    makespan = max(finish.values()) if finish else 0.0
    return DeploymentEstimate(
        makespan_ms=float(makespan),
        cross_node_edges=int(cross_edges),
        cross_node_bytes=float(cross_bytes),
        total_comm_ms=float(total_comm_ms),
    )


def build_deployment_plan(
    aggregated_profile_json: Path,
    num_nodes: int,
    strategy: str,
    bandwidth_mbps: float,
    link_latency_ms: float = 0.2,
) -> dict:
    data = load_aggregated_profile(aggregated_profile_json)
    tasks = list(data["dag_structure"]["nodes"])
    assignment = assign_tasks(tasks, num_nodes=num_nodes, strategy=strategy)
    estimate = estimate_schedule(
        dag_structure=data["dag_structure"],
        task_cost_ms=data["task_cost_ms"],
        edge_size_bytes=data["edge_size_bytes"],
        assignment=assignment,
        bandwidth_mbps=bandwidth_mbps,
        link_latency_ms=link_latency_ms,
    )

    node_task_order: dict[str, list[str]] = defaultdict(list)
    for t in data["dag_structure"]["execution_order"]:
        node_task_order[f"node{assignment[t]:02d}"].append(t)

    return {
        "profile_file": str(aggregated_profile_json),
        "phase": data.get("phase"),
        "num_nodes": num_nodes,
        "strategy": strategy,
        "bandwidth_mbps": bandwidth_mbps,
        "link_latency_ms": link_latency_ms,
        "assignment": assignment,
        "node_task_order": dict(node_task_order),
        "estimate": {
            "makespan_ms": estimate.makespan_ms,
            "cross_node_edges": estimate.cross_node_edges,
            "cross_node_bytes": estimate.cross_node_bytes,
            "total_comm_ms": estimate.total_comm_ms,
        },
    }


def save_plan(plan: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

