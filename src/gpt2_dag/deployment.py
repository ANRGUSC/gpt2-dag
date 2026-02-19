from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from saga import Network, TaskGraph
from saga.schedulers.bil import BILScheduler
from saga.schedulers.cpop import CpopScheduler
from saga.schedulers.etf import ETFScheduler
from saga.schedulers.heft import HeftScheduler
from saga.schedulers.met import METScheduler
from saga.schedulers.minmin import MinMinScheduler


LAYER_PAT = re.compile(r"_(\d{2})")
SHARD_PAT = re.compile(r"_(\d{2})_(\d+)$")


@dataclass
class DeploymentEstimate:
    makespan_ms: float
    cross_node_edges: int
    cross_node_bytes: float
    total_comm_ms: float


SAGA_SCHEDULERS = {
    "heft": HeftScheduler,
    "cpop": CpopScheduler,
    "etf": ETFScheduler,
    "minmin": MinMinScheduler,
    "met": METScheduler,
    "bil": BILScheduler,
}


def _load_data_file(path: Path) -> dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(txt)
    return json.loads(txt)


def load_aggregated_profile(path: Path) -> dict:
    return _load_data_file(path)


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


def _edge_map(dag_structure: dict) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = defaultdict(list)
    for e in dag_structure["edges"]:
        preds[e["target"]].append(e["source"])
    return preds


def _cross_node_stats(assignment: dict[str, str], edge_size_bytes: dict[str, float], dag_structure: dict) -> tuple[int, float]:
    cross_edges = 0
    cross_bytes = 0.0
    for e in dag_structure["edges"]:
        src, tgt = e["source"], e["target"]
        if assignment[src] != assignment[tgt]:
            cross_edges += 1
            cross_bytes += float(edge_size_bytes.get(f"{src}->{tgt}", 0.0))
    return cross_edges, cross_bytes


def estimate_schedule(
    dag_structure: dict,
    task_cost_ms: dict[str, float],
    edge_size_bytes: dict[str, float],
    assignment: dict[str, int],
    bandwidth_mbps: float = 100.0,
    link_latency_ms: float = 0.2,
) -> DeploymentEstimate:
    order = dag_structure["execution_order"]
    preds = _edge_map(dag_structure)

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


def _mbps_to_bytes_per_ms(bandwidth_mbps: float) -> float:
    return bandwidth_mbps * 1_000_000.0 / 8.0 / 1000.0


def _load_network_for_saga(path: Path) -> tuple[Network, dict]:
    raw = _load_data_file(path)
    if "nodes" not in raw:
        raise ValueError("Network file must contain a 'nodes' list")
    node_entries = raw["nodes"]
    if not node_entries:
        raise ValueError("Network file nodes list is empty")

    nodes: list[tuple[str, float]] = []
    node_names: list[str] = []
    for n in node_entries:
        if "name" not in n:
            raise ValueError("Each node must have 'name'")
        name = str(n["name"])
        speed = float(n.get("speed", n.get("speed_factor", 1.0)))
        if speed <= 0:
            raise ValueError(f"Node {name} speed must be > 0")
        nodes.append((name, speed))
        node_names.append(name)

    default_bandwidth_mbps = float(raw.get("default_bandwidth_mbps", 100.0))
    default_speed = _mbps_to_bytes_per_ms(default_bandwidth_mbps)
    link_map: dict[tuple[str, str], float] = {}
    for link in raw.get("links", []):
        src = str(link["source"])
        dst = str(link["target"])
        key = tuple(sorted((src, dst)))
        if "speed_bytes_per_ms" in link:
            speed = float(link["speed_bytes_per_ms"])
        else:
            speed = _mbps_to_bytes_per_ms(float(link.get("bandwidth_mbps", default_bandwidth_mbps)))
        if speed <= 0:
            raise ValueError(f"Link {src}<->{dst} speed must be > 0")
        link_map[key] = speed

    edges: list[tuple[str, str, float]] = []
    for i, src in enumerate(node_names):
        for dst in node_names[i:]:
            if src == dst:
                edges.append((src, dst, 1e12))
                continue
            speed = link_map.get(tuple(sorted((src, dst))), default_speed)
            edges.append((src, dst, speed))

    summary = {
        "nodes": [{"name": n, "speed": s} for n, s in nodes],
        "default_bandwidth_mbps": default_bandwidth_mbps,
        "links": [{"source": s, "target": t, "speed_bytes_per_ms": v} for s, t, v in edges if s != t],
    }
    return Network.create(nodes=nodes, edges=edges), summary


def _build_task_graph_for_saga(dag_structure: dict, task_cost_ms: dict[str, float], edge_size_bytes: dict[str, float]) -> TaskGraph:
    tasks = [(t, float(task_cost_ms[t])) for t in dag_structure["nodes"]]
    deps = []
    for e in dag_structure["edges"]:
        key = f"{e['source']}->{e['target']}"
        deps.append((e["source"], e["target"], float(edge_size_bytes.get(key, 0.0))))
    return TaskGraph.create(tasks=tasks, dependencies=deps)


def _scheduler_from_name(name: str):
    key = name.strip().lower()
    if key not in SAGA_SCHEDULERS:
        raise ValueError(f"Unsupported SAGA scheduler '{name}'. Supported: {sorted(SAGA_SCHEDULERS.keys())}")
    return SAGA_SCHEDULERS[key]()


def build_saga_mapping(
    dag_structure: dict,
    task_cost_ms: dict[str, float],
    edge_size_bytes: dict[str, float],
    network_file: Path,
    scheduler_name: str = "heft",
) -> dict[str, Any]:
    task_graph = _build_task_graph_for_saga(dag_structure, task_cost_ms, edge_size_bytes)
    network, network_summary = _load_network_for_saga(network_file)
    scheduler = _scheduler_from_name(scheduler_name)
    schedule = scheduler.schedule(network, task_graph)

    task_set = set(dag_structure["nodes"])
    assignment: dict[str, str] = {}
    node_task_order: dict[str, list[str]] = defaultdict(list)
    detailed_schedule: dict[str, list[dict[str, float | str]]] = {}

    for node_name, scheduled in schedule.items():
        valid = [st for st in scheduled if st.name in task_set]
        valid.sort(key=lambda st: st.start)
        node_task_order[node_name] = [st.name for st in valid]
        detailed_schedule[node_name] = [
            {"task": st.name, "start_ms": float(st.start), "end_ms": float(st.end)} for st in valid
        ]
        for st in valid:
            assignment[st.name] = node_name

    missing = task_set - set(assignment.keys())
    if missing:
        raise RuntimeError(f"SAGA schedule missing tasks: {sorted(missing)}")

    cross_edges, cross_bytes = _cross_node_stats(assignment, edge_size_bytes, dag_structure)
    return {
        "scheduler": scheduler_name,
        "network_file": str(network_file),
        "network_summary": network_summary,
        "assignment": assignment,
        "node_task_order": dict(node_task_order),
        "detailed_schedule": detailed_schedule,
        "estimate": {
            "makespan_ms": float(schedule.makespan),
            "cross_node_edges": int(cross_edges),
            "cross_node_bytes": float(cross_bytes),
            "total_comm_ms": None,
        },
    }


def _runtime_contract(node_task_order: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "shared_code_package": [
            "src/gpt2_dag/tensor_dag.py",
            "src/gpt2_dag/dagprofiler_workflow.py",
            "src/gpt2_dag/worker_runtime.py",
        ],
        "worker_entrypoint": "python -m gpt2_dag.worker_runtime --plan <plan.json> --node <node_name>",
        "per_node_tasks": node_task_order,
    }


def build_deployment_plan(
    aggregated_profile_json: Path,
    num_nodes: int,
    strategy: str,
    bandwidth_mbps: float,
    link_latency_ms: float = 0.2,
    network_file: Path | None = None,
    saga_scheduler: str = "heft",
) -> dict:
    data = load_aggregated_profile(aggregated_profile_json)
    tasks = list(data["dag_structure"]["nodes"])

    if strategy == "saga":
        if network_file is None:
            raise ValueError("network_file is required when strategy='saga'")
        saga_plan = build_saga_mapping(
            dag_structure=data["dag_structure"],
            task_cost_ms=data["task_cost_ms"],
            edge_size_bytes=data["edge_size_bytes"],
            network_file=network_file,
            scheduler_name=saga_scheduler,
        )
        return {
            "profile_file": str(aggregated_profile_json),
            "phase": data.get("phase"),
            "strategy": "saga",
            "num_nodes": len(saga_plan["network_summary"]["nodes"]),
            "saga_scheduler": saga_plan["scheduler"],
            "network_file": saga_plan["network_file"],
            "assignment": saga_plan["assignment"],
            "node_task_order": saga_plan["node_task_order"],
            "detailed_schedule": saga_plan["detailed_schedule"],
            "estimate": saga_plan["estimate"],
            "network_summary": saga_plan["network_summary"],
            "runtime_contract": _runtime_contract(saga_plan["node_task_order"]),
        }

    assignment_int = assign_tasks(tasks, num_nodes=num_nodes, strategy=strategy)
    estimate = estimate_schedule(
        dag_structure=data["dag_structure"],
        task_cost_ms=data["task_cost_ms"],
        edge_size_bytes=data["edge_size_bytes"],
        assignment=assignment_int,
        bandwidth_mbps=bandwidth_mbps,
        link_latency_ms=link_latency_ms,
    )

    node_task_order: dict[str, list[str]] = defaultdict(list)
    assignment: dict[str, str] = {}
    for t in data["dag_structure"]["execution_order"]:
        node_name = f"node{assignment_int[t]:02d}"
        node_task_order[node_name].append(t)
        assignment[t] = node_name

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
        "runtime_contract": _runtime_contract(dict(node_task_order)),
    }


def save_plan(plan: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

