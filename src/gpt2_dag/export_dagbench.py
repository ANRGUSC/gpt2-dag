from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import networkx as nx
import yaml


def _graph_stats(tasks: list[dict], deps: list[dict]) -> dict:
    g = nx.DiGraph()
    for task in tasks:
        g.add_node(task["name"])
    for dep in deps:
        g.add_edge(dep["source"], dep["target"])

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Exported graph is not a DAG")

    depth = nx.dag_longest_path_length(g) + 1 if len(g) > 0 else 0
    levels: dict[str, int] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
    width = 0
    if levels:
        counts: dict[int, int] = {}
        for lv in levels.values():
            counts[lv] = counts.get(lv, 0) + 1
        width = max(counts.values())

    total_comp = sum(float(t["cost"]) for t in tasks)
    total_comm = sum(float(d["size"]) for d in deps)
    ccr = (total_comm / total_comp) if total_comp > 0 else None
    parallelism = (len(tasks) / depth) if depth > 0 else None
    return {
        "num_tasks": len(tasks),
        "num_edges": len(deps),
        "depth": int(depth),
        "width": int(width),
        "ccr": None if ccr is None else float(round(ccr, 6)),
        "parallelism": None if parallelism is None else float(round(parallelism, 6)),
    }


def _default_network(num_nodes: int = 12) -> dict:
    nodes = [{"name": f"N{i}", "speed": 1.0} for i in range(num_nodes)]
    edges = []
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            speed = 1_000_000_000.0 if i == j else 500.0
            edges.append({"source": f"N{i}", "target": f"N{j}", "speed": speed})
    return {"nodes": nodes, "edges": edges}


def export_single_workflow(
    aggregated_profile_json: Path,
    out_dir: Path,
    workflow_id: str,
    workflow_name: str,
    phase_label: str,
    prompt_len: int,
    decode_context_len: int,
) -> Path:
    data = json.loads(aggregated_profile_json.read_text(encoding="utf-8"))

    nodes = data["dag_structure"]["nodes"]
    edges = data["dag_structure"]["edges"]
    task_cost_ms = data["task_cost_ms"]
    edge_size_bytes = data["edge_size_bytes"]

    tasks = [{"name": n, "cost": float(task_cost_ms[n])} for n in nodes]
    deps = []
    for e in edges:
        key = f"{e['source']}->{e['target']}"
        deps.append(
            {
                "source": e["source"],
                "target": e["target"],
                "size": float(edge_size_bytes[key]),
            }
        )

    graph_payload = {
        "name": workflow_id,
        "task_graph": {
            "tasks": tasks,
            "dependencies": deps,
        },
        "network": _default_network(num_nodes=12),
    }

    stats = _graph_stats(tasks, deps)
    if phase_label == "prefill":
        phase_description = (
            "Prefill phase for GPT-2 request handling: run once on the full prompt to build KV cache. "
            f"This workflow profiles q_len={prompt_len}, kv_len={prompt_len}."
        )
        usage_notes = (
            "Prefill and decode are complementary workflows for one request. "
            "Use prefill exactly once per prompt to initialize KV cache. "
            "Then run decode repeatedly for token generation steps. "
            "Total latency model used with this pair is: prefill_once + L_g * decode_step. "
            f"For decode profiling, avg_kv is set to L_p + L_g/2 (here L_p={prompt_len}, L_g={decode_context_len})."
        )
    else:
        phase_description = (
            "Decode phase for GPT-2 request handling: generate one token using existing KV cache. "
            "This workflow is intended to be repeated for each generated token."
        )
        usage_notes = (
            "Prefill and decode are complementary workflows for one request. "
            "Run prefill once for the prompt, then run decode for each generation step (up to L_g steps, early stop on EOS). "
            "Total latency model used with this pair is: prefill_once + L_g * decode_step. "
            f"Decode costs here are profiled at avg_kv = L_p + L_g/2 with L_p={prompt_len}, L_g={decode_context_len}."
        )

    metadata = {
        "id": workflow_id,
        "name": workflow_name,
        "description": (
            f"{phase_description} GPT-2 tensor DAG with Sh=12 shards per transformer layer. "
            "Task costs are measured compute_time_ms from dagprofiler; dependency sizes are measured bytes."
        ),
        "domains": ["ml-pipeline", "edge-computing"],
        "provenance": {
            "source": "GPT-2 Hugging Face implementation profiled via dagprofiler",
            "paper_title": "GPT-2 Tensor DAG Sharding (measured trace conversion)",
            "repo_url": "https://github.com/huggingface/transformers",
            "extraction_method": "trace-conversion",
            "extractor": "codex-gpt5",
            "extraction_date": str(date.today()),
            "notes": (
                "Derived from local analysis of Hugging Face GPT-2 code using dagprofiler. "
                "DAG implementation repository: https://github.com/ANRGUSC/gpt2-dag. "
                "Profiler repository: https://github.com/ANRGUSC/dagprofiler. "
                f"phase={phase_label}, prompt_len={prompt_len}, decode_context_len={decode_context_len}, shard_count=12. "
                f"{usage_notes}"
            ),
        },
        "license": {
            "source_license": "Apache-2.0",
            "dagbench_license": "Apache-2.0",
            "notes": "Workflow representation generated from measured local profiling traces.",
        },
        "completeness": "full",
        "cost_model": "deterministic",
        "network": {
            "included": True,
            "topology": "fully-connected",
            "num_nodes_min": 12,
            "num_nodes_max": 12,
        },
        "graph_stats": stats,
        "campaign": "campaign_006_gpt2_tensor_trace",
        "tags": [
            "gpt2",
            "tensor-parallel",
            "sh12",
            "dagprofiler",
            "trace-conversion",
            "gpt2-dag",
            phase_label,
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.json"
    meta_path = out_dir / "metadata.yaml"
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return out_dir


def export_both(
    prefill_agg: Path,
    decode_agg: Path,
    out_root: Path,
    prompt_len: int,
    decode_context_len: int,
) -> list[Path]:
    prefill_dir = out_root / "ml_pipelines" / "gpt2_tensor_sh12_prefill"
    decode_dir = out_root / "ml_pipelines" / "gpt2_tensor_sh12_decode"
    paths = []
    paths.append(
        export_single_workflow(
            aggregated_profile_json=prefill_agg,
            out_dir=prefill_dir,
            workflow_id="ml.gpt2_tensor_sh12_prefill",
            workflow_name="GPT-2 Tensor DAG Sh12 Prefill",
            phase_label="prefill",
            prompt_len=prompt_len,
            decode_context_len=decode_context_len,
        )
    )
    paths.append(
        export_single_workflow(
            aggregated_profile_json=decode_agg,
            out_dir=decode_dir,
            workflow_id="ml.gpt2_tensor_sh12_decode",
            workflow_name="GPT-2 Tensor DAG Sh12 Decode",
            phase_label="decode",
            prompt_len=prompt_len,
            decode_context_len=decode_context_len,
        )
    )
    return paths
