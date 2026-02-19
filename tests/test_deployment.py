from __future__ import annotations

import json
from pathlib import Path

from gpt2_dag.deployment import build_deployment_plan


def test_build_deployment_plan_decode_profile():
    root = Path(__file__).resolve().parent.parent
    profile = root / "artifacts" / "profiles" / "gpt2_tensor_sh12_decode_aggregated.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    plan = build_deployment_plan(
        aggregated_profile_json=profile,
        num_nodes=8,
        strategy="tensor",
        bandwidth_mbps=100.0,
        link_latency_ms=0.2,
    )
    assert plan["phase"] == "decode"
    assert plan["num_nodes"] == 8
    assert plan["strategy"] == "tensor"
    assert len(plan["assignment"]) == len(data["dag_structure"]["nodes"])
    assert plan["estimate"]["makespan_ms"] > 0.0
    assert plan["estimate"]["cross_node_edges"] >= 0

