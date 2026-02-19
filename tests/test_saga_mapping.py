from __future__ import annotations

from pathlib import Path

from gpt2_dag.deployment import build_deployment_plan


def test_build_saga_plan_from_example_network():
    root = Path(__file__).resolve().parent.parent
    profile = root / "artifacts" / "profiles" / "gpt2_tensor_sh12_decode_aggregated.json"
    network = root / "docs" / "examples" / "compute_network_example.yaml"

    plan = build_deployment_plan(
        aggregated_profile_json=profile,
        num_nodes=1,
        strategy="saga",
        bandwidth_mbps=100.0,
        network_file=network,
        saga_scheduler="heft",
    )

    assert plan["strategy"] == "saga"
    assert plan["saga_scheduler"] == "heft"
    assert plan["num_nodes"] == 4
    assert len(plan["assignment"]) > 0
    assert len(plan["node_task_order"]) == 4
    assert plan["estimate"]["makespan_ms"] > 0.0
    assert "runtime_contract" in plan

