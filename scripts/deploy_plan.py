from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpt2_dag.deployment import build_deployment_plan, save_plan


def main() -> None:
    p = argparse.ArgumentParser(description="Create a multi-node deployment plan for GPT-2 tensor DAG.")
    p.add_argument("--profile", required=True, help="Path to aggregated profile JSON")
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--strategy", choices=["tensor", "pipeline"], default="tensor")
    p.add_argument("--bandwidth-mbps", type=float, default=100.0)
    p.add_argument("--link-latency-ms", type=float, default=0.2)
    p.add_argument("--out", required=True, help="Output plan JSON path")
    args = p.parse_args()

    plan = build_deployment_plan(
        aggregated_profile_json=Path(args.profile),
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        bandwidth_mbps=args.bandwidth_mbps,
        link_latency_ms=args.link_latency_ms,
    )
    save_plan(plan, Path(args.out))

    est = plan["estimate"]
    print(
        json.dumps(
            {
                "out": args.out,
                "phase": plan["phase"],
                "num_nodes": plan["num_nodes"],
                "strategy": plan["strategy"],
                "makespan_ms": est["makespan_ms"],
                "cross_node_edges": est["cross_node_edges"],
                "cross_node_bytes": est["cross_node_bytes"],
                "total_comm_ms": est["total_comm_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

