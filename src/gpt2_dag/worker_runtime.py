from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_plan(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Worker runtime placeholder for distributed GPT-2 DAG execution.")
    p.add_argument("--plan", required=True, help="Deployment plan JSON path")
    p.add_argument("--node", required=True, help="Node name (e.g., node00 or rpi-a)")
    p.add_argument("--dry-run", action="store_true", help="Print assigned tasks and exit")
    args = p.parse_args()

    plan = _load_plan(Path(args.plan))
    tasks = plan.get("node_task_order", {}).get(args.node, [])
    if args.dry_run or True:
        print(json.dumps({"node": args.node, "num_tasks": len(tasks), "tasks": tasks}, indent=2))
        return


if __name__ == "__main__":
    main()

