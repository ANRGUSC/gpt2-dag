from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gpt2_dag.dagprofiler_workflow import profile_phase
from gpt2_dag.export_dagbench import export_both


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--decode-context-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--shards", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--mirror-dagbench",
        default="C:\\Users\\bhask\\codex\\dagbench\\workflows",
    )
    args = p.parse_args()

    root = ROOT
    artifacts = root / "artifacts"
    profiles_dir = artifacts / "profiles"
    export_root = artifacts / "dagbench_workflows"

    prefill = profile_phase(
        phase="prefill",
        out_dir=profiles_dir,
        prompt_len=args.prompt_len,
        decode_context_len=args.decode_context_len,
        repeats=args.repeats,
        warmup=args.warmup,
        shard_count=args.shards,
        seed=args.seed,
    )
    decode = profile_phase(
        phase="decode",
        out_dir=profiles_dir,
        prompt_len=args.prompt_len,
        decode_context_len=args.decode_context_len,
        repeats=args.repeats,
        warmup=args.warmup,
        shard_count=args.shards,
        seed=args.seed,
    )

    generated = export_both(
        prefill_agg=prefill,
        decode_agg=decode,
        out_root=export_root,
        prompt_len=args.prompt_len,
        decode_context_len=args.decode_context_len,
    )

    mirror_root = Path(args.mirror_dagbench)
    for workflow_dir in generated:
        rel = workflow_dir.relative_to(export_root)
        target = mirror_root / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(workflow_dir, target)
            print(f"Mirrored {workflow_dir} -> {target}")
        except PermissionError:
            print(f"Mirror skipped (permission denied): {target}")


if __name__ == "__main__":
    main()
