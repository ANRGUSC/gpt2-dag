# Deployment Guide (Multi-Node Split)

This repo includes a concrete deployment planner to split the tensor DAG across multiple nodes.

## 1) Generate measured profiles first

```powershell
python run_pipeline.py --prompt-len 128 --decode-context-len 128 --repeats 9 --warmup 2 --shards 12
```

This creates:
- `artifacts/profiles/gpt2_tensor_sh12_prefill_aggregated.json`
- `artifacts/profiles/gpt2_tensor_sh12_decode_aggregated.json`

## 2) Build a node assignment + schedule estimate

### Decode phase on 8 nodes (tensor strategy)

```powershell
python scripts/deploy_plan.py `
  --profile artifacts/profiles/gpt2_tensor_sh12_decode_aggregated.json `
  --num-nodes 8 `
  --strategy tensor `
  --bandwidth-mbps 100 `
  --out artifacts/deployment/decode_k8_tensor.json
```

### Prefill phase on 8 nodes (pipeline strategy)

```powershell
python scripts/deploy_plan.py `
  --profile artifacts/profiles/gpt2_tensor_sh12_prefill_aggregated.json `
  --num-nodes 8 `
  --strategy pipeline `
  --bandwidth-mbps 100 `
  --out artifacts/deployment/prefill_k8_pipeline.json
```

## 3) What the plan file contains

Each output JSON includes:
- `assignment`: `task -> node_id`
- `node_task_order`: per-node ordered task list
- `estimate`:
  - `makespan_ms`
  - `cross_node_edges`
  - `cross_node_bytes`
  - `total_comm_ms`

Use this as the deployment contract for your runtime/orchestrator.

## Strategy definitions

- `pipeline`
  - All tasks in each layer are colocated on the same stage node.
  - Best for minimizing cross-node transfers, lower shard parallelism.

- `tensor`
  - Shard tasks (`attn_shard_*`, `mlp_shard_*`) are spread across nodes by shard id.
  - Merge/qkv tasks stay stage-local.
  - Higher parallelism, potentially higher communication.

