# Deployment Guide (Multi-Node Split + SAGA Mapping)

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

### Decode phase with SAGA + network input (HEFT)

```powershell
python scripts/deploy_plan.py `
  --profile artifacts/profiles/gpt2_tensor_sh12_decode_aggregated.json `
  --strategy saga `
  --network-file docs/examples/compute_network_example.yaml `
  --scheduler heft `
  --out artifacts/deployment/decode_saga_heft.json
```

This is the full flow:

- `gpt2-dag` -> DAG tasks + measured costs (via `dagprofiler`)
- measured DAG + `compute_network_example.yaml` -> `SAGA` scheduler
- output mapping and schedule:
  - task-to-node assignment
  - per-node ordered task list
  - per-node scheduled start/end times

## 3) What the plan file contains

Each output JSON includes:
- `assignment`: `task -> node_name`
- `node_task_order`: per-node ordered task list
- `detailed_schedule` (SAGA strategy): start/end for each task on each node
- `estimate`:
  - `makespan_ms`
  - `cross_node_edges`
  - `cross_node_bytes`
  - `total_comm_ms`
- `runtime_contract`:
  - shared code files to ship to each node
  - worker entrypoint command
  - per-node assigned task list

Use this as the deployment contract for your runtime/orchestrator.

## Strategy definitions

- `pipeline`
  - All tasks in each layer are colocated on the same stage node.
  - Best for minimizing cross-node transfers, lower shard parallelism.

- `tensor`
  - Shard tasks (`attn_shard_*`, `mlp_shard_*`) are spread across nodes by shard id.
  - Merge/qkv tasks stay stage-local.
  - Higher parallelism, potentially higher communication.

- `saga`
  - Converts measured DAG + network input file into a SAGA scheduling problem.
  - Uses selected SAGA scheduler to compute mapping and timing.
  - Network-aware by construction.

## Compute-network input file

Example input:
- `docs/examples/compute_network_example.yaml`

Schema:
- `nodes`:
  - `name`: node id string
  - `speed`: relative compute factor for measured task costs
- `default_bandwidth_mbps`: default inter-node bandwidth
- `links` (optional overrides):
  - `source`, `target`
  - `bandwidth_mbps` or `speed_bytes_per_ms`

Notes:
- Task costs are currently measured in ms; node `speed` is treated as a relative multiplier.
- Link bandwidth is converted to bytes/ms internally for SAGA edge timing.

## How code is shipped and executed across nodes

This repo uses a shared-code + mapped-tasks model:

- Every node gets the same package (`gpt2_dag`).
- The deployment plan decides which task names each node runs.
- Worker process uses:
  - `python -m gpt2_dag.worker_runtime --plan <plan.json> --node <node_name>`

So tasks are not separate per-file binaries; they are dispatched by task name from one shared runtime package.
