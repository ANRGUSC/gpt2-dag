# Architecture Guide

## Which files are the actual DAG-ized GPT-2 code?

These are the core Python files:

- `src/gpt2_dag/tensor_dag.py`
  - Defines `GPT2TensorDAGLMHeadModel`.
  - This is the DAG-ized GPT-2 model implementation.
  - Each transformer layer is executed as:
    - `qkv_l`
    - `attn_shard_l_0..11`
    - `attn_merge_l`
    - `mlp_shard_l_0..11`
    - `mlp_merge_l`

- `src/gpt2_dag/dagprofiler_workflow.py`
  - Defines the explicit `dagprofiler` task graph with per-task execution and measured costs.
  - Produces `prefill` and `decode` aggregated profiles.

- `src/gpt2_dag/export_dagbench.py`
  - Converts aggregated profiles into DAGBench workflow format (`graph.json` + `metadata.yaml`).

## DAG Figures

- Layer-level DAG:
  - `docs/figures/tensor_dag_layer_sh12.svg`
- Full 12-layer overview:
  - `docs/figures/tensor_dag_full_overview.svg`

To regenerate:

```powershell
python scripts/render_dag_figures.py
```

