# gpt2-dag

Tensor-DAG implementation of GPT-2 (Sh=12 shards/layer), with:
- exact-functionality parity tests against Hugging Face GPT-2 code
- measured compute and communication profiling via `dagprofiler`
- DAGBench workflow export (`prefill` and `decode`)

## What this repo guarantees

The DAG model (`GPT2TensorDAGLMHeadModel`) uses the same weights and the same GPT-2 block equations as Hugging Face GPT-2, but executes each layer through explicit tensor-sharded DAG steps:
- `qkv -> attn_shard_* -> attn_merge -> mlp_shard_* -> mlp_merge`

Parity tests verify:
- exact state_dict equality
- exact logits and cache equality on prefill and decode
- exact token-by-token greedy generation equivalence

## Run parity tests

```powershell
cd gpt2_dag_work
$env:PYTHONPATH = "src"
pytest -q tests/test_equivalence.py
```

## Run profiling + DAGBench export

```powershell
cd gpt2_dag_work
python run_pipeline.py --prompt-len 128 --decode-context-len 128 --repeats 9 --warmup 2 --shards 12
```

Outputs:
- `artifacts/profiles/gpt2_tensor_sh12_prefill_aggregated.json`
- `artifacts/profiles/gpt2_tensor_sh12_decode_aggregated.json`
- `artifacts/dagbench_workflows/ml_pipelines/gpt2_tensor_sh12_prefill/*`
- `artifacts/dagbench_workflows/ml_pipelines/gpt2_tensor_sh12_decode/*`

`run_pipeline.py` also mirrors generated workflows into local DAGBench:
- `C:\Users\bhask\codex\dagbench\workflows\ml_pipelines\gpt2_tensor_sh12_prefill`
- `C:\Users\bhask\codex\dagbench\workflows\ml_pipelines\gpt2_tensor_sh12_decode`

