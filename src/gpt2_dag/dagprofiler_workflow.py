from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dagprofiler import Config, DAG, Task
from transformers import GPT2Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gpt2.modeling_gpt2 import create_causal_mask

from .tensor_dag import GPT2TensorDAGLMHeadModel, _conv1d_output_slice, _run_attention, _split_qkv_by_heads, cache_to_tuples, tuples_to_cache


@dataclass
class RuntimeContext:
    model: GPT2TensorDAGLMHeadModel
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor | None
    past_key_values: Cache | None
    shard_count: int
    use_cache: bool
    cache_position: torch.LongTensor | None = None
    causal_mask: torch.Tensor | None = None


def _build_runtime(
    model: GPT2TensorDAGLMHeadModel,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    shard_count: int,
    use_cache: bool = True,
) -> RuntimeContext:
    return RuntimeContext(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        shard_count=shard_count,
        use_cache=use_cache,
    )


def _build_dag(runtime: RuntimeContext) -> DAG:
    model = runtime.model
    n_layer = model.config.n_layer
    shards = runtime.shard_count
    heads_per_shard = model.config.n_head // shards
    inner_dim = model.config.n_inner if model.config.n_inner is not None else 4 * model.config.n_embd
    mlp_chunk = inner_dim // shards

    class EmbedTask(Task):
        inputs = []
        outputs = ["h"]

        def compute(self, cfg=None):
            if runtime.past_key_values is None and runtime.use_cache:
                runtime.past_key_values = DynamicCache(config=model.config)
            inputs_embeds = model.transformer.wte(runtime.input_ids)
            past_seen = runtime.past_key_values.get_seq_length() if runtime.past_key_values is not None else 0
            runtime.cache_position = torch.arange(
                past_seen,
                past_seen + runtime.input_ids.shape[1],
                device=runtime.input_ids.device,
            )
            position_ids = runtime.cache_position.unsqueeze(0)
            position_embeds = model.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
            runtime.causal_mask = create_causal_mask(
                config=model.config,
                input_embeds=inputs_embeds,
                attention_mask=runtime.attention_mask,
                cache_position=runtime.cache_position,
                past_key_values=runtime.past_key_values,
                position_ids=position_ids,
            )
            hidden_states = model.transformer.drop(hidden_states)
            return {"h": hidden_states}

    def make_qkv_task(layer_idx: int):
        class QKVTask(Task):
            inputs = ["h"]
            outputs = ["residual"] + [f"q{s}" for s in range(shards)] + [f"k{s}" for s in range(shards)] + [f"v{s}" for s in range(shards)]

            def compute(self, h, cfg=None):
                block = model.transformer.h[layer_idx]
                residual = h
                ln1 = block.ln_1(h)
                q, k, v = _split_qkv_by_heads(block.attn, ln1)
                if runtime.use_cache and runtime.past_key_values is not None:
                    k, v = runtime.past_key_values.update(
                        k,
                        v,
                        layer_idx,
                        {"cache_position": runtime.cache_position},
                    )
                out: dict[str, Any] = {"residual": residual}
                for s in range(shards):
                    hs = s * heads_per_shard
                    he = hs + heads_per_shard
                    out[f"q{s}"] = q[:, hs:he, :, :]
                    out[f"k{s}"] = k[:, hs:he, :, :]
                    out[f"v{s}"] = v[:, hs:he, :, :]
                return out

        return QKVTask

    def make_attn_shard_task(layer_idx: int, shard_idx: int):
        class AttnShardTask(Task):
            inputs = [f"q{shard_idx}", f"k{shard_idx}", f"v{shard_idx}"]
            outputs = [f"ctx{shard_idx}"]

            def compute(self, **kwargs):
                block = model.transformer.h[layer_idx]
                q = kwargs[f"q{shard_idx}"]
                k = kwargs[f"k{shard_idx}"]
                v = kwargs[f"v{shard_idx}"]
                ctx, _ = _run_attention(block.attn, q, k, v, runtime.causal_mask)
                return {f"ctx{shard_idx}": ctx}

        return AttnShardTask

    def make_attn_merge_task(layer_idx: int):
        class AttnMergeTask(Task):
            inputs = ["residual"] + [f"ctx{s}" for s in range(shards)]
            outputs = ["residual2", "ln2"]

            def compute(self, residual, **kwargs):
                block = model.transformer.h[layer_idx]
                ctx_parts = [kwargs[f"ctx{s}"] for s in range(shards)]
                attn_ctx = torch.cat(ctx_parts, dim=2)
                attn_ctx = attn_ctx.reshape(*attn_ctx.shape[:-2], -1).contiguous()
                attn_out = block.attn.c_proj(attn_ctx)
                attn_out = block.attn.resid_dropout(attn_out)
                hidden_after_attn = residual + attn_out
                ln2 = block.ln_2(hidden_after_attn)
                return {"residual2": hidden_after_attn, "ln2": ln2}

        return AttnMergeTask

    def make_mlp_shard_task(layer_idx: int, shard_idx: int):
        class MLPShardTask(Task):
            inputs = ["ln2"]
            outputs = [f"hact{shard_idx}"]

            def compute(self, ln2, cfg=None):
                block = model.transformer.h[layer_idx]
                start = shard_idx * mlp_chunk
                end = start + mlp_chunk
                h_chunk = _conv1d_output_slice(block.mlp.c_fc, ln2, start, end)
                h_chunk = block.mlp.act(h_chunk)
                return {f"hact{shard_idx}": h_chunk}

        return MLPShardTask

    def make_mlp_merge_task(layer_idx: int):
        class MLPMergeTask(Task):
            inputs = ["residual2"] + [f"hact{s}" for s in range(shards)]
            outputs = ["h"]

            def compute(self, residual2, **kwargs):
                block = model.transformer.h[layer_idx]
                parts = [kwargs[f"hact{s}"] for s in range(shards)]
                mlp_hidden = torch.cat(parts, dim=-1)
                ff = block.mlp.c_proj(mlp_hidden)
                ff = block.mlp.dropout(ff)
                h = residual2 + ff
                return {"h": h}

        return MLPMergeTask

    class LnFTask(Task):
        inputs = ["h"]
        outputs = ["h"]

        def compute(self, h, cfg=None):
            return {"h": model.transformer.ln_f(h)}

    class LMHeadTask(Task):
        inputs = ["h"]
        outputs = ["logits"]

        def compute(self, h, cfg=None):
            return {"logits": model.lm_head(h)}

    tasks: dict[str, Task] = {"embed": EmbedTask(name="embed")}
    edges: list[tuple] = []

    prev = "embed"
    for layer_idx in range(n_layer):
        qkv_name = f"qkv_{layer_idx:02d}"
        tasks[qkv_name] = make_qkv_task(layer_idx)(name=qkv_name)
        edges.append((prev, qkv_name, ["h"]))

        attn_merge_name = f"attn_merge_{layer_idx:02d}"
        tasks[attn_merge_name] = make_attn_merge_task(layer_idx)(name=attn_merge_name)
        edges.append((qkv_name, attn_merge_name, ["residual"]))

        for shard_idx in range(shards):
            attn_shard_name = f"attn_shard_{layer_idx:02d}_{shard_idx}"
            tasks[attn_shard_name] = make_attn_shard_task(layer_idx, shard_idx)(name=attn_shard_name)
            edges.append((qkv_name, attn_shard_name, [f"q{shard_idx}", f"k{shard_idx}", f"v{shard_idx}"]))
            edges.append((attn_shard_name, attn_merge_name, [f"ctx{shard_idx}"]))

        mlp_merge_name = f"mlp_merge_{layer_idx:02d}"
        tasks[mlp_merge_name] = make_mlp_merge_task(layer_idx)(name=mlp_merge_name)
        edges.append((attn_merge_name, mlp_merge_name, ["residual2"]))

        for shard_idx in range(shards):
            mlp_shard_name = f"mlp_shard_{layer_idx:02d}_{shard_idx}"
            tasks[mlp_shard_name] = make_mlp_shard_task(layer_idx, shard_idx)(name=mlp_shard_name)
            edges.append((attn_merge_name, mlp_shard_name, ["ln2"]))
            edges.append((mlp_shard_name, mlp_merge_name, [f"hact{shard_idx}"]))

        prev = mlp_merge_name

    tasks["ln_f"] = LnFTask(name="ln_f")
    tasks["lm_head"] = LMHeadTask(name="lm_head")
    edges.append((prev, "ln_f", ["h"]))
    edges.append(("ln_f", "lm_head", ["h"]))

    return DAG(tasks=tasks, edges=edges, config=Config(phase="gpt2_tensor_sh12"))


def _aggregate_profiles(profiles: list) -> dict[str, Any]:
    first = profiles[0]
    task_names = [m.task for m in first.metrics.task_metrics]
    edge_names = list(first.metrics.edge_weights.keys())

    task_cost_ms: dict[str, float] = {}
    for task in task_names:
        vals = []
        for prof in profiles:
            for m in prof.metrics.task_metrics:
                if m.task == task:
                    vals.append(m.compute_time_ms)
                    break
        task_cost_ms[task] = float(statistics.median(vals))

    edge_size_bytes: dict[str, float] = {}
    for edge in edge_names:
        vals = [prof.metrics.edge_weights[edge] / 8.0 for prof in profiles]
        edge_size_bytes[edge] = float(statistics.median(vals))

    return {
        "dag_structure": first.dag_structure,
        "task_cost_ms": task_cost_ms,
        "edge_size_bytes": edge_size_bytes,
        "num_runs": len(profiles),
    }


def profile_phase(
    phase: str,
    out_dir: Path,
    prompt_len: int = 128,
    decode_context_len: int = 128,
    repeats: int = 9,
    warmup: int = 2,
    shard_count: int = 12,
    seed: int = 7,
) -> Path:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"Unsupported phase: {phase}")

    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    cfg = GPT2Config()
    cfg._attn_implementation = "eager"
    model = GPT2TensorDAGLMHeadModel(cfg, shard_count=shard_count)
    model.eval()

    batch = 1
    if phase == "prefill":
        input_ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), dtype=torch.long)
        attention_mask = torch.ones((batch, prompt_len), dtype=torch.float32)
        kv_tuples = None
    else:
        prefill_ids = torch.randint(0, cfg.vocab_size, (batch, decode_context_len), dtype=torch.long)
        prefill_out = model(
            input_ids=prefill_ids,
            use_cache=True,
            return_dict=True,
        )
        kv_tuples = cache_to_tuples(prefill_out.past_key_values)
        input_ids = torch.randint(0, cfg.vocab_size, (batch, 1), dtype=torch.long)
        attention_mask = torch.ones((batch, decode_context_len + 1), dtype=torch.float32)

    profiles = []
    total_runs = warmup + repeats
    for run_idx in range(total_runs):
        cache = tuples_to_cache(kv_tuples, model.config) if kv_tuples is not None else DynamicCache(config=model.config)
        runtime = _build_runtime(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            shard_count=shard_count,
            use_cache=True,
        )
        dag = _build_dag(runtime)
        profile = dag.run(seed=seed + run_idx)
        if run_idx >= warmup:
            profiles.append(profile)

    agg = _aggregate_profiles(profiles)
    agg["phase"] = phase
    agg["prompt_len"] = prompt_len
    agg["decode_context_len"] = decode_context_len
    agg["shard_count"] = shard_count
    agg["model"] = "gpt2-small"
    agg["hf_repo"] = "https://github.com/huggingface/transformers"
    agg["dagprofiler_repo"] = "https://github.com/ANRGUSC/dagprofiler"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gpt2_tensor_sh12_{phase}_aggregated.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return out_path

