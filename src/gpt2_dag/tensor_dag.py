from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, create_causal_mask, eager_attention_forward


def _conv1d_output_slice(conv: nn.Module, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    # GPT-2 Conv1D uses weight shape [in, out].
    size_out = x.size()[:-1] + (end - start,)
    y = torch.addmm(conv.bias[start:end], x.reshape(-1, x.size(-1)), conv.weight[:, start:end])
    return y.view(size_out)


def _split_qkv_by_heads(attn_module: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_states, key_states, value_states = attn_module.c_attn(hidden_states).split(attn_module.split_size, dim=2)
    shape = (*query_states.shape[:-1], -1, attn_module.head_dim)
    query_states = query_states.view(shape).transpose(1, 2)
    key_states = key_states.view(shape).transpose(1, 2)
    value_states = value_states.view(shape).transpose(1, 2)
    return query_states, key_states, value_states


def _run_attention(
    attn_module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn_module.reorder_and_upcast_attn:
        return attn_module._upcast_and_reordered_attn(query_states, key_states, value_states, attention_mask)
    return eager_attention_forward(
        attn_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=attn_module.attn_dropout.p if attn_module.training else 0.0,
    )


def cache_to_tuples(cache: Cache | None) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if cache is None:
        return None
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            out.append((torch.empty(0), torch.empty(0)))
        else:
            out.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    return out


def tuples_to_cache(
    kv_tuples: list[tuple[torch.Tensor, torch.Tensor]] | None,
    config,
) -> DynamicCache | None:
    if kv_tuples is None:
        return None
    cache = DynamicCache(config=config)
    for layer_idx, (keys, values) in enumerate(kv_tuples):
        if keys.numel() == 0:
            continue
        cache.update(keys.clone(), values.clone(), layer_idx)
    return cache


@dataclass
class TensorLayerOutputs:
    hidden_states: torch.Tensor
    attn_weights: torch.Tensor | None


class GPT2TensorDAGLMHeadModel(GPT2LMHeadModel):
    """GPT-2 LM head model executed as explicit tensor-sharded per-layer DAG steps."""

    def __init__(self, config, shard_count: int = 12):
        super().__init__(config)
        if config.n_head % shard_count != 0:
            raise ValueError(f"n_head={config.n_head} must be divisible by shard_count={shard_count}")
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        if inner_dim % shard_count != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by shard_count={shard_count}")
        self.shard_count = int(shard_count)

    @classmethod
    def from_hf_model(cls, model: GPT2LMHeadModel, shard_count: int = 12) -> "GPT2TensorDAGLMHeadModel":
        cfg = copy.deepcopy(model.config)
        dag_model = cls(cfg, shard_count=shard_count)
        dag_model.load_state_dict(model.state_dict(), strict=True)
        dag_model.eval()
        return dag_model

    def _tensor_block_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
        output_attentions: bool,
    ) -> TensorLayerOutputs:
        block = self.transformer.h[layer_idx]
        residual = hidden_states
        ln1 = block.ln_1(hidden_states)

        query_states, key_states, value_states = _split_qkv_by_heads(block.attn, ln1)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                layer_idx,
                {"cache_position": cache_position},
            )

        heads_per_shard = block.attn.num_heads // self.shard_count
        ctx_parts = []
        attn_parts = []
        for shard_idx in range(self.shard_count):
            hs = shard_idx * heads_per_shard
            he = hs + heads_per_shard
            q_shard = query_states[:, hs:he, :, :]
            k_shard = key_states[:, hs:he, :, :]
            v_shard = value_states[:, hs:he, :, :]
            ctx_shard, attn_w_shard = _run_attention(block.attn, q_shard, k_shard, v_shard, causal_mask)
            ctx_parts.append(ctx_shard)
            if output_attentions:
                attn_parts.append(attn_w_shard)

        attn_ctx = torch.cat(ctx_parts, dim=2)
        attn_ctx = attn_ctx.reshape(*attn_ctx.shape[:-2], -1).contiguous()
        attn_out = block.attn.c_proj(attn_ctx)
        attn_out = block.attn.resid_dropout(attn_out)
        hidden_after_attn = residual + attn_out

        residual2 = hidden_after_attn
        ln2 = block.ln_2(hidden_after_attn)

        inner_dim = block.mlp.c_fc.nf
        mlp_chunk = inner_dim // self.shard_count
        mlp_parts = []
        for shard_idx in range(self.shard_count):
            start = shard_idx * mlp_chunk
            end = start + mlp_chunk
            shard_fc = _conv1d_output_slice(block.mlp.c_fc, ln2, start, end)
            shard_act = block.mlp.act(shard_fc)
            mlp_parts.append(shard_act)
        mlp_hidden = torch.cat(mlp_parts, dim=-1)
        ff = block.mlp.c_proj(mlp_hidden)
        ff = block.mlp.dropout(ff)
        hidden_out = residual2 + ff

        attn_weights = torch.cat(attn_parts, dim=1) if output_attentions else None
        return TensorLayerOutputs(hidden_states=hidden_out, attn_weights=attn_weights)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError("GPT2TensorDAGLMHeadModel currently supports decoder-only GPT-2 (no cross-attention).")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise ValueError("past_key_values must be a transformers Cache instance")

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if token_type_ids is not None:
            hidden_states = hidden_states + self.transformer.wte(token_type_ids)

        hidden_states = self.transformer.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer_idx in range(len(self.transformer.h)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = self._tensor_block_forward(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                past_key_values=past_key_values if use_cache else None,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs.attn_weights,)

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        past_key_values_out = past_key_values if use_cache else None
        if not return_dict:
            output = (logits, past_key_values_out, all_hidden_states, all_self_attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values_out,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )

