from __future__ import annotations

import copy

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2_dag.tensor_dag import GPT2TensorDAGLMHeadModel, cache_to_tuples


def _build_pair(seed: int = 11):
    torch.manual_seed(seed)
    cfg = GPT2Config()
    cfg._attn_implementation = "eager"
    hf = GPT2LMHeadModel(cfg).eval()
    dag = GPT2TensorDAGLMHeadModel(copy.deepcopy(cfg), shard_count=12).eval()
    dag.load_state_dict(hf.state_dict(), strict=True)
    return hf, dag


def _assert_cache_equal(hf_cache, dag_cache):
    hf_t = cache_to_tuples(hf_cache)
    dag_t = cache_to_tuples(dag_cache)
    assert hf_t is not None and dag_t is not None
    assert len(hf_t) == len(dag_t)
    for (hk, hv), (dk, dv) in zip(hf_t, dag_t):
        assert torch.equal(hk, dk)
        assert torch.equal(hv, dv)


def test_state_dict_exact_match():
    hf, dag = _build_pair(seed=1)
    hf_sd = hf.state_dict()
    dag_sd = dag.state_dict()
    assert hf_sd.keys() == dag_sd.keys()
    for name in hf_sd:
        assert torch.equal(hf_sd[name], dag_sd[name]), name


def test_prefill_forward_and_cache_exact_match():
    hf, dag = _build_pair(seed=2)
    input_ids = torch.randint(0, hf.config.vocab_size, (1, 32), dtype=torch.long)
    attention_mask = torch.ones((1, 32), dtype=torch.float32)

    out_hf = hf(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    out_dag = dag(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

    assert torch.equal(out_hf.logits, out_dag.logits)
    _assert_cache_equal(out_hf.past_key_values, out_dag.past_key_values)


def test_decode_step_exact_match():
    hf, dag = _build_pair(seed=3)
    prompt = torch.randint(0, hf.config.vocab_size, (1, 24), dtype=torch.long)
    pre_hf = hf(input_ids=prompt, use_cache=True, return_dict=True)
    pre_dag = dag(input_ids=prompt, use_cache=True, return_dict=True)

    nxt = torch.randint(0, hf.config.vocab_size, (1, 1), dtype=torch.long)
    attn_mask = torch.ones((1, 25), dtype=torch.float32)
    out_hf = hf(
        input_ids=nxt,
        past_key_values=pre_hf.past_key_values,
        attention_mask=attn_mask,
        use_cache=True,
        return_dict=True,
    )
    out_dag = dag(
        input_ids=nxt,
        past_key_values=pre_dag.past_key_values,
        attention_mask=attn_mask,
        use_cache=True,
        return_dict=True,
    )

    assert torch.equal(out_hf.logits, out_dag.logits)
    _assert_cache_equal(out_hf.past_key_values, out_dag.past_key_values)


def test_greedy_generation_exact_match():
    hf, dag = _build_pair(seed=4)
    prompt = torch.randint(0, hf.config.vocab_size, (1, 8), dtype=torch.long)
    seq_hf = prompt.clone()
    seq_dag = prompt.clone()
    cache_hf = None
    cache_dag = None

    for _ in range(16):
        in_hf = seq_hf if cache_hf is None else seq_hf[:, -1:]
        in_dag = seq_dag if cache_dag is None else seq_dag[:, -1:]
        out_hf = hf(input_ids=in_hf, past_key_values=cache_hf, use_cache=True, return_dict=True)
        out_dag = dag(input_ids=in_dag, past_key_values=cache_dag, use_cache=True, return_dict=True)
        assert torch.equal(out_hf.logits, out_dag.logits)
        tok_hf = out_hf.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_dag = out_dag.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        assert torch.equal(tok_hf, tok_dag)
        seq_hf = torch.cat([seq_hf, tok_hf], dim=1)
        seq_dag = torch.cat([seq_dag, tok_dag], dim=1)
        cache_hf = out_hf.past_key_values
        cache_dag = out_dag.past_key_values

    assert torch.equal(seq_hf, seq_dag)


def test_multiple_sequence_shapes_exact_match():
    hf, dag = _build_pair(seed=5)
    for prompt_len in (1, 7, 32):
        prompt = torch.randint(0, hf.config.vocab_size, (1, prompt_len), dtype=torch.long)
        out_hf = hf(input_ids=prompt, use_cache=True, return_dict=True)
        out_dag = dag(input_ids=prompt, use_cache=True, return_dict=True)
        assert torch.equal(out_hf.logits, out_dag.logits)
        _assert_cache_equal(out_hf.past_key_values, out_dag.past_key_values)

        cache_hf = out_hf.past_key_values
        cache_dag = out_dag.past_key_values
        for _ in range(4):
            nxt = torch.randint(0, hf.config.vocab_size, (1, 1), dtype=torch.long)
            out_hf = hf(input_ids=nxt, past_key_values=cache_hf, use_cache=True, return_dict=True)
            out_dag = dag(input_ids=nxt, past_key_values=cache_dag, use_cache=True, return_dict=True)
            assert torch.equal(out_hf.logits, out_dag.logits)
            _assert_cache_equal(out_hf.past_key_values, out_dag.past_key_values)
            cache_hf = out_hf.past_key_values
            cache_dag = out_dag.past_key_values
