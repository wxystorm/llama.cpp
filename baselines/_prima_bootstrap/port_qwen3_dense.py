#!/usr/bin/env python3
"""Minimal Qwen3 Dense support for the pinned OpenCPIL/prima.cpp baseline.

Port only the model architecture, rank-local tensors, graph and FLOP profiling.
Do not change PRIMA's communication or HALDA scheduling policy.
Fails closed if the upstream source no longer matches the pinned snapshot.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def modify_section(text: str, begin: str, end: str, mutate) -> str:
    i = text.index(begin)
    j = text.index(end, i + len(begin))
    return text[:i] + mutate(text[i:j]) + text[j:]


def port(src: str) -> str:
    if "LLM_ARCH_QWEN3" in src:
        raise RuntimeError("Qwen3 appears to be ported already; do not patch twice")

    src = replace_once(src,
        "    LLM_ARCH_QWEN2,\n    LLM_ARCH_QWEN2MOE,",
        "    LLM_ARCH_QWEN2,\n    LLM_ARCH_QWEN3,\n    LLM_ARCH_QWEN2MOE,",
        "architecture enum")
    src = replace_once(src,
        '    { LLM_ARCH_QWEN2,           "qwen2"        },',
        '    { LLM_ARCH_QWEN2,           "qwen2"        },\n'
        '    { LLM_ARCH_QWEN3,           "qwen3"        },',
        "architecture GGUF names")

    # Reuse Qwen2 GGUF naming but load two extra per-head normalization weights.
    map_start = "    {\n        LLM_ARCH_QWEN2,\n        {\n"
    map_end = "    {\n        LLM_ARCH_QWEN2MOE,\n"
    start = src.index(map_start)
    end = src.index(map_end, start)
    qwen2_map = src[start:end]
    qwen3_map = qwen2_map.replace("LLM_ARCH_QWEN2", "LLM_ARCH_QWEN3", 1)
    qwen3_map = replace_once(qwen3_map,
        '            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },',
        '            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },\n'
        '            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },',
        "Qwen3 Q norm name")
    qwen3_map = replace_once(qwen3_map,
        '            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },',
        '            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },\n'
        '            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },',
        "Qwen3 K norm name")
    src = src[:end] + qwen3_map + src[end:]

    # Dense Qwen3 GGUF: 0.6B/1.7B/4B/8B/14B/32B sizes. No MoE or VL.
    src = replace_once(src,
        "        case LLM_ARCH_QWEN2MOE:\n            {",
        "        case LLM_ARCH_QWEN3:\n"
        "            {\n"
        "                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);\n"
        "                switch (hparams.n_layer) {\n"
        "                    case 28: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;\n"
        "                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;\n"
        "                    case 40: model.type = e_model::MODEL_14B; break;\n"
        "                    case 64: model.type = e_model::MODEL_32B; break;\n"
        "                    default: model.type = e_model::MODEL_UNKNOWN;\n"
        "                }\n"
        "            } break;\n"
        "        case LLM_ARCH_QWEN2MOE:\n            {",
        "Qwen3 hyperparameters")

    def patch_loader(block: str) -> str:
        block = replace_once(block,
            "    const int64_t n_embd        = hparams.n_embd;\n"
            "    const int64_t n_embd_gqa    = hparams.n_embd_v_gqa();",
            "    const bool    is_qwen3     = model.arch == LLM_ARCH_QWEN3;\n"
            "    const int64_t n_embd        = hparams.n_embd;\n"
            "    const int64_t n_embd_gqa    = hparams.n_embd_v_gqa();\n"
            "    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();\n"
            "    const int64_t n_embd_head   = hparams.n_embd_head_k;\n"
            "    const int64_t n_embd_q      = is_qwen3 ? hparams.n_head() * n_embd_head : n_embd;",
            "Qwen3 Q projection dimension")
        block = replace_once(block,
            '        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0, set_needed);\n'
            '        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0, set_needed);\n'
            '        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0, set_needed);\n'
            '        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0, set_needed);',
            '        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_q}, 0, set_needed);\n'
            '        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, is_qwen3 ? n_embd_k_gqa : n_embd_gqa}, 0, set_needed);\n'
            '        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0, set_needed);\n'
            '        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0, set_needed);\n'
            '        if (is_qwen3) {\n'
            '            layer.attn_q_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head}, 0, set_needed);\n'
            '            layer.attn_k_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head}, 0, set_needed);\n'
            '        }',
            "Qwen3 per-head tensors")
        block = replace_once(block,
            '        layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0, set_needed);\n'
            '        layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0, set_needed);\n'
            '        layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0, set_needed);',
            '        // Qwen2 has Q/K/V biases; Qwen3 Dense has none.\n'
            '        if (!is_qwen3) {\n'
            '            layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0, set_needed);\n'
            '            layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0, set_needed);\n'
            '            layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0, set_needed);\n'
            '        }',
            "Qwen3 bias-free attention")
        return block

    src = modify_section(src, "static void llm_load_qwen2_tensors(", "// Returns false if cancelled", patch_loader)

    src = replace_once(src,
        "            case LLM_ARCH_QWEN2:\n"
        "                llm_load_qwen2_tensors(ml, model, ctx_map, n_world, my_rank, n_layer_window, true);",
        "            case LLM_ARCH_QWEN2:\n"
        "            case LLM_ARCH_QWEN3:\n"
        "                llm_load_qwen2_tensors(ml, model, ctx_map, n_world, my_rank, n_layer_window, true);",
        "distributed model load")

    def patch_graph(block: str) -> str:
        block = replace_once(block,
            "                Qcur = ggml_rope_ext(\n"
            "                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,",
            "                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);\n"
            "                if (model.arch == LLM_ARCH_QWEN3) {\n"
            "                    Qcur = llm_build_norm(ctx0, Qcur, hparams,\n"
            "                        model.layers[local_il].attn_q_norm, NULL,\n"
            "                        LLM_NORM_RMS, cb, il);\n"
            '                    cb(Qcur, "Qcur_normed", il);\n'
            "                }\n"
            "                Qcur = ggml_rope_ext(\n"
            "                    ctx0, Qcur, inp_pos, nullptr,",
            "Qwen3 per-head Q norm before RoPE")
        block = replace_once(block,
            "                Kcur = ggml_rope_ext(\n"
            "                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,",
            "                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);\n"
            "                if (model.arch == LLM_ARCH_QWEN3) {\n"
            "                    Kcur = llm_build_norm(ctx0, Kcur, hparams,\n"
            "                        model.layers[local_il].attn_k_norm, NULL,\n"
            "                        LLM_NORM_RMS, cb, il);\n"
            '                    cb(Kcur, "Kcur_normed", il);\n'
            "                }\n"
            "                Kcur = ggml_rope_ext(\n"
            "                    ctx0, Kcur, inp_pos, nullptr,",
            "Qwen3 per-head K norm before RoPE")
        return block

    src = modify_section(src,
        "    std::vector<ggml_cgraph *> build_qwen2() {",
        "    struct ggml_cgraph * build_qwen2moe() {",
        patch_graph)

    src = replace_once(src,
        "GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2) &&",
        "GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_QWEN3) &&",
        "supported graph architectures")
    src = replace_once(src,
        "        case LLM_ARCH_QWEN2:\n"
        "            {\n"
        "                result = llm.build_qwen2();",
        "        case LLM_ARCH_QWEN2:\n"
        "        case LLM_ARCH_QWEN3:\n"
        "            {\n"
        "                result = llm.build_qwen2();",
        "distributed graph builder dispatch")
    src = replace_once(src,
        "        case LLM_ARCH_QWEN2:\n"
        "        case LLM_ARCH_QWEN2MOE:\n"
        "        case LLM_ARCH_OLMOE:",
        "        case LLM_ARCH_QWEN2:\n"
        "        case LLM_ARCH_QWEN3:\n"
        "        case LLM_ARCH_QWEN2MOE:\n"
        "        case LLM_ARCH_OLMOE:",
        "NeoX RoPE architecture")
    src = replace_once(src,
        "            case LLM_ARCH_QWEN2:\n"
        "                llm_load_qwen2_tensors(*ml, *model, ctx_map, 1, 0, n_layer_window, false);",
        "            case LLM_ARCH_QWEN2:\n"
        "            case LLM_ARCH_QWEN3:\n"
        "                llm_load_qwen2_tensors(*ml, *model, ctx_map, 1, 0, n_layer_window, false);",
        "HALDA model profiling loader")
    src = replace_once(src,
        '            } else if (blk_suffix == "attn_q.weight") {',
        '            } else if (blk_suffix == "attn_q_norm.weight" || blk_suffix == "attn_k_norm.weight") {\n'
        '                count_n_flops(n_flops, GGML_TYPE_F32, PROFILER_LAYER_BACKEND, 4 * n_embd_head_k + 1);\n'
        '                count_n_flops(n_flops, cur->type, PROFILER_LAYER_BACKEND, n_embd_head_k);\n'
        '            } else if (blk_suffix == "attn_q.weight") {',
        "HALDA Q/K norm weight accounting")
    src = replace_once(src,
        "model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_NEMOTRON",
        "model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_QWEN3 || model.arch == LLM_ARCH_NEMOTRON",
        "attention precision")
    return src


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument("llama_cpp", type=Path)
    args = argp.parse_args()
    file = args.llama_cpp
    old = file.read_text(encoding="utf-8")
    updated = port(old)
    assert updated != old
    assert updated.count("LLM_ARCH_QWEN3") >= 10
    file.write_text(updated, encoding="utf-8")
    print(f"[PRIMA_QWEN3_PORT] status=PASS file={file} size={len(updated)}")


if __name__ == "__main__":
    main()
