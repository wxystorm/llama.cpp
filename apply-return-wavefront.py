#!/usr/bin/env python3
"""Apply the Qwen2/Qwen2.5 return-wavefront prefill experiment.

Target: wxystorm/llama.cpp fork-new after the 2026-09-17 baseline fixes
(QKV slice fix + Meta trailing-host-view tail fix), with the later task-DAG
experiments reset from the working tree.

The patch keeps one normal full ggml graph.  It chunk-splits only the Tensor
suffix *inside that graph* and lets Meta cross a layer boundary one chunk at a
time while older Phone returns are still in flight.  No per-(layer,chunk)
prepare_ubatch/scheduler allocation is introduced.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATCH = HERE / "base-range-meta.patch"


def parse_patch(path: Path):
    files: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    old: list[str] | None = None
    new: list[str] | None = None

    def flush() -> None:
        nonlocal old, new
        if current is not None and old is not None and new is not None:
            files.setdefault(current, []).append(("".join(old), "".join(new)))
        old = new = None

    for raw in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if raw.startswith("diff --git "):
            flush()
            current = raw.split()[2][2:]
            continue
        if raw.startswith("--- ") or raw.startswith("+++ "):
            continue
        if raw.startswith("@@"):
            flush()
            old, new = [], []
            continue
        if old is None or new is None:
            continue
        if raw.startswith("+"):
            new.append(raw[1:])
        elif raw.startswith("-"):
            old.append(raw[1:])
        elif raw.startswith(" "):
            old.append(raw[1:])
            new.append(raw[1:])
        elif raw == "\\ No newline at end of file\n":
            pass
        else:
            raise RuntimeError(f"unexpected patch line: {raw!r}")
    flush()
    return files


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one baseline match, found {n}")
    return text.replace(old, new, 1)


def insert_after(text: str, anchor: str, addition: str, label: str) -> str:
    return replace_once(text, anchor, anchor + addition, label)


def apply_text_patch(repo: Path, patch: Path, backup: bool, check: bool) -> dict[Path, str]:
    changes = parse_patch(patch)
    rewritten: dict[Path, str] = {}
    for rel, hunks in changes.items():
        path = repo / rel
        if not path.is_file():
            raise RuntimeError(f"missing baseline file: {rel}")
        text = path.read_text(encoding="utf-8")
        cursor = 0
        for idx, (old, new) in enumerate(hunks, 1):
            pos = text.find(old, cursor)
            if pos < 0:
                positions = []
                start = 0
                while True:
                    p = text.find(old, start)
                    if p < 0:
                        break
                    positions.append(p)
                    start = p + 1
                if len(positions) != 1:
                    raise RuntimeError(
                        f"{rel}: base hunk {idx} not found uniquely (matches={len(positions)})"
                    )
                pos = positions[0]
            text = text[:pos] + new + text[pos + len(old):]
            cursor = pos + len(new)
        rewritten[path] = text
        print(f"base checked {rel}: {len(hunks)} rewrite(s)")

    if not check:
        for path, text in rewritten.items():
            if backup:
                bak = path.with_name(path.name + ".bak-return-wavefront")
                if not bak.exists():
                    shutil.copy2(path, bak)
            path.write_text(text, encoding="utf-8")
    return rewritten


QWEN2_ANCHOR = '''    auto * inp_attn = build_attn_inp_kv();\n\n    ggml_tensor * inp_out_ids = build_output_head ? build_inp_out_ids() : nullptr;\n\n    for (int il = layer_begin; il < layer_end; ++il) {\n        ggml_tensor * inpSA = inpL;\n'''

QWEN2_REPLACEMENT = r'''    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_output_head ? build_inp_out_ids() : nullptr;

    // Return-wavefront is intentionally a full-graph path.  The GPU prefix is
    // built exactly as before; only the final contiguous TENSOR_SPLIT suffix is
    // represented as layer-major token chunks inside this same graph.
    const int return_wave_chunk_tokens = llama_hybrid_runtime_prefill_chunk_tokens();
    int return_wave_first_layer = -1;
    bool return_wave_eligible =
        std::getenv("LLAMA_HYBRID_RETURN_WAVEFRONT") != nullptr &&
        !stage_graph && build_output_head && n_tokens > 1 &&
        ubatch.n_pos == 1 && !ubatch.equal_seqs() && ubatch.n_seqs_unq == 1 &&
        model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
        return_wave_chunk_tokens > 0 && loras->empty();

    if (return_wave_eligible) {
        for (int il = layer_begin; il < layer_end; ++il) {
            if (model.hybrid_layer_mode(il) == llama_hybrid_layer_mode::TENSOR_SPLIT) {
                return_wave_first_layer = il;
                break;
            }
        }
        // V1 is the measured GPU -> TENSOR topology: keep a normal prefix and
        // require the Tensor split to continue through the final transformer
        // layer.  This avoids mixing the experiment with a later PHONE stage.
        return_wave_eligible = return_wave_first_layer > layer_begin;
        for (int il = return_wave_first_layer; return_wave_eligible && il < layer_end; ++il) {
            return_wave_eligible =
                model.hybrid_layer_mode(il) == llama_hybrid_layer_mode::TENSOR_SPLIT &&
                cvec->tensor_for(il) == nullptr;
        }
    }

    for (int il = layer_begin; il < layer_end; ++il) {
        if (return_wave_eligible && il == return_wave_first_layer) {
            const std::vector<int> chunk_sizes =
                llama_hybrid_split_by_chunk_size((int) n_tokens, return_wave_chunk_tokens);
            GGML_ASSERT(!chunk_sizes.empty());

            std::vector<int64_t> chunk_begin(chunk_sizes.size(), 0);
            std::vector<ggml_tensor *> hidden_chunks;
            std::vector<llm_graph_input_attn_kv *> attn_chunks;
            hidden_chunks.reserve(chunk_sizes.size());
            attn_chunks.reserve(chunk_sizes.size());

            int64_t token_begin = 0;
            for (size_t ci = 0; ci < chunk_sizes.size(); ++ci) {
                const int64_t token_count = chunk_sizes[ci];
                GGML_ASSERT(token_count > 0);
                chunk_begin[ci] = token_begin;
                hidden_chunks.push_back(ggml_view_2d(
                    ctx0, inpL, inpL->ne[0], token_count,
                    inpL->nb[1], token_begin * inpL->nb[1]));
                attn_chunks.push_back(build_attn_inp_kv_range(
                    (uint32_t) token_begin, (uint32_t) token_count));
                token_begin += token_count;
            }
            GGML_ASSERT(token_begin == n_tokens);

            // Layer-major order is deliberate.  We finish all Phone compute
            // submissions for layer L before entering L+1, while return/reduce
            // tasks from L may remain in flight.  Meta waits only OUT(L,C)
            // before executing Attention(L+1,C), so the ready prefix flows
            // forward without the old whole-layer return barrier.
            for (int wave_layer = il; wave_layer < layer_end; ++wave_layer) {
                for (size_t ci = 0; ci < chunk_sizes.size(); ++ci) {
                    const int64_t begin = chunk_begin[ci];
                    const int64_t count = chunk_sizes[ci];
                    ggml_tensor * inpSA = hidden_chunks[ci];

                    ggml_tensor * attn_norm = build_norm(
                        inpSA, model.layers[wave_layer].attn_norm, NULL,
                        LLM_NORM_RMS, wave_layer);
                    const std::string attn_norm_name =
                        "prefill_wave_attn_norm_chunk_" + std::to_string(ci);
                    cb(attn_norm, attn_norm_name.c_str(), wave_layer);

                    auto [Qcur, Kcur, Vcur] = build_qkv(
                        model.layers[wave_layer], attn_norm,
                        n_embd_head, n_head, n_head_kv, wave_layer);

                    ggml_tensor * pos_chunk = ggml_view_1d(
                        ctx0, inp_pos, count, begin * inp_pos->nb[0]);
                    Qcur = ggml_rope_ext(
                        ctx0, Qcur, pos_chunk, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(
                        ctx0, Kcur, pos_chunk, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                    ggml_tensor * attn = build_attn(
                        attn_chunks[ci], model.layers[wave_layer].wo,
                        model.layers[wave_layer].wo_b, model.layers[wave_layer].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f / sqrtf(float(n_embd_head)), wave_layer);
                    const std::string attn_name =
                        "prefill_wave_attn_out_chunk_" + std::to_string(ci);
                    cb(attn, attn_name.c_str(), wave_layer);

                    ggml_tensor * ffn_inp = ggml_add(ctx0, attn, inpSA);
                    const std::string ffn_inp_name =
                        "prefill_wave_ffn_inp_chunk_" + std::to_string(ci);
                    cb(ffn_inp, ffn_inp_name.c_str(), wave_layer);

                    ggml_tensor * ffn_norm = build_norm(
                        ffn_inp, model.layers[wave_layer].ffn_norm, NULL,
                        LLM_NORM_RMS, wave_layer);
                    const std::string norm_name =
                        "prefill_ffn_norm_chunk_" + std::to_string(ci);
                    cb(ffn_norm, norm_name.c_str(), wave_layer);

                    ggml_tensor * down = build_ffn(
                        ffn_norm,
                        model.layers[wave_layer].ffn_up, NULL, NULL,
                        model.layers[wave_layer].ffn_gate, NULL, NULL,
                        model.layers[wave_layer].ffn_down, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, wave_layer);
                    const std::string down_name =
                        "prefill_ffn_down_chunk_" + std::to_string(ci);
                    cb(down, down_name.c_str(), wave_layer);

                    // The split FFN return worker reduces Phone into this PC
                    // buffer and then adds ffn_inp in-place.  Therefore, after
                    // that exact return task completes, 'down' is OUT(L,C) and
                    // can be consumed directly by Attention(L+1,C).
                    ggml_build_forward_expand(gf, down);
                    hidden_chunks[ci] = down;
                }
            }

            cur = hidden_chunks.back();
            for (int ci = (int) hidden_chunks.size() - 2; ci >= 0; --ci) {
                cur = ggml_concat(ctx0, hidden_chunks[(size_t) ci], cur, 1);
            }
            cb(cur, "l_out", layer_end - 1);

            if (!build_output_head) {
                res->t_stage_output = cur;
                ggml_build_forward_expand(gf, cur);
                return;
            }

            if (inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }
            cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
            cb(cur, "result_norm", -1);
            res->t_embd = cur;

            cur = build_lora_mm(model.output, cur, model.output_s);
            if (model.output_b != nullptr) {
                cur = ggml_add(ctx0, cur, model.output_b);
            }
            cb(cur, "result_output", -1);
            res->t_logits = cur;
            ggml_build_forward_expand(gf, cur);
            return;
        }

        ggml_tensor * inpSA = inpL;
'''


def patch_qwen2(text: str) -> str:
    if "prefill_wave_ffn_inp_chunk_" in text:
        raise RuntimeError("src/models/qwen2.cpp: return-wavefront appears already applied")
    return replace_once(text, QWEN2_ANCHOR, QWEN2_REPLACEMENT, "qwen2 return-wavefront insertion")


def patch_meta_after_v2(text: str) -> str:
    # Rename the old monolithic-DAG bookkeeping to the new return-wavefront
    # graph.  These identifiers exist only in the selected v2 Meta hunks.
    replacements = {
        "ggml_backend_meta_parse_prefill_dag_ffn_inp_chunk": "ggml_backend_meta_parse_prefill_wave_ffn_inp_chunk",
        "prefill_dag_ffn_inp_chunk_": "prefill_wave_ffn_inp_chunk_",
        "find_recent_prefill_dag_ffn_inp": "find_recent_prefill_wave_ffn_inp",
        "prefill_dag_graph": "return_wavefront_graph",
        "prefill_dag_first_layer": "return_wavefront_first_layer",
        "prefill_dag_dependency_wait_count": "return_wave_dependency_wait_count",
        "prefill_dag_dependency_wait_us": "return_wave_dependency_wait_us",
        "prefill_dag_dependency_wait_max_us": "return_wave_dependency_wait_max_us",
        "prefill_dag_overlap_boundaries": "return_wave_overlap_boundaries",
        "PREFILL_DAG_COMMIT": "RETURN_WAVEFRONT_COMMIT",
        "PREFILL_DAG_DEP_WAIT": "RETURN_WAVEFRONT_DEP_WAIT",
        "PREFILL_DAG_STALL": "RETURN_WAVEFRONT_SUM",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    dep_anchor = '''    auto wait_prefill_reduce_dependency = [&](int layer, int chunk, bool & waited) {\n        waited = false;\n        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {\n            if (pending_prefill_reduce_task[lane] == 0 ||\n                pending_prefill_reduce_layer[lane] != layer ||\n                pending_prefill_reduce_chunk[lane] != chunk) {\n                continue;\n            }\n            waited = true;\n            return wait_prefill_reduce_lane(lane);\n        }\n        return GGML_STATUS_SUCCESS;\n    };\n'''
    dep_add = '''    auto wait_prefill_reduces_before = [&](int layer_exclusive, bool & waited, int64_t & wait_us) {\n        waited = false;\n        wait_us = 0;\n        ggml_status result = GGML_STATUS_SUCCESS;\n        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {\n            if (pending_prefill_reduce_task[lane] == 0 ||\n                pending_prefill_reduce_layer[lane] >= layer_exclusive) {\n                continue;\n            }\n            waited = true;\n            const int64_t begin_us = ggml_time_us();\n            const ggml_status status = wait_prefill_reduce_lane(lane);\n            wait_us += ggml_time_us() - begin_us;\n            if (result == GGML_STATUS_SUCCESS && status != GGML_STATUS_SUCCESS) {\n                result = status;\n            }\n        }\n        return result;\n    };\n'''
    text = insert_after(text, dep_anchor, dep_add, "meta one-layer wait helper")

    # Enable the two existing snapshot-return lanes automatically for the
    # return-wavefront graph.  The old explicit env switch still works.
    old_dual = '''            const bool dual_prefill_return =\n                use_snapshot_pipeline && std::getenv("GGML_META_PREFILL_DUAL_RETURN") != nullptr;\n'''
    new_dual = '''            const bool dual_prefill_return =\n                use_snapshot_pipeline &&\n                (std::getenv("GGML_META_PREFILL_DUAL_RETURN") != nullptr || return_wavefront_graph);\n'''
    text = replace_once(text, old_dual, new_dual, "meta auto dual return")

    old_counters = '''    int64_t return_wave_dependency_wait_count  = 0;\n    int64_t return_wave_dependency_wait_us     = 0;\n    int64_t return_wave_dependency_wait_max_us = 0;\n    int64_t return_wave_overlap_boundaries     = 0;\n'''
    new_counters = '''    int64_t return_wave_dependency_wait_count  = 0;\n    int64_t return_wave_dependency_wait_us     = 0;\n    int64_t return_wave_dependency_wait_max_us = 0;\n    int64_t return_wave_overlap_boundaries     = 0;\n    int64_t return_wave_ahead_attn_chunks      = 0;\n    int64_t return_wave_ahead_phone_submits    = 0;\n    int64_t return_wave_old_layer_wait_count   = 0;\n    int64_t return_wave_old_layer_wait_us      = 0;\n'''
    text = replace_once(text, old_counters, new_counters, "meta counters")

    old_barrier = '''        const bool continues_prefill_layer = norm_can_overlap || down_can_overlap;\n\n        if (return_wavefront_graph && (is_prefill_norm_sg || is_prefill_down_sg)) {\n            if (is_prefill_norm_sg && prefill_norm_layer > return_wavefront_first_layer) {\n                bool waited = false;\n                const int64_t wait_start_us = ggml_time_us();\n                const ggml_status status = wait_prefill_reduce_dependency(\n                    prefill_norm_layer - 1, prefill_norm_chunk, waited);\n                const int64_t wait_us = ggml_time_us() - wait_start_us;\n                if (waited) {\n                    ++return_wave_dependency_wait_count;\n                    return_wave_dependency_wait_us += wait_us;\n                    return_wave_dependency_wait_max_us =\n                        std::max(return_wave_dependency_wait_max_us, wait_us);\n                    if (pipeline_debug) {\n                        printf(\n                            "[RETURN_WAVEFRONT_DEP_WAIT] layer=%d chunk=%d predecessor=%d "\n                            "wait_ms=%.3f\\n",\n                            prefill_norm_layer, prefill_norm_chunk,\n                            prefill_norm_layer - 1, wait_us / 1000.0);\n                    }\n                } else if (has_pending_prefill_reduce()) {\n                    ++return_wave_overlap_boundaries;\n                }\n                if (status != GGML_STATUS_SUCCESS) {\n                    return status;\n                }\n            } else if (has_pending_prefill_reduce()) {\n                ++return_wave_overlap_boundaries;\n            }\n        } else if (has_pending_prefill_reduce() && !continues_prefill_layer) {\n'''
    new_barrier = '''        const bool continues_prefill_layer = norm_can_overlap || down_can_overlap;\n        if (return_wavefront_graph && (is_prefill_norm_sg || is_prefill_down_sg)) {\n            if (is_prefill_norm_sg) {\n                // V1 permits L+1 while returns from L are in flight, but never\n                // L+2 while an L return still owns a lane.\n                bool older_waited = false;\n                int64_t older_wait_us = 0;\n                const ggml_status older_status = wait_prefill_reduces_before(\n                    prefill_norm_layer - 1, older_waited, older_wait_us);\n                if (older_waited) {\n                    ++return_wave_old_layer_wait_count;\n                    return_wave_old_layer_wait_us += older_wait_us;\n                }\n                if (older_status != GGML_STATUS_SUCCESS) {\n                    return older_status;\n                }\n\n                if (prefill_norm_layer > return_wavefront_first_layer) {\n                    bool waited = false;\n                    const int64_t wait_start_us = ggml_time_us();\n                    const ggml_status status = wait_prefill_reduce_dependency(\n                        prefill_norm_layer - 1, prefill_norm_chunk, waited);\n                    const int64_t wait_us = ggml_time_us() - wait_start_us;\n                    if (waited) {\n                        ++return_wave_dependency_wait_count;\n                        return_wave_dependency_wait_us += wait_us;\n                        return_wave_dependency_wait_max_us =\n                            std::max(return_wave_dependency_wait_max_us, wait_us);\n                        if (pipeline_debug) {\n                            printf(\n                                "[RETURN_WAVEFRONT_DEP_WAIT] layer=%d chunk=%d predecessor=%d wait_ms=%.3f\\n",\n                                prefill_norm_layer, prefill_norm_chunk,\n                                prefill_norm_layer - 1, wait_us / 1000.0);\n                        }\n                    }\n                    if (status != GGML_STATUS_SUCCESS) {\n                        return status;\n                    }\n\n                    // Any other L-1 return still in flight means this chunk's\n                    // Attention is doing useful ahead work during old-layer D2H.\n                    if (has_pending_prefill_reduce_for_layer(prefill_norm_layer - 1)) {\n                        ++return_wave_overlap_boundaries;\n                        ++return_wave_ahead_attn_chunks;\n                    }\n                }\n            } else if (prefill_down_layer > return_wavefront_first_layer &&\n                       has_pending_prefill_reduce_for_layer(prefill_down_layer - 1)) {\n                // The down subgraph starts backend 0 and backend 1 together,\n                // so this is an immediate next-layer Phone FFN submission.\n                ++return_wave_ahead_phone_submits;\n            }\n        } else if (has_pending_prefill_reduce() && !continues_prefill_layer) {\n'''
    text = replace_once(text, old_barrier, new_barrier, "meta return-wavefront barrier")

    old_summary = '''    if (return_wavefront_graph) {\n        printf(\n            "[RETURN_WAVEFRONT_SUM] dependency_wait_count=%" PRId64\n            " dependency_wait_ms=%.3f dependency_wait_max_ms=%.3f"\n            " overlap_boundaries=%" PRId64\n            " legacy_layer_barrier_count=%" PRId64 "\\n",\n            return_wave_dependency_wait_count,\n            return_wave_dependency_wait_us / 1000.0,\n            return_wave_dependency_wait_max_us / 1000.0,\n            return_wave_overlap_boundaries,\n            layer_barrier_wait_count);\n    }\n'''
    new_summary = '''    if (return_wavefront_graph) {\n        printf(\n            "[RETURN_WAVEFRONT_SUM] enabled=1 dependency_wait_count=%" PRId64\n            " dependency_wait_ms=%.3f dependency_wait_max_ms=%.3f"\n            " overlap_boundaries=%" PRId64\n            " ahead_attn=%" PRId64\n            " ahead_phone_submit=%" PRId64\n            " old_layer_wait_count=%" PRId64\n            " old_layer_wait_ms=%.3f"\n            " legacy_layer_barrier_count=%" PRId64 "\\n",\n            return_wave_dependency_wait_count,\n            return_wave_dependency_wait_us / 1000.0,\n            return_wave_dependency_wait_max_us / 1000.0,\n            return_wave_overlap_boundaries,\n            return_wave_ahead_attn_chunks,\n            return_wave_ahead_phone_submits,\n            return_wave_old_layer_wait_count,\n            return_wave_old_layer_wait_us / 1000.0,\n            layer_barrier_wait_count);\n    }\n'''
    text = replace_once(text, old_summary, new_summary, "meta summary")

    return text


def verify_postconditions(repo: Path, texts: dict[Path, str]) -> None:
    def txt(rel: str) -> str:
        p = repo / rel
        return texts.get(p, p.read_text(encoding="utf-8"))

    required = {
        "src/llama-kv-cache.h": ["current_sinfo_range", "set_input_k_idxs_range"],
        "src/llama-graph.h": ["build_attn_inp_kv_range", "token_count"],
        "src/llama-graph.cpp": ["llama_graph_ubatch_range", "n_tokens_qkv = cur->ne[1]"],
        "src/models/qwen2.cpp": ["LLAMA_HYBRID_RETURN_WAVEFRONT", "prefill_wave_ffn_inp_chunk_"],
        "ggml/src/ggml-backend-meta.cpp": [
            "return_wavefront_graph", "pending_prefill_reduce_chunk",
            "wait_prefill_reduces_before", "[RETURN_WAVEFRONT_SUM]",
        ],
    }
    for rel, markers in required.items():
        data = txt(rel)
        for marker in markers:
            if marker not in data:
                raise RuntimeError(f"postcondition failed: {rel} missing {marker!r}")

    # The old monolithic-DAG graph marker must not remain in the final path.
    if "prefill_dag_ffn_inp_chunk_" in txt("ggml/src/ggml-backend-meta.cpp"):
        raise RuntimeError("postcondition failed: old prefill_dag marker remains in Meta")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="llama.cpp checkout root")
    ap.add_argument("--check", action="store_true", help="validate every rewrite without writing")
    ap.add_argument("--backup", action="store_true", help="create *.bak-return-wavefront before writing")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "src/models/qwen2.cpp").is_file():
        raise SystemExit(f"not a llama.cpp checkout: {repo}")
    if not BASE_PATCH.is_file():
        raise SystemExit(f"missing package file: {BASE_PATCH}")

    # Refuse to stack on the task-DAG experiment.  Reset the working tree to
    # the user's last committed baseline first, as agreed in the design.
    context = (repo / "src/llama-context.cpp").read_text(encoding="utf-8")
    if "PREFILL_DAG_V3_BEGIN" in context or "PREFILL_DAG_OVERHEAD" in context:
        raise SystemExit(
            "task-DAG markers are still present in src/llama-context.cpp; reset the uncommitted DAG experiment first"
        )

    # Validate the two fixes that the current fork-new baseline already has.
    graph_cpp = (repo / "src/llama-graph.cpp").read_text(encoding="utf-8")
    if "const int64_t n_tokens_qkv = cur->ne[1];" not in graph_cpp:
        raise SystemExit("baseline is missing the QKV slice hotfix (n_tokens_qkv = cur->ne[1])")
    meta0 = (repo / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
    if "is_passthrough_host_view" not in meta0:
        raise SystemExit("baseline is missing the Meta trailing-host-view tail fix")
    if "return_wavefront_graph" in meta0 or "prefill_wave_ffn_inp_chunk_" in meta0:
        raise SystemExit("return-wavefront appears already applied")

    # Apply the previously validated range-input + Meta exact-dependency base
    # hunks in memory first.  They are then specialized below.
    base_texts = apply_text_patch(repo, BASE_PATCH, backup=False, check=True)

    final_texts = dict(base_texts)

    qwen_path = repo / "src/models/qwen2.cpp"
    qwen_text = qwen_path.read_text(encoding="utf-8")
    final_texts[qwen_path] = patch_qwen2(qwen_text)

    meta_path = repo / "ggml/src/ggml-backend-meta.cpp"
    final_texts[meta_path] = patch_meta_after_v2(final_texts[meta_path])

    verify_postconditions(repo, final_texts)

    for path in sorted(final_texts):
        print(f"checked {path.relative_to(repo)}")

    if args.check:
        print("return-wavefront patch check: PASS")
        return 0

    for path, text in final_texts.items():
        if args.backup:
            bak = path.with_name(path.name + ".bak-return-wavefront")
            if not bak.exists():
                shutil.copy2(path, bak)
        path.write_text(text, encoding="utf-8")

    print(f"return-wavefront patch applied to {len(final_texts)} files")
    print("No commit or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
