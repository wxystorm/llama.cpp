#!/usr/bin/env python3
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='.')
args = ap.parse_args()
path = Path(args.repo).resolve() / 'src' / 'llama-graph.cpp'
text = path.read_text(encoding='utf-8')

repls = [
(
'''    const int64_t n_embd_q  = n_embd_head * n_head;\n    const int64_t n_embd_kv = n_embd_head * n_head_kv;\n\n    ggml_tensor * Qcur, * Kcur, * Vcur;\n''',
'''    const int64_t n_embd_q  = n_embd_head * n_head;\n    const int64_t n_embd_kv = n_embd_head * n_head_kv;\n    // QKV may be built for an XT-sized token slice in the prefill DAG.\n    // Use the actual input token dimension, not the macro ubatch size.\n    const int64_t n_tokens_qkv = cur->ne[1];\n\n    ggml_tensor * Qcur, * Kcur, * Vcur;\n'''
),
(
'''        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);\n        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],\n            ggml_row_size(qkv->type, n_embd_q));\n        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],\n            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));\n''',
'''        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens_qkv,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);\n        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens_qkv,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],\n            ggml_row_size(qkv->type, n_embd_q));\n        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens_qkv,\n            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],\n            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));\n'''
),
(
'''        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens); //\n        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);\n        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);\n''',
'''        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens_qkv); //\n        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_qkv);\n        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_qkv);\n'''
),
]

for i, (old, new) in enumerate(repls, 1):
    if old not in text:
        if new in text:
            print(f'hunk {i}: already applied')
            continue
        raise SystemExit(f'hunk {i}: expected source not found')
    text = text.replace(old, new, 1)
    print(f'hunk {i}: applied')

path.write_text(text, encoding='utf-8')
print(f'updated {path}')
