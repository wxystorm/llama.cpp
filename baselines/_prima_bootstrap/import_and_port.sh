#!/usr/bin/env bash
# Run from any path inside the isolated wxystorm/llama.cpp branch.
# Usage: bash baselines/_prima_bootstrap/import_and_port.sh [--push]
set -euo pipefail
branch=baseline/prima-qwen3-dense-20261107
upstream_sha=288df6a2cc2120387e12d6f40798f30cc525c9ea
root=$(git rev-parse --show-toplevel)
cd "$root"
if test "$(git branch --show-current)" != "$branch"; then
  echo "Refusing to modify another branch; expected $branch" >&2
  exit 2
fi
dest=baselines/prima.cpp
if test ! -f "$dest/src/llama.cpp"; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  git clone --depth 1 --branch main https://github.com/OpenCPIL/prima.cpp.git "$tmp/source"
  actual=$(git -C "$tmp/source" rev-parse HEAD)
  if test "$actual" != "$upstream_sha"; then
    echo "Upstream moved to $actual; expected $upstream_sha. Do not import an unreviewed version." >&2
    exit 3
  fi
  mkdir -p "$dest"
  tar -C "$tmp/source" --exclude=.git -cf - . | tar -C "$dest" -xf -
fi
test -f "$dest/LICENSE"
python3 -m py_compile baselines/_prima_bootstrap/port_qwen3_dense.py
if ! grep -q 'LLM_ARCH_QWEN3' "$dest/src/llama.cpp"; then
  python3 baselines/_prima_bootstrap/port_qwen3_dense.py "$dest/src/llama.cpp"
fi
test -f "$dest/src/llama.cpp"
git add "$dest"
if ! git diff --cached --quiet; then
  git commit -m "baseline(prima): import pinned upstream and port Qwen3 Dense"
fi
if test "${1:-}" = "--push"; then
  git push origin "HEAD:$branch"
fi
echo "PRIMA Qwen3 Dense port staged in $dest (compile + parity remain unverified)."
