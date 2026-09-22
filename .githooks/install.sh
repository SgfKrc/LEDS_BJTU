#!/bin/sh
# install.sh：启用本仓库的 git 钩子（core.hooksPath=.githooks）
#
# 注意：启用后 Git **只**执行 .githooks/ 下的钩子，.git/hooks/ 里的 Git LFS 原钩子不再生效 ——
# 所以 .githooks/pre-push 等文件里都显式转发了 `git lfs <hook>`，行为与原钩子等价。

set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

chmod +x .githooks/pre-push .githooks/post-commit .githooks/post-checkout .githooks/post-merge
git config core.hooksPath .githooks

echo "[ok] 已启用 core.hooksPath=.githooks"
echo "     pre-push 会先跑文档检查（scripts/run_doc_checks.py），再转发 Git LFS。"
echo "     绕过：git push --no-verify"
