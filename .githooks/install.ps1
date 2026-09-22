# install.ps1：启用本仓库的 git 钩子（core.hooksPath=.githooks）—— Windows 版
#
# 为什么需要安装步骤：`core.hooksPath` 是**仓库级 config**，不会随 clone 传播；
# 不装的话 `.githooks/` 里的 pre-push 不会生效（CI 仍会兜底跑同一套检查）。
#
# 注意：启用后 Git **只**执行 .githooks/ 下的钩子，.git/hooks/ 里 Git LFS 装的原钩子不再生效
# —— 因此 .githooks/pre-push 等文件里都显式转发了 `git lfs <hook>`，行为与原钩子等价。

$ErrorActionPreference = 'Stop'

$repoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

# Windows 上 core.fileMode=false，无法用 chmod 提交可执行位；用 index 显式设置，
# 使 Linux/macOS 检出后钩子仍可执行。
foreach ($hook in 'pre-push', 'post-commit', 'post-checkout', 'post-merge', 'install.sh') {
    git update-index --chmod=+x ".githooks/$hook" 2>$null | Out-Null
}

git config core.hooksPath .githooks

Write-Host '[ok] 已启用 core.hooksPath=.githooks'
Write-Host '     pre-push 会先跑文档检查（scripts/run_doc_checks.py），再转发 Git LFS。'
Write-Host '     绕过：git push --no-verify'
