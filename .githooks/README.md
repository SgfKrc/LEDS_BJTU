# .githooks —— 本地 git 钩子（入库，需手动启用）

仓库原先**没有任何自动化检查**，`.git/hooks/` 里那 4 个钩子全是 Git LFS 装的。这里提供
一套**入库的**钩子，让文档问题在推送前就被拦住；同一套检查在
[`.github/workflows/checks.yml`](../.github/workflows/checks.yml) 里作为兜底再跑一遍。

## 启用

```bash
git config core.hooksPath .githooks     # 或 ./.githooks/install.sh / .\.githooks\install.ps1
```

`core.hooksPath` 是仓库级 config、不随 clone 传播，所以**每台机器要装一次**（不装不影响
使用，只是少一层本地拦截，CI 仍会兜底）。

## 内容

| 钩子 | 作用 |
| --- | --- |
| `pre-push` | 先跑 `scripts/run_doc_checks.py`（文档链接 + README 双语同步），再**转发 Git LFS** |
| `post-commit` / `post-checkout` / `post-merge` | 纯 Git LFS 转发（等价于原钩子） |

## 两个要点

1. **必须转发 LFS**：改用 `core.hooksPath` 后 `.git/hooks/` 里的原 LFS 钩子不再执行，
   若 `pre-push` 不调 `git lfs pre-push "$@"`，推送 LFS 跟踪的大文件会**静默漏传**。
2. **找不到 python 就放行**：`pre-push` 依次尝试 `.venv-test/{bin/python,Scripts/python.exe}`、
   `python3`、`python`（以及 `$PYTHON`）。都没找到时只**警告**并继续 —— 环境缺失不该把 push
   卡死；CI 会跑**同一套**检查兜底。

## 为什么 CI 与钩子共用一条命令

`scripts/run_doc_checks.py` 是唯一入口，两处都调它。若各写一份清单，加检查时迟早只改一处，
另一个入口就形同虚设。

## 绕过

```bash
git push --no-verify
```
