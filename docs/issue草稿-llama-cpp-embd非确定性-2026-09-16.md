# Issue 草稿：llama.cpp `embd` 注入非确定性（2026-09-16）

> 用途：**由人工审阅・改写后，用本人账号提交到 `ggml-org/llama.cpp`**。
> 准备方式说明：上游 `CONTRIBUTING.md` 有明确的 AI 使用政策（模板里写明"copysasting language model outputs is strictly prohibited"），因此本文档只作为**草稿与素材**，正式提交前请自行改写语言与细节。
>
> 关联材料：[上游llama-embd注入非确定性-复现与issue材料](上游llama-embd注入非确定性-复现与issue材料-2026-09-16.md)（含完整实验矩阵与逐层定位）
> 复现 gist（已创建，公开）：**https://gist.github.com/SgfKrc/ce1d31578fd39b531c6274bc8434558e**（`embd-nondet-probe.cpp`、`incr-probe.cpp`、`RESULTS.md`）

## 提交信息（照抄即可）

| 项 | 值 |
| --- | --- |
| 仓库 | `ggml-org/llama.cpp` |
| 模板 | **`019-bug-misc.yml`（Bug (misc.)）** —— 因为有 issue form，正文按它的字段顺序组织 |
| 标题（模板要求 `Misc. bug: ` 前缀） | `Misc. bug: llama_batch.embd (embedding input) is not bitwise reproducible on the CPU backend` |
| 建议标签 | `bug-unconfirmed`（模板自带） |

## 正文（以下整段可复制）

---

### Name and Version

version: 6f04274cc145cf76aafc46eb8c1ea054ca221368
built with MSYS2 UCRT64 g++ (x86_64-w64-mingw32) on Windows, CPU-only build:

```
cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF
mingw32-make -C build-cpu -j4
```

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

No stock binary is involved; a minimal probe is provided in the gist linked below. Invocations:

```
probe.exe model.gguf embd5.f32 repeat 6 1     # embd path, 6 repeats in one context
probe.exe model.gguf embd5.f32 tokens 6 1     # token path control
```

### Problem description & steps to reproduce

**Summary.** On the CPU backend, decoding through the embedding-input path
(`llama_batch.embd` set, `ubatch->token == NULL`) is **not bitwise reproducible**: repeating the
exact same decode in the same process and context, calling `llama_memory_clear(llama_get_memory(ctx), true)`
before every decode, yields **different logits** each time (cosine 0.984–0.9999). The token path
(`llama_batch.token`) is **bitwise identical** under the same conditions. `argmax` stays constant,
so greedy generation is unaffected — the impact is on reproducibility.

**Minimal reproducer** (sources + data): https://gist.github.com/SgfKrc/ce1d31578fd39b531c6274bc8434558e

1. build the probe from the gist against a CPU-only build of this commit (exact command in the gist);
2. prepare `embd.f32` = one `N x n_embd` little-endian float32 batch of embeddings
   (random values reproduce the issue; **2 tokens is already enough**);
3. run `probe.exe model.gguf embd5.f32 repeat 6 1`, then the `tokens` mode as the control.

**Observed** (6 repeats, single process, `n_threads = n_threads_batch = 1`, `n_ctx = n_batch = n_ubatch = 512`):

| case | bitwise equal across repeats |
| --- | --- |
| token path | YES (cosine 1.00000000) |
| embd, **1** token (pos 0) | YES |
| embd, **2 / 3 / 5** tokens | **NO** (cosine 0.984–0.9999) |
| embd, 5 tokens, **incremental** injection (1 token per decode, pos 0..4) | **NO** |
| embd, 5 tokens, all-zero input | YES, and output is all-zero |
| embd, 5 tokens, separate processes | NO (pairwise cosine 0.9946–0.9971) |
| embd, 5 tokens, two contexts in one process | NO (cosine 0.9885) |

`argmax` is constant (= 11751) in all cases and top-10 overlap stays high.

**Per-layer localization** via `llama_get_hidden_state`: with this model (`qwen35`: hybrid SSM +
full attention every 4 layers), layers 0–2 (SSM) are **bitwise identical** between runs and the
divergence starts exactly at **layer 3, the first full-attention layer**. Together with
"1 token is deterministic, 2 tokens are not" and "incremental injection with pos >= 1 is also not",
this points at attention computed over a **non-empty KV history**.

**Ruled out so far** (each with an experiment, details in the gist):

- unwritten `inp->tokens` input (explicitly zeroing it inside `llm_graph_input_embd::set_input` does not help)
- multi-threading (`n_threads = n_threads_batch = 1`)
- KV capacity / unused KV region (swept `n_ctx` 8…512)
- reading uninitialized or stale KV content (all-zero input yields all-zero output)
- ggml fused ops (`GGML_CPU_DISABLE_FUSION=1`)
- CPU Flash Attention (`flash_attn_type = DISABLED`)
- CPU repack (not used in this build: "cannot be used with preferred buffer type CPU_REPACK, using CPU instead")

**Expected behaviour.** Same context + same input + cleared memory ⇒ identical output, as on the
token path; a pure-CPU backend should be deterministic.

**Extra context.** This surfaced while prototyping a layer-pipeline relay: compute layers `0..k`
with one model instance, then inject the resulting hidden state into a truncated model that
computes `k+1..N` through `llama_batch.embd`. Greedy output still matches the non-relay baseline
for 16/16 steps, but bit-exact reproducibility across processes is not achievable today.

---

## 提交前自检清单（建议）

- [ ] 用中文/英文重述一遍 Summary 与 steps（**改成你自己的语言**，避免直贴）
- [ ] 确认 gist 可公开访问：https://gist.github.com/SgfKrc/ce1d31578fd39b531c6274bc8434558e
- [ ] 选模板 **Bug (misc.)**（`Misc. bug: ` 前缀由模板自动加）
- [ ] 提交后把 issue URL 回填到本文件与《上游llama-embd注入非确定性-复现与issue材料》的变更记录

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-16 | 新建：按 `019-bug-misc` 模板整理的可提交 issue 正文草稿 + 提交信息表 + gist 链接（gist 已创建）。**不代为提交**（上游有 AI 使用政策），由人工审阅改写后提交。 |
