# 上游 llama.cpp `embd` 注入路径非确定性：复现、定位与 issue 材料（2026-09-16）

> 状态：**现行 · issue 材料就绪（根因未坐实）**
>
> 适用范围：llama.cpp **CPU 后端**在 `llama_batch.embd`（向量注入）输入路径上的**非确定性**；QLH 层间接力的可复现性口径与绕开方案。
>
> 关联：[跨框架层接力重启评估](跨框架层接力重启评估-2026-09-15.md) §7.10/§7.11（该非确定性正是在那里被发现并量化的）· [基线重写方案](基线重写方案-2026-09-16.md) §1.4（工程约束的来源）

## 0. 结论摘要

- **现象可稳定复现**：同一进程、同一 context、同一输入、`llama_memory_clear` 之后重复 decode，`llama_batch.embd` 路径的结果**逐位不等**（cosine 0.984–0.9999）；而 **token 路径逐位完全一致**。
- **已排除 7 类成因**（见 §6，每条都有实验）：未写入的 `inp->tokens`、多线程归约、KV 容量/未使用区域、残留 KV 内容、ggml 融合算子、Flash Attention、CPU repack。
- **已定位**：非确定性**只在 attention 层出现**（SSM 层逐位置完全一致，**分叉精确始于第一个 full-attention 层**），且**需要 >1 个 token**；结果由输入决定（全零输入 → 全零输出且确定），但会在**少数几个状态之间循环**。
- **未找到可直接修改的代码点** → 本轮**不提供上游修复补丁**。交付：① 可粘贴的 issue 材料（§4–§6 + §10 英文正文草稿）② pin 版本口径与**应用层绕开方案**（§8）③ 若上游修复或我们独立定位后的打补丁流程（§9）。
- **对 QLH 的口径**（已写入《跨框架层接力重启评估》§7.11.3 与《基线重写方案》§1.4）：判据用 argmax、接力实验固定单进程、跨进程"行为等价"与"逐位复现"分开声明。

## 1. 现象（What）

```text
同一进程 / 同一 context / 同一输入（5×2048 f32）/ 每次 decode 前 llama_memory_clear(mem, true)

#[1] token 路径（llama_batch.token）      → run0..run5 逐位完全一致      （cosine = 1.00000000）
#[2] embd  路径（llama_batch.embd），1 token → 逐位完全一致
#[3] embd  路径，2 / 3 / 5 token          → 逐位不等（cosine 0.984 – 0.9999）
     argmax 在所有情形下都稳定（= 11751），top-10 高度重合
```

- 抖动**不是**"每次完全随机"：多次运行的结果落在**少数几个值**上并有循环模式（例：`run0==run2`、`run1==run3==run4`）。
- 跨进程（同一命令独立跑 3 个进程）同样不一致（pairwise cosine 0.9946–0.9971）。
- 同进程内**两个不同 context** 各 decode 一次，也不一致（cosine 0.9885）。

## 2. 环境（Environment）

| 项 | 值 |
| --- | --- |
| commit | `6f04274cc145cf76aafc46eb8c1ea054ca221368`（2026-08-14；含 PR #27073 hidden-state 提取） |
| 构建 | `cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF` + `mingw32-make -j4`（MSYS2 **UCRT64** g++） |
| 运行 | Windows（`windows/amd64`），**纯 CPU 后端**（`n_gpu_layers = 0`），`n_threads = n_threads_batch = 1`（除注明处） |
| 模型 | `Qwen3.5-2B`（`qwen35` 架构：混合 **SSM/Gated Delta Net** + **每 4 层一个 full attention**），f16 GGUF；另用"裁掉前 4 层"的同源模型 |
| 注入数据 | `5 × 2048` f32（来自原模型 layer 3 输出的真实 hidden） |
| 上下文参数 | `n_ctx = 512`、`n_batch = n_ubatch = 512`（另做 n_ctx 扫描，见 §6） |

## 3. 最小复现（How）

### 3.1 复现探针（自包含，直接编译）

```cpp
// embd-nondet-probe.cpp —— 同一 context 重复 decode 同一 embd，报告逐位一致性
// 编译：g++ -std=c++17 -O2 -I <llama.cpp>/include -I <llama.cpp>/ggml/include embd-nondet-probe.cpp \
//        <build>/src/libllama.a <build>/ggml/src/ggml.a <build>/ggml/src/ggml-cpu.a <build>/ggml/src/ggml-base.a \
//        -lws2_32 -lgomp -static-libgcc -static-libstdc++ -o probe.exe
// 用法：probe.exe <model.gguf> <embd.f32> [mode] [N] [threads]
//   mode: repeat（embd 重复）/ tokens（token 路径对照）
//   embd.f32 = N_tokens × n_embd 的 little-endian float32
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double cosine_of(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += (double) a[i]*b[i]; na += (double) a[i]*a[i]; nb += (double) b[i]*b[i]; }
    return (na > 0 && nb > 0) ? dot / (std::sqrt(na)*std::sqrt(nb)) : 0.0;
}

int main(int argc, char ** argv) {
    const char * model_path = argv[1];
    const char * embd_path  = argv[2];
    const std::string mode  = argc > 3 ? argv[3] : "repeat";
    const int N = argc > 4 ? atoi(argv[4]) : 6;
    const int n_threads = argc > 5 ? atoi(argv[5]) : 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const int n_embd  = llama_model_n_embd_inp(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<float> raw;
    { FILE * f = fopen(embd_path, "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      raw.resize((size_t) sz / 4); (void) fread(raw.data(), 4, raw.size(), f); fclose(f); }
    const int n_tokens = (int) (raw.size() / (size_t) n_embd);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;   // 两者都要设：decode 走 n_threads_batch
    llama_context * ctx = llama_init_from_model(model, cp);

    std::vector<float> first; bool all_eq = true;
    for (int r = 0; r < N; ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);         // 关键：每次都清 memory
        llama_batch b = (mode == "tokens")
            ? llama_batch_init(n_tokens, 0, 1)
            : llama_batch_init(n_tokens, n_embd, 1);
        for (int i = 0; i < n_tokens; ++i) {
            if (mode == "tokens") { int32_t t; memcpy(&t, &raw[(size_t) i * n_embd], 4); b.token[i] = t % n_vocab; }
            b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }
        b.n_tokens = n_tokens;
        if (mode != "tokens") { memcpy(b.embd, raw.data(), raw.size() * sizeof(float)); }
        llama_decode(ctx, b);
        llama_batch_free(b);
        const float * lg = llama_get_logits_ith(ctx, n_tokens - 1);
        if (r == 0) { first.assign(lg, lg + n_vocab); }
        else {
            if (memcmp(first.data(), lg, (size_t) n_vocab * 4) != 0) { all_eq = false; }
            printf("  run0 vs run%d: cosine=%.8f bitwise_equal=%s\n", r,
                   cosine_of(first.data(), lg, n_vocab), memcmp(first.data(), lg, (size_t) n_vocab * 4) == 0 ? "YES" : "NO");
        }
    }
    printf("VERDICT: %s\n", all_eq ? "BITWISE DETERMINISTIC" : "NON-DETERMINISTIC");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
```

### 3.2 运行

```bash
# 2 个 token 就足够触发（1 个 token 不会）
python -c "import numpy as np; np.random.default_rng(0).standard_normal((5,2048), dtype=np.float32).tofile('embd5.f32')"
./probe.exe model.gguf embd5.f32 repeat 6 1     # → NON-DETERMINISTIC
head -c 8192 embd5.f32 > embd1.f32
./probe.exe model.gguf embd1.f32 repeat 6 1     # → BITWISE DETERMINISTIC
# 对照：token 路径（把同一文件前 4 字节当 int32 token 解释）
./probe.exe model.gguf embd5.f32 tokens 6 1    # → BITWISE DETERMINISTIC
```

> 说明：用**随机向量**即可复现，不依赖任何特定模型权重语义；本文档的数据则来自真实 hidden（5×2048）。

## 4. 实验矩阵与数据（Observed）

| # | 实验 | 结果 |
| --- | --- | --- |
| 1 | token 路径重复 6 次 | **逐位一致**（cosine = 1.00000000） |
| 2 | embd 注入 **1 token** 重复 6 次 | **逐位一致** |
| 3 | embd 注入 **2 token** 重复 6 次 | 逐位不等（cosine 0.9979–1.0，偶发整轮一致） |
| 4 | embd 注入 **3 token** 重复 4 次 | 逐位不等 |
| 5 | embd 注入 **5 token** 重复 6 次（连跑 3 次） | **逐位不等**（cosine 0.984–0.9999；argmax 恒为 11751） |
| 6 | 5 token + `n_threads = n_threads_batch = 1` | 仍不等 → 排除多线程 |
| 7 | 5 token + 显式清零未写入的 `inp->tokens`（改 `llm_graph_input_embd::set_input`） | 仍不等 → 排除该假设 |
| 8 | 5 token + 扫 `n_ctx ∈ {8,16,32,64,128,512}` | 全档不等 → 排除 KV 容量 |
| 9 | 5 token + `GGML_CPU_DISABLE_FUSION=1` | 仍不等 → 排除融合算子 |
| 10 | 5 token + `flash_attn_type = DISABLED` | 仍不等 → 排除 Flash Attention |
| 11 | 5 token + **全零输入** | **输出全零且逐位一致**（说明"非输入来源"只在非零输入时产生贡献） |
| 12 | 跨进程（同一命令独立 3 进程） | 不等（pairwise cosine 0.9946–0.9971） |
| 13 | 同进程、两个 context 各 decode 一次 | 不等（cosine 0.9885） |
| 14 | **逐层 × 逐位置**定位（`llama_get_hidden_state`） | **layer 0/1/2 逐位一致；layer 3 起不等** |
| 15 | 结果分布 | 落在少数几个值并循环（`run0==run2`、`run1==run3==run4` 等） |
| 16 | 重新编译同一源码后 | 现象依旧（连跑 3 次均不等；偶发某一轮"整轮一致"的巧合） |

## 5. 定位结论（Where）

1. **只在 `llama_batch.embd` 路径**（`ubatch->token == NULL`）出现；token 路径完全确定。
2. **需要 >1 个 token**：1 token 确定，2 token 起不确定。
3. **分叉始于第一个 full-attention 层**：在本模型（每 4 层一个 full attention）上，**layer 0–2（SSM 层）逐位置逐位一致**，**layer 3（= 第一个 full-attention 层）起不一致**。→ 问题与 **attention 的跨位置机制**（KV / mask / 其 buffer）强相关。
4. 结果**仍由输入决定**（全零 → 全零且确定），说明不是"无中生有的随机注入"，而是某个**与输入相乘**的量在两次 decode 之间发生了变化。
5. 变化量**有限且循环**（少数状态），且**跨进程/跨 context 都复现** → 更像"**某个 buffer/状态在两次 decode 之间被复用且残留**"，而不是"每次全新的随机值"。

## 6. 已排除项（Ruled out，逐条含证据）

| 假设 | 实验 | 结果 |
| --- | --- | --- |
| 未写入的 `inp->tokens` 被读 | 在 `set_input` 中对该 input 显式清零（§4 #7） | **排除**（仍不等） |
| 多线程浮点归约顺序 | `n_threads = n_threads_batch = 1`（§4 #6） | **排除** |
| KV cache 容量 / 未使用区域 | 扫 `n_ctx` 8→512（§4 #8） | **排除** |
| 读到未初始化/残留的 KV 内容 | 全零输入 → 全零输出（§4 #11） | **排除**（若读到残留，零输入也会产生非零输出） |
| ggml 融合算子 | `GGML_CPU_DISABLE_FUSION=1`（§4 #9） | **排除** |
| CPU Flash Attention | `flash_attn_type = DISABLED`（§4 #10） | **排除** |
| CPU repack（权重重排） | 该构建日志明确 `cannot be used with preferred buffer type CPU_REPACK, using CPU instead`（未启用） | **不适用** |

## 7. 候选假设（未坐实）与建议方向

1. **attention 相关 buffer 的残留/复用**：某处 buffer 或状态在两次 decode 之间未被完全重置，其残留值只在非零输入时被"乘进"结果。建议方向：在 `llama_context` 里核对 embd 路径下 attention 的 mask/KV 相关 buffer 的初始化与复用（对比 token 路径）。
2. **embd 路径图/buffer 布局与 token 路径不同**导致的 SIMD 路径差异：异常地表现为"非确定"（更可能是某 buffer 未初始化，恰好在不同布局下被不同内容填充）。
3. 建议上游提供**最小复现脚本**并把判据定为"同进程 + 清 memory + 同输入 ⇒ 逐位一致"（本仓库可提供该探针）。

## 8. 对 QLH 的影响与绕开方案（Actionable）

**影响面**：任何用 `llama_batch.embd` 注入 hidden 的用法（跨框架层接力、多模态嵌入、**模型分片的 hidden 传递**）都**不能假设逐位可复现**。

**绕开口径（已写入项目文档）**：

1. **判据用 argmax / top-k 重合**，不用 cosine 阈值；
2. **接力实验固定在单进程内完成**（`llama-relay-gen` 即此形态）；
3. 跨进程形态（A/B 两节点）必须把"**行为等价**"与"**逐位复现**"分开声明；
4. **增量式逐 token 注入**（每步只注入 1 个新 token）在本机表现为确定（1 token 注入逐位一致）——**pos > 0 的增量注入尚待验证**，若成立即可作为"可复现接力"的实现形态。

## 9. pin 版本与打补丁流程

**当前处置（根因未坐实，无可改代码点）**：

- **pin**：以 `6f04274cc145cf76aafc46eb8c1ea054ca221368` 为实验基线版本（本机 clone：`build/cross-framework-layer-poc/llama.cpp`），并在引用了该行为的文档里标注本问题。
- **不改上游**（《基线重写方案》§5 排除项第 6 条）。

**若上游修复、或我们后续独立定位到代码点**：

```bash
# 1) 在 clone 里形成补丁
cd build/cross-framework-layer-poc/llama.cpp
git diff > ../patches/embd-nondeterminism-<日期>.patch      # patches/ 已存在且 gitignored

# 2) 应用（换机器/换版本时）
git apply --check ../patches/embd-nondeterminism-<日期>.patch && git apply ../patches/...

# 3) 重编译并复验（判据：§3.1 探针 repeat 模式 → VERDICT: BITWISE DETERMINISTIC）
cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF
mingw32-make -C build-cpu -j4 llama
```

> 本轮尝试过的**未成功修复**：清零 `inp->tokens`（探针 A）—— 已回退，工作树恢复干净。

## 10. issue 正文草稿（英文，可直接粘贴）

```markdown
### Title
CPU backend: `llama_batch.embd` (embedding input) decoding is non-deterministic, while the token path is bitwise deterministic

### Summary
On the CPU backend, repeated identical decodes through the **embedding input path**
(`llama_batch.embd` set, `ubatch->token == NULL`) produce **different logits** — in the same
process, same context, same input, after `llama_memory_clear(mem, true)` between decodes.
The **token path** (`llama_batch.token`) is **bitwise identical** under exactly the same
conditions. `argmax` stays stable, so the practical impact is on *reproducibility*, not on
greedy generation.

### Environment
- commit: 6f04274cc145cf76aafc46eb8c1ea054ca221368
- build: `cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF` (MSYS2 UCRT64 g++)
- runtime: Windows amd64, CPU backend, `n_gpu_layers = 0`, `n_threads = n_threads_batch = 1`
- model: Qwen3.5-2B (`qwen35`: hybrid SSM + full attention every 4 layers), f16 GGUF
- ctx: `n_ctx=512`, `n_batch=n_ubatch=512`

### Steps to reproduce
1. Prepare one `embd.f32` file = `n_tokens × n_embd` little-endian float32 (random values are fine).
2. Build and run the minimal probe in §3.1 of this report (also pasted below), mode `repeat`:
   it decodes the same batch N times in the **same context**, calling
   `llama_memory_clear(llama_get_memory(ctx), true)` before each decode, and reports
   `bitwise_equal` for each run vs run 0.
3. Also run mode `tokens` (same file interpreted as int32 tokens) as the control.

### Observed
| case | result |
| --- | --- |
| token path, 6 repeats | bitwise identical (cosine 1.00000000) |
| embd path, **1** token, 6 repeats | bitwise identical |
| embd path, **2/3/5** tokens, 6 repeats | **bitwise different** (cosine 0.984–0.9999; `argmax` constant) |
| embd path, all-zero input | all-zero output, bitwise identical |
| embd path, separate processes | different across processes (pairwise cosine 0.9946–0.9971) |

### Additional localization (this report)
- **Per-layer comparison** via `llama_get_hidden_state`: layers before the **first full-attention
  layer** are bitwise identical; divergence starts exactly at that layer (for this model:
  layers 0–2 identical, layer 3 differs). SSM layers are deterministic.
- **Requires >1 token**: 1 token is deterministic, 2 tokens already are not.
- Results fall into a **small set of states that recur** (e.g. run0==run2, run1==run3==run4),
  suggesting **buffer/state reuse** rather than fresh randomness.

### Ruled out
- unwritten `inp->tokens` (explicitly zeroing that input in `llm_graph_input_embd::set_input` does not help)
- multi-threading (`n_threads = n_threads_batch = 1`)
- KV capacity / unused KV region (swept `n_ctx` 8…512)
- reading uninitialized/stale KV content (all-zero input yields all-zero output)
- ggml fused ops (`GGML_CPU_DISABLE_FUSION=1`)
- CPU Flash Attention (`flash_attn_type = DISABLED`)

### Expected behavior
Same context + same input + cleared memory ⇒ **identical** output, as on the token path
(and as any user would expect for a pure-CPU deterministic backend).

### Extra context
This surfaced while prototyping **layer-pipeline relay** (compute layers `0..k` with one model
instance and inject the resulting hidden state into a truncated model that computes `k+1..N`
via `llama_batch.embd`). Greedy generation still matches the non-relay baseline 16/16 steps,
but bit-exact reproducibility across processes is not achievable today.
```

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-16 | 新建：把 §7.10/§7.11 发现的 `embd` 非确定性整理成可提交的 issue 材料。含环境、最小复现探针（自包含源码）、16 项实验数据、逐层定位（分叉始于第一个 full-attention 层、需要 >1 token）、7 条已排除成因、候选假设、QLH 侧绕开口径、pin 版本与打补丁流程、英文 issue 正文草稿。**根因未坐实，本轮不提供上游修复补丁**；曾尝试的"清零 `inp->tokens`"已回退，工作树干净。 |
