/*
 * qlh_keep_head.c — keep-head 上游的 C shim（P2「C 路线」的接入点）
 * ==================================================================
 *
 * 为什么需要 shim，而不是直接在 Python 里 ctypes 调 libllama：
 * `llama_context_params` / `llama_model_params` 是**按值传参的大结构体**，其字段布局
 * 随 llama.cpp 版本变动。用 PyPI `llama-cpp-python` 的 ctypes 声明去调**自建**的
 * libllama，实测报 `llama_init_from_model: failed to initialize the context:
 * Unsupported ctx type` —— 即写入的字段错位（把 n_ctx 写到了 DLL 眼中的 `type`）。
 * 本 shim 用**与被调 DLL 同一份 `include/llama.h`** 编译，从根上消除该风险；
 * Python 侧只剩 4 个平凡签名的函数。
 *
 * 语义（与 `src/llama_keep_head.py` 的文档一致）：
 *   mode 0 = nextn   → 模型**自身的最后一层输出**（`output_norm` 之前）。配 head 裁层工件
 *                      （保留 blk.0..K-1）⇒ 只跑 K 层，正是 relay 上游该交的 hidden。
 *   mode 1 = layer_inp → **第 cut_layer 层的输入**（= 前 cut_layer 层的输出）。配整模工件
 *                      可验证语义，但会跑满全部层。
 *
 * 编译（Windows / MinGW，见 scripts/model_tools/build_keep_head_shim.ps1）：
 *   gcc -shared -O2 -o qlh_keep_head.dll qlh_keep_head.c \
 *       -I<llama.cpp>/include -L<llama.cpp>/build/src -lllama
 *
 * ⚠️ 依赖上游实验性形态的 API（`llama_set_embeddings_layer_inp` /
 * `llama_get_embeddings_nextn_ith` 等），由
 * `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch` 导出；升级 llama.cpp
 * 时必须重新核对（见 `scripts/model_tools/llama_quantize.lock.json` 的 `marker`）。
 */

#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct qlh_keep_head {
    struct llama_model   *model;
    struct llama_context *ctx;
    int n_embd;
    int n_layer;
    int mode;        /* 0 = nextn, 1 = layer_inp */
    int cut_layer;
} qlh_keep_head;

static void qlh_set_err(char *err, size_t errlen, const char *msg) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen, "%s", msg);
    }
}

/* 返回值：句柄；失败返回 NULL 并把原因写进 err。
 * `n_seq_max` ★ P3：context 的并行序列上限（多序列数据流所必需）；<=0 视为 1（旧行为）。 */
void * qlh_kh_load(const char * model_path,
                   int32_t n_ctx, int32_t n_threads, int32_t n_batch,
                   int32_t n_seq_max,
                   int32_t mode, int32_t cut_layer,
                   int32_t * out_n_embd, int32_t * out_n_layer,
                   char * err, size_t errlen) {
    if (model_path == NULL) {
        qlh_set_err(err, errlen, "model_path is NULL");
        return NULL;
    }
    if (mode != 0 && mode != 1) {
        qlh_set_err(err, errlen, "mode must be 0 (nextn) or 1 (layer_inp)");
        return NULL;
    }

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        qlh_set_err(err, errlen, "llama_model_load_from_file failed");
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) (n_ctx > 0 ? n_ctx : 4096);
    cparams.n_batch         = (uint32_t) (n_batch > 0 ? n_batch : 512);
    cparams.n_ubatch        = (uint32_t) (n_batch > 0 ? n_batch : 512);
    cparams.n_threads       = (n_threads > 0 ? n_threads : 4);
    cparams.n_threads_batch = (n_threads > 0 ? n_threads : 4);
    cparams.n_seq_max       = (uint32_t) (n_seq_max > 0 ? n_seq_max : 1);

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        llama_model_free(model);
        qlh_set_err(err, errlen, "llama_init_from_model failed");
        return NULL;
    }

    const int n_layer = llama_model_n_layer(model);
    if (mode == 1 && (cut_layer < 0 || cut_layer >= n_layer)) {
        llama_free(ctx);
        llama_model_free(model);
        qlh_set_err(err, errlen, "cut_layer out of range for layer_inp mode");
        return NULL;
    }

    qlh_keep_head * handle = (qlh_keep_head *) calloc(1, sizeof(qlh_keep_head));
    if (handle == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        qlh_set_err(err, errlen, "out of memory");
        return NULL;
    }
    handle->model     = model;
    handle->ctx       = ctx;
    handle->n_embd    = llama_model_n_embd(model);
    handle->n_layer   = n_layer;
    handle->mode      = mode;
    handle->cut_layer = cut_layer;

    if (mode == 0) {
        llama_set_embeddings_nextn(ctx, true, false);   /* unmasked: rows 按 token 稠密 */
    } else {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) cut_layer, true);
    }

    if (out_n_embd  != NULL) { *out_n_embd  = handle->n_embd;  }
    if (out_n_layer != NULL) { *out_n_layer = handle->n_layer; }
    return handle;
}

/* 从当前 batch 结果里取出本次前向的 hidden（按模式），写进 out。返回 0 或负错误码。 */
static int32_t qlh_extract_hidden(qlh_keep_head * handle, int32_t n_tokens, float * out) {
    if (handle->mode == 1) {
        const float * ptr = llama_get_embeddings_layer_inp(handle->ctx,
                                                          (uint32_t) handle->cut_layer);
        if (ptr == NULL) {
            return -3;
        }
        memcpy(out, ptr, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));
        return 0;
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * ptr = llama_get_embeddings_nextn_ith(handle->ctx, i);
        if (ptr == NULL) {
            return -4;
        }
        memcpy(out + (size_t) i * (size_t) handle->n_embd, ptr,
               (size_t) handle->n_embd * sizeof(float));
    }
    return 0;
}

/* 统一标注：**每个 token 都要标输出**。
 * 末层 hidden（`t_h_nextn`）只对「有输出的行」计算，若只标最后一行，取 n_tokens 行就会踩到
 * GGML_ASSERT "tensor read out of bounds"（实测踩过）。上游不需要 logits，但需要行数完整。 */
static void qlh_fill_common(struct llama_batch * batch, int32_t n_tokens, int32_t n_past) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch->pos[i]       = n_past + i;
        batch->n_seq_id[i]  = 1;
        batch->seq_id[i][0] = 0;
        batch->logits[i]    = 1;
    }
    batch->n_tokens = n_tokens;
}

/* ★ P3：多序列显式绑定 —— 每个 token 自带 `n_seq_id / seq_id / pos`，**不依赖隐式位置递增**。
 * 与主仓 `llama_engine.forward_layers_from_hidden(seq_ids=..., positions=...)` 同一契约；
 * 多序列交错推进时，位置由调用方决定（每序列可各自递增）。
 * ⚠️ `positions == NULL` 时回落到 `n_past + i`（单序列增量语义）—— 少了这一项，
 *    单序列 decode 的第二步会从位置 0 重放，直接触发
 *    "tokens ... have inconsistent sequence positions" 而 decode 失败（实测踩过）。 */
static void qlh_fill_explicit(struct llama_batch * batch, int32_t n_tokens, int32_t n_past,
                              const int32_t * n_seq_id, const int32_t * seq_ids,
                              const int32_t * positions) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t per_token = (n_seq_id == NULL) ? 1 : n_seq_id[i];
        batch->n_seq_id[i]  = per_token > 0 ? per_token : 1;
        batch->seq_id[i][0] = (seq_ids == NULL) ? 0 : seq_ids[i];
        batch->pos[i]       = (positions == NULL) ? (n_past + i) : positions[i];
        batch->logits[i]    = 1;
    }
    batch->n_tokens = n_tokens;
}

/* 前向：跑 tokens（从 n_past 起），把结果写进 out（容量 n_tokens * n_embd 个 float）。
 * 返回 0 成功；否则为负的错误码（-1 参数错，-2 decode 失败，-3/-4 取 hidden 失败）。 */
int32_t qlh_kh_forward(void * handle_void,
                       const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                       float * out) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || tokens == NULL || out == NULL || n_tokens <= 0) {
        return -1;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
    }
    qlh_fill_common(&batch, n_tokens, n_past);

    const int32_t rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return -2;
    }
    return qlh_extract_hidden(handle, n_tokens, out);
}

/* ★ 中间段能力：吃 hidden（`embd` 注入）→ 吐 hidden（本模型末层 / 第 cut_layer 层输入）。
 * 这是「1 个 torch 上游 + n 个 llama 下游」链式拼接的关键 —— 中间的 llama 段必须能
 * 既接受上游 hidden 又交出 hidden。返回码同 qlh_kh_forward（另有 -5 = embd 参数错）。 */
int32_t qlh_kh_forward_embd_seq(void * handle_void,
                                const float * embd, int32_t n_tokens, int32_t n_past,
                                const int32_t * n_seq_id, const int32_t * seq_ids,
                                const int32_t * positions, float * out);

int32_t qlh_kh_forward_embd(void * handle_void,
                            const float * embd, int32_t n_tokens, int32_t n_past,
                            float * out) {
    return qlh_kh_forward_embd_seq(handle_void, embd, n_tokens, n_past, NULL, NULL, NULL, out);
}

/* ★ P3：多序列版 `embd` 前向 —— 显式 `n_seq_id / seq_ids / positions`（长度均为 n_tokens）。
 * 三个数组都可为 NULL（等价于旧的单序列、位置自 n_past 起递增）⇒ 向后兼容。
 * ⚠️ 需要 `qlh_kh_load(..., n_seq_max >= 序列数)`，否则 llama.cpp 拒绝 >0 的 seq_id。 */
int32_t qlh_kh_forward_embd_seq(void * handle_void,
                                const float * embd, int32_t n_tokens, int32_t n_past,
                                const int32_t * n_seq_id, const int32_t * seq_ids,
                                const int32_t * positions, float * out) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || embd == NULL || out == NULL || n_tokens <= 0) {
        return -5;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, 1);
    qlh_fill_explicit(&batch, n_tokens, n_past, n_seq_id, seq_ids, positions);
    memcpy(batch.embd, embd, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));

    const int32_t rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return -2;
    }
    return qlh_extract_hidden(handle, n_tokens, out);
}

int32_t qlh_kh_n_embd(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    return handle == NULL ? 0 : handle->n_embd;
}

int32_t qlh_kh_n_layer(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    return handle == NULL ? 0 : handle->n_layer;
}

/* ★ P3：清空 KV / recurrent 记忆。跨机服务在**同一进程**里服务多条连接时必须调用 ——
 * 否则新连接从位置 0 开始会与上一条连接留下的位置冲突（llama.cpp 报
 * "tokens ... have inconsistent sequence positions"，实测表现为远端 runner_failed）。 */
void qlh_kh_reset(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL) {
        return;
    }
    llama_memory_t mem = llama_get_memory(handle->ctx);
    if (mem != NULL) {
        llama_memory_clear(mem, true);
    }
}

void qlh_kh_close(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return;
    }
    if (handle->ctx != NULL) {
        llama_free(handle->ctx);
        handle->ctx = NULL;
    }
    if (handle->model != NULL) {
        llama_model_free(handle->model);
        handle->model = NULL;
    }
    free(handle);
}
