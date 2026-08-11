# SD 1.5 离线资产包与签名源站发布计划

> 状态：规划（阶段 1 进行中；阶段 2 待 GPU 空窗）
>
> 更新日期：2026-08-11
> 适用范围：SD 1.5 系列五个已冻结资产的正式离线资产包制作、签名源站发布与离线导入验收；SD-N2 发布资产门收尾
>
> 总计划入口：[总体下一步计划](总体下一步计划.md) L4-SD1.5；专项背景见 [SD 1.5引擎与分布式图像生成实施计划](SD%201.5引擎与分布式图像生成实施计划.md)

## 1. 背景与目标

SD 1.5 系列五个资产已全部冻结并下载到本机（原版 SD 1.5、90s DreamBooth、IP-Adapter、inpainting、InstructPix2Pix），质量自动门、双人目视与许可复核（openrail / CreativeML OpenRAIL-M 允许附副本分发）均已完成。按 SD 专项"提供模型的三种方式"之离线方式与 SD-N2 完成判据，唯一缺口是**正式离线资产包及发布流程**——发布资产门未过前 SD-N2 不得标记 Completed。

本计划把该缺口拆为两个阶段，避免与并行中的 gemma-4 开发产生脏读写冲突。

## 2. 资产清单（均已冻结，来源 `src/diffusion/assets.py`）

| asset_id | 名称 | 许可证 | 本地目录 | 包级规模 |
|---|---|---|---|---|
| `sd15_original_v1` | 原版 SD 1.5 | creativeml-openrail-m | `models/sd15-original-v1` | ~2.74 GB |
| `sd15_90s_retrovers_v1` | 90s DreamBooth | openrail | `models/sd15-90s-retrovers-v1` | ~4.87 GB |
| `sd15_ip_adapter_v1` | IP-Adapter | apache-2.0 | `models/sd15-ip-adapter-v1` | ~2.57 GB |
| `sd15_inpaint_v1` | SD inpainting | creativeml-openrail-m | `models/sd15-inpaint-v1` | ~2.74 GB |
| `sd15_instruct_pix2pix_v1` | InstructPix2Pix | mit | `models/sd15-instruct-pix2pix-v1` | ~2.74 GB |

打包元数据直接复用 `DiffusionAssetSpec`（revision、逐文件 size+sha256、preset、model_card_url），不新增事实源。

## 3. 阶段划分与冲突边界

### 阶段 1（零冲突，可随时执行）
交付：打包脚本、许可证原文入库、serve.py 签名源站 `kind=sd15-asset`、单元测试。

冲突边界：**不修改 `src/` 任何现有文件、不运行 GPU、不动 `.venv-packaging-cuda` 依赖**；新增文件限 `scripts/package_sd15_assets.py`、`packaging/sd15-licenses/`、`packaging/serve.py`（kind 分类）、`tests/test_sd15_asset_packaging.py`。serve.py 的改动与 gemma-4（api_server/model_config/llama_engine）零交集。

### 阶段 2（需 GPU 空窗，gemma-4 开发告一段落后执行）
交付：五资产真实打包与离线导入闭环验收。

冲突边界：逐资产 1–4 step 离线冒烟占用 RTX 4060 显存（峰值 reserved ~3.4–3.6 GB），必须避开 gemma-4 的 GPU 推理窗口；打包 IO 约 13 GB，输出到 `build/sd15-assets/`（不写 `models/` 内部）。

## 4. 离线包结构（每资产一包）

```text
QLH-SD15-Assets-<asset_id>-v0.1.0.zip
├── manifest.json          # asset_id/artifact_id/name/repo/revision/preset/license_id/
│                          #   model_card_url/files[{path,size,sha256}]/包级总大小
├── LICENSE.txt            # 许可证原文（packaging/sd15-licenses/ 入库副本）
├── MODEL_CARD.md          # 本地 README.md（模型卡）原样携带
├── IMPORT.md              # 解包到 models/<local_dir> 后由 assets.py 自动发现与校验
└── <模型文件…>             # spec.files 全部文件（逐文件校验后打包）
```

包旁生成 `QLH-SD15-Assets-<asset_id>-v0.1.0.zip.sha256` 侧车。

文件名规则：`QLH-SD15-Assets-<asset_id>-v0.1.0.zip`（serve.py 分类识别前缀 `qlh-sd15-assets-`）。

## 5. 签名源站接入（serve.py）

- `_classify_update_asset` 增加分类：文件名匹配 `qlh-sd15-assets-*` → `kind="sd15-asset"`，platform/arch 记 `any`；
- 发布物放 `packaging/dist/` 扫描目录，进入 `/latest.json` 清单，经 UP-N2 Ed25519 签名门控；
- 消费者侧（导入工具/下载器）按 manifest 的 kind 过滤与验签，禁止未验签资产进入导入路径。

## 6. 验收口径

### 阶段 1（自动化）
- 打包脚本：合成资产 fixture 打包 → manifest 字段齐全、逐文件 SHA 与 spec 一致、LICENSE/MODEL_CARD/IMPORT 齐备、`.sha256` 侧车正确、缺文件/哈希不符 fail-closed（`tests/test_sd15_asset_packaging.py`）；
- serve.py：`_classify_update_asset` 对 `qlh-sd15-assets-*` 返回 `sd15-asset`；manifest 构建含该类资产（复用现有 serve 测试）。

### 阶段 2（实机）
- 五资产真实打包：打包前逐文件 size+SHA 与 spec 一致，包级 SHA 侧车正确；
- 离线导入闭环：解包到干净临时目录 → `verify_asset_directory` 通过 → 服务自动发现注册 → **Hub 强制离线**下单资产 1–4 step 真实生成冒烟，全程无网络访问；
- SD-N2 标记 `Completed`，专项计划与总计划同步登记。

## 7. 依赖与风险

| 项 | 说明 |
|---|---|
| 磁盘 | 阶段 2 需 `build/sd15-assets/` 约 13 GB 余量（与 gemma 模型 7.6 GB 错峰确认） |
| GPU | 阶段 2 冒烟与 gemma-4 推理互斥，按空窗执行 |
| venv | 全程使用 `.venv-packaging-cuda`（diffusers 0.35.2 锁定），不升级依赖 |
| 网络 | 许可证原文一次入库（packaging/sd15-licenses/），打包与导入全程离线 |

## 8. 变更记录

| 日期 | 事件 |
|---|---|
| 2026-08-11 | 建立本文档；阶段 1 完成（打包脚本、许可证入库×3 + OpenRAIL 待补、serve.py `sd15-asset` kind、单元测试 11/11、serve 回归 16/16） |
