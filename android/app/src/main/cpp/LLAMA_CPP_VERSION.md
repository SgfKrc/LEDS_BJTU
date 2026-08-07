# llama.cpp 版本说明（Android Full 原生构建）

- 上游仓库：`https://github.com/ggml-org/llama.cpp`
- 固定 commit：`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`
- 接入方式：Git submodule（2026-08-08 由 vendored 源码树迁移，见 docs/Android llama.cpp Submodule迁移方案.md）
- 用途：Android full-local GGUF 推理运行时（JNI：`qlh_llama_jni.cpp`，CMake `add_subdirectory(llama.cpp build-llama)`）
- Android 验证：迁移后 Full Release 构建通过（2026-08-08，assembleFullRelease BUILD SUCCESSFUL）；Lite 不依赖本目录。真机回归待做（结果后补）。
- 项目补丁：无（迁移前差异校验通过，vendored 与上游 commit 完全一致）
