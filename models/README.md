# models/ — 模型文件存放目录

> ⚠️ **模型权重文件不在 Git 仓库中**，请按下方说明下载。
>
> 本文主要说明默认 Qwen-1.8B 示例工件。其他 Qwen/DeepSeek 实验模型由 `src/model_config.py` 注册，各模型必须按其格式与引擎要求单独准备。

---

## 下载模型

### 方式一：ModelScope（推荐，国内更快）

🔗 [https://modelscope.cn/models/Qwen/Qwen-1.8B-Chat](https://modelscope.cn/models/Qwen/Qwen-1.8B-Chat)

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
```

### 方式二：Hugging Face

🔗 [https://huggingface.co/Qwen/Qwen-1.8B-Chat](https://huggingface.co/Qwen/Qwen-1.8B-Chat)

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat
```

### 方式三：百度网盘（备用）

> 📎 网盘链接：[https://pan.baidu.com/s/1QnZXZb50ssZQIAuBQOKEUA?pwd=avne](https://pan.baidu.com/s/1QnZXZb50ssZQIAuBQOKEUA?pwd=avne)
>
> 提取码：avne

---

## 目录结构

下载完成后应包含以下文件：

```
models/qwen-1_8b-chat/
├── model-00001-of-00002.safetensors  (~1.9 GB)
├── model-00002-of-00002.safetensors  (~1.6 GB)
├── model.safetensors.index.json       # 分片索引
├── config.json                        # 模型结构配置
├── configuration_qwen.py              # 模型配置类
├── tokenizer_config.json              # 分词器配置
├── qwen.tiktoken                      # 词表文件
├── tokenization_qwen.py               # 分词器实现
├── modeling_qwen.py                   # 模型定义 (trust_remote_code)
├── qwen_generation_utils.py           # 生成工具
├── cpp_kernels.py                     # 自定义 C++ kernel
├── generation_config.json             # 生成配置
├── LICENSE
├── NOTICE
└── README.md
```

---

## 说明

- **只需一份权重**：bitsandbytes 采用"加载时量化"，同一份 FP16 权重通过代码切换 INT4/INT8/FP16 模式
- **流水线节点**：参与同一次 PyTorch 层流水线的动态节点集合必须拥有经过校验的同一模型工件，再由 `load_layer_range()` 加载各自的连续层段；节点数量不固定
- **GGUF 节点**：当前使用 llama.cpp 做本地完整推理，不能因为模型显示名称相同就加入 PyTorch 层流水线
- **命名注意**：目录名用 `1_8b` 而非 `1.8b`，避免 Python 把点号误解析为包分隔符

## 量化切换

```python
# 在 config.py 中切换，无需改动模型文件：
QUANT_TYPE = "int4"   # fp16 | int8 | int4
```
