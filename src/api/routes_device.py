"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['auto_configure', 'get_device_profile', 'select_gpu']}

async def get_device_profile():
    """
    获取完整设备画像。

    包含 CPU / RAM / GPU / 磁盘 / OS 信息，
    设备档位、评分、推荐配置、警告。
    启动时自动检测一次，后续请求返回缓存。
    """
    if _api_module.device_profile is None:
        import asyncio

        await asyncio.to_thread(_api_module._device_profile_ready.wait, 15)
        if _api_module.device_profile is None:
            raise _api_module.HTTPException(503, "设备画像仍在检测中，请稍后重试")
    return _api_module.device_profile


async def auto_configure():
    """
    根据设备画像自动应用推荐配置。

    更新 KV 缓存大小、序列长度、生成参数等运行时配置。
    不重新加载模型（如需切换量化精度，请手动调用 /api/models/load）。
    """
    if _api_module.device_profile is None:
        try:
            profiler = _api_module.get_profile()
            _api_module.device_profile = profiler.to_dict()
        except Exception as e:
            raise _api_module.HTTPException(500, f"设备检测失败: {e}")
    _api_module.scheduler.update_local_device_profile(_api_module.device_profile)

    rec = _api_module.device_profile.get("recommendations", [])
    warnings = _api_module.device_profile.get("warnings", [])
    tier = _api_module.device_profile.get("tier", "laptop")
    score = _api_module.device_profile.get("score_total", 50)

    # 从 device_profiler 获取推荐配置
    from device_profiler import DeviceProfiler
    profiler = _api_module.get_profile()
    config = profiler.recommend_config()

    # 应用 KV 缓存配置（如果尚未加载模型，则更新默认值）
    import config as cfg
    cfg.PAGE_SIZE = config["page_size"]
    cfg.MAX_PAGE_NUM = config["max_pages"]
    cfg.MAX_SEQ_LEN = config["max_seq_len"]

    # 更新生成配置（设置档位上限）
    _api_module.model_host.generation_config["max_new_tokens"] = config["max_new_tokens"]
    _api_module.model_host.generation_config["tier_max_new_tokens"] = config["max_new_tokens"]

    # 如果 KV 缓存已存在，重建
    if _api_module.kv_cache and _api_module.model_host.model_loaded:
        _api_module.kv_cache.clear()
        from paged_kv_cache import PagedKVCache
        _api_module.kv_cache = PagedKVCache(
            page_size=config["page_size"],
            max_pages=config["max_pages"],
            device=_api_module.kv_cache.device,
            dtype=_api_module.kv_cache.dtype,
        )
        _api_module.logger.info(
            f"KV 缓存已重建: page_size={config['page_size']}, "
            f"max_pages={config['max_pages']}"
        )

    _api_module.logger.info(f"自适应配置已应用: {config['description']}")

    return {
        "status": "configured",
        "tier": tier,
        "score": score,
        "applied_config": config,
        "recommendations": rec,
        "warnings": warnings,
    }


async def select_gpu(req: _api_module.SelectGpuRequest):
    """
    切换推理 GPU。

    在集显（CPU 推理）和独显（CUDA）之间切换。
    切换后需要重新加载模型才能生效。

    游戏本默认使用独显（CUDA 加速），用户可手动切换到集显（低功耗）。
    """
    if _api_module.device_profile is None:
        raise _api_module.HTTPException(400, "设备画像未就绪，请先调用 GET /api/device/profile")

    gpus = _api_module.device_profile.get("gpus", [])
    if req.gpu_index < 0 or req.gpu_index >= len(gpus):
        raise _api_module.HTTPException(
            400,
            f"无效的 GPU 序号: {req.gpu_index}。"
            f"可用范围: 0-{len(gpus) - 1}（共 {len(gpus)} 个 GPU）",
        )

    # 更新 profiler 中的选中 GPU
    from device_profiler import get_profile
    profiler = get_profile()
    if not profiler.select_gpu(req.gpu_index):
        raise _api_module.HTTPException(500, "GPU 切换失败")

    # 更新缓存的 device_profile
    _api_module.device_profile = profiler.to_dict()
    _api_module.scheduler.update_local_device_profile(_api_module.device_profile)

    selected = gpus[req.gpu_index]
    _api_module.logger.info(
        f"GPU 已切换: [{req.gpu_index}] {selected['name']} "
        f"({selected['gpu_type']}, CUDA: {selected['cuda_available']})"
    )

    return {
        "status": "switched",
        "selected_gpu_index": req.gpu_index,
        "selected_gpu": {
            "name": selected["name"],
            "gpu_type": selected["gpu_type"],
            "cuda_available": selected["cuda_available"],
            "vram_total_gb": selected["vram_total_gb"],
        },
        "device": profiler.recommend_config()["device"],
        "warning": (
            "切换 GPU 后需要重新加载模型才能生效。"
            if _api_module.model_host.model_loaded
            else None
        ),
    }


def register_routes() -> None:
    router.add_api_route("/api/device/profile", get_device_profile, methods=["GET"])
    router.add_api_route("/api/device/auto-configure", auto_configure, methods=["POST"])
    router.add_api_route("/api/device/select-gpu", select_gpu, methods=["POST"])
