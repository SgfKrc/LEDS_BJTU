"""
单元测试 — 设备画像检测
=======================
使用 DeviceProfiler.mock_*() 模拟不同设备环境，
测试档位分级、评分计算、推荐配置生成。
无需真实硬件。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from device_profiler import DeviceProfiler, DeviceTier


# ================================================================
# 设备档位枚举测试
# ================================================================

class TestDeviceTier:
    """测试设备档位枚举"""

    def test_all_tiers_exist(self):
        """应包含 5 个档位"""
        tiers = list(DeviceTier)
        assert len(tiers) == 5

    def test_tier_values(self):
        """档位值应正确"""
        assert DeviceTier.WORKSTATION.value == "workstation"
        assert DeviceTier.LAPTOP.value == "laptop"
        assert DeviceTier.ULTRABOOK.value == "ultrabook"
        assert DeviceTier.EDGE.value == "edge"
        assert DeviceTier.MOBILE.value == "mobile"

    def test_tier_labels(self):
        """档位标签应为中文"""
        assert "工作站" in DeviceTier.WORKSTATION.label
        assert "游戏本" in DeviceTier.LAPTOP.label
        assert "轻薄本" in DeviceTier.ULTRABOOK.label
        assert "边缘" in DeviceTier.EDGE.label
        assert "移动" in DeviceTier.MOBILE.label

    def test_tier_order(self):
        """档位应按性能降序排列"""
        tiers = list(DeviceTier)
        assert tiers[0] == DeviceTier.WORKSTATION
        assert tiers[-1] == DeviceTier.MOBILE


# ================================================================
# 模拟设备测试
# ================================================================

class TestMockDevices:
    """测试模拟设备"""

    def test_mock_mobile(self):
        """模拟移动设备应有正确的属性"""
        p = DeviceProfiler.mock_mobile()
        d = p.to_dict()

        assert d["tier"] == "mobile"
        assert d["ram"]["total_gb"] <= 4
        assert d["gpu"]["cuda_available"] is False
        # ARM 平台
        assert "arm" in d["platform"]["machine"].lower() or "aarch" in d["platform"]["machine"].lower()

    def test_mock_edge(self):
        """模拟边缘设备"""
        p = DeviceProfiler.mock_edge()
        d = p.to_dict()

        assert d["tier"] == "edge"
        assert 4 <= d["ram"]["total_gb"] <= 8
        assert d["gpu"]["cuda_available"] is False

    def test_mock_devices_have_all_fields(self):
        """模拟设备应有完整的字段"""
        for mock_fn in [DeviceProfiler.mock_mobile, DeviceProfiler.mock_edge]:
            p = mock_fn()
            d = p.to_dict()

            # 必要顶层字段
            for key in ["tier", "tier_label", "tier_icon", "score_total",
                         "cpu", "ram", "gpu", "disk", "platform",
                         "recommendations", "warnings"]:
                assert key in d, f"{mock_fn.__name__}: 缺少字段 {key}"

    def test_mock_mobile_recommends_cpu(self):
        """移动设备应推荐 CPU 推理"""
        p = DeviceProfiler.mock_mobile()
        config = p.recommend_config()
        assert config["device"] == "cpu"
        assert config["quant_type"] == "int4"

    def test_mock_edge_recommends_cpu(self):
        """边缘设备应推荐 CPU 推理"""
        p = DeviceProfiler.mock_edge()
        config = p.recommend_config()
        assert config["device"] == "cpu"


# ================================================================
# 评分算法测试
# ================================================================

class TestScoring:
    """测试设备评分算法"""

    def test_mobile_score_range(self):
        """移动设备评分应在 0-15"""
        p = DeviceProfiler.mock_mobile()
        score = p.score
        assert 0 <= score <= 15, f"移动设备评分应在 0-15，实际: {score}"

    def test_edge_score_range(self):
        """边缘设备评分应在 0-25"""
        p = DeviceProfiler.mock_edge()
        score = p.score
        assert 0 <= score <= 25, f"边缘设备评分应在 0-25，实际: {score}"

    def test_score_is_float(self):
        """评分应为浮点数"""
        for mock_fn in [DeviceProfiler.mock_mobile, DeviceProfiler.mock_edge]:
            p = mock_fn()
            assert isinstance(p.score, (int, float))


# ================================================================
# 推荐配置测试
# ================================================================

class TestRecommendConfig:
    """测试自适应配置推荐"""

    def test_mobile_config_bounds(self):
        """移动设备配置应有合理的上限"""
        p = DeviceProfiler.mock_mobile()
        config = p.recommend_config()

        assert config["max_new_tokens"] <= 256
        assert config["max_seq_len"] <= 1024
        assert config["page_size"] <= 64
        assert config["max_pages"] <= 64

    def test_edge_config_bounds(self):
        """边缘设备配置应有合理的上限"""
        p = DeviceProfiler.mock_edge()
        config = p.recommend_config()

        assert config["max_new_tokens"] <= 512
        assert config["max_seq_len"] <= 2048
        assert config["page_size"] <= 128

    def test_config_has_description(self):
        """推荐配置应包含描述"""
        for mock_fn in [DeviceProfiler.mock_mobile, DeviceProfiler.mock_edge]:
            p = mock_fn()
            config = p.recommend_config()
            assert "description" in config
            assert len(config["description"]) > 0

    def test_config_all_required_keys(self):
        """推荐配置应包含所有必要字段"""
        required_keys = ["quant_type", "page_size", "max_pages",
                          "max_seq_len", "max_new_tokens",
                          "use_compile", "device", "description"]
        for mock_fn in [DeviceProfiler.mock_mobile, DeviceProfiler.mock_edge]:
            p = mock_fn()
            config = p.recommend_config()
            for key in required_keys:
                assert key in config, f"{mock_fn.__name__}: 缺少 {key}"


# ================================================================
# to_dict 序列化测试
# ================================================================

class TestToDict:
    """测试设备画像序列化"""

    def test_to_dict_is_json_serializable(self):
        """to_dict() 输出应可 JSON 序列化"""
        import json
        for mock_fn in [DeviceProfiler.mock_mobile, DeviceProfiler.mock_edge]:
            p = mock_fn()
            d = p.to_dict()
            # 不应抛出异常
            serialized = json.dumps(d, ensure_ascii=False)
            assert len(serialized) > 0

    def test_to_dict_cpu_fields(self):
        """CPU 信息应完整"""
        p = DeviceProfiler.mock_edge()
        d = p.to_dict()
        cpu = d["cpu"]
        for key in ["physical_cores", "logical_cores", "freq_max_mhz", "freq_mhz"]:
            assert key in cpu, f"CPU 缺少字段 {key}"

    def test_to_dict_ram_fields(self):
        """RAM 信息应完整"""
        p = DeviceProfiler.mock_edge()
        d = p.to_dict()
        ram = d["ram"]
        assert "total_gb" in ram
        assert ram["total_gb"] > 0

    def test_to_dict_gpu_fields(self):
        """GPU 信息应完整"""
        p = DeviceProfiler.mock_edge()
        d = p.to_dict()
        gpu = d["gpu"]
        for key in ["name", "vram_total_gb", "cuda_available", "is_integrated"]:
            assert key in gpu, f"GPU 缺少字段 {key}"
