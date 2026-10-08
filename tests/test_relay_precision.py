"""`src/relay_precision.py` 的回归 —— 两侧精度对齐判据与装配期检查。

这些断言钉住的是**结论本身**（判据表与实测一致、未测组合不猜、不改路由语义），
不是实现细节：任何一条红都意味着"误差优化结论"与生产行为脱钩了。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import relay_precision as P  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """模块级登记是进程共享的 ⇒ 每个用例前后都清干净，并摘掉严格开关。"""
    monkeypatch.delenv(P.STRICT_ENV, raising=False)
    P.reset_relay_precision_notes()
    yield
    P.reset_relay_precision_notes()


class _SpyLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message):
        self.warnings.append(str(message))


# ------------------------------------------------------------------ 档位识别

def test_gguf_file_type_registered_values_match_real_artifacts():
    """实测确认的两个值：f16 工件 `general.file_type='1'`、Q4_K_M 工件 `'15'`。"""
    assert P.precision_from_gguf_file_type("1") == "f16"
    assert P.precision_from_gguf_file_type("15") == "q4_k_m"
    # llama-cpp-python 把它暴露成**字符串**（实测）⇒ int 也要能收。
    assert P.precision_from_gguf_file_type(15) == "q4_k_m"


def test_gguf_file_type_unregistered_value_is_not_guessed():
    """未登记的枚举值必须回 `ftype:<n>` 而不是猜一个档名。"""
    assert P.precision_from_gguf_file_type("999") == "ftype:999"
    assert P.precision_from_gguf_file_type("not-a-number") == "unknown"


def test_precision_from_path_recognizes_dequant_dirs():
    assert P.precision_from_path(r"build\keephead\qwen25-05b-f16-dequant") == "f16"
    assert P.precision_from_path("build/keephead/qwen25-05b-q4km-dequant") == "q4_k_m"


def test_precision_from_path_does_not_misfire_on_model_names():
    """回归：`qwen2.5-0.5b-instruct` 这类名字**不能**被误判成某个量化档。"""
    assert P.precision_from_path("models/qwen2.5-0.5b-instruct") == "unknown"
    assert P.precision_from_path("models/qwen25-05b-f16.gguf") == "f16"
    assert P.precision_from_path(None) == "unknown"


def test_normalize_precision_folds_aliases_but_keeps_config_intent():
    assert P.normalize_precision("fp16") == "f16"
    assert P.normalize_precision("float32") == "f32"
    assert P.normalize_precision("Q4_K_M") == "q4_k_m"
    assert P.normalize_precision("q8_0") == "int8"
    # ⚠️ `int4` 是 config.QUANT_TYPE 的**意图**，不是具体网格 ⇒ **不得**被猜成 q4_k_m。
    assert P.normalize_precision("int4") == "int4"
    assert P.normalize_precision("") == "unknown"
    assert P.normalize_precision(None) == "unknown"


# ------------------------------------------------------------------ 判据表

def test_measured_pairs_match_the_recorded_experiment():
    """判据表的三个数值必须与台账 JSON **逐位**一致（改了实验就得改这里）。"""
    assert P.MEASURED_PRECISION_PAIRS[("f32", "q4_k_m")][0] == pytest.approx(1.963e-01, rel=1e-3)
    assert P.MEASURED_PRECISION_PAIRS[("q4_k_m", "q4_k_m")][0] == pytest.approx(4.713e-03, rel=1e-3)
    assert P.MEASURED_PRECISION_PAIRS[("f16", "f16")][0] == pytest.approx(3.325e-04, rel=1e-3)


def test_measured_pairs_are_verified_against_the_artifact_json():
    """★ 直接读台账 JSON 交叉校验 —— 防止有人"顺手改判据数值"却不动实验。

    `build/` 属实验产物；缺文件时**跳过**（与仓库"缺工件型跳过"惯例一致），
    但存在时**必须**逐位对上。
    """
    import json
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for (_up, _down), (rel, _level, evidence) in P.MEASURED_PRECISION_PAIRS.items():
        path = root / P.EVIDENCE_DIR / evidence
        if not path.is_file():
            continue
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        assert rows, f"{evidence} 没有 rows"
        assert rel == pytest.approx(rows[0]["rel_err"], abs=0.0), (
            f"判据表 {evidence} 的 rel 与台账不符：表={rel!r} 台账={rows[0]['rel_err']!r}")
        checked += 1
    if checked == 0:
        pytest.skip("台账 JSON 不在（build/ 属实验产物）")


def test_evaluate_flags_mismatched_precision_as_critical():
    v = P.evaluate_relay_precision("fp32", "Q4_K_M")
    assert (v.upstream, v.downstream) == ("f32", "q4_k_m")
    assert v.level == P.LEVEL_CRITICAL
    assert v.aligned is False
    assert v.expected_rel == pytest.approx(1.963e-01, rel=1e-3)
    assert "同源" in v.advice


def test_evaluate_marks_same_source_f16_as_aligned():
    v = P.evaluate_relay_precision("f16-dequant", "f16")
    assert v.level == P.LEVEL_OK
    assert v.aligned is True
    assert v.expected_rel == pytest.approx(3.325e-04, rel=1e-3)


def test_evaluate_returns_unknown_for_unmeasured_pair_and_says_so():
    """⚠️ 核心纪律：未实测的组合**不许猜** —— 必须回 unknown 并要求先实测。"""
    v = P.evaluate_relay_precision("f32", "f16")
    assert v.level == P.LEVEL_UNKNOWN
    assert v.aligned is False
    assert v.expected_rel is None
    assert "未实测" in v.advice
    assert "xframe_dl_relay_positions" in v.advice


# ------------------------------------------------------------------ 登记与检查

def test_check_is_silent_until_both_sides_are_known():
    assert P.check_relay_precision() is None
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    assert P.check_relay_precision() is None  # 只知下游 ⇒ 不报错（只用一侧是正常形态）


def test_downstream_prefers_authoritative_metadata_over_path_and_config():
    P.note_downstream_precision(model_path="whatever-f16.gguf", quant_type="int4",
                                metadata={"general.file_type": "15"})
    assert P.relay_precision_report()["downstream"] == "q4_k_m"


def test_upstream_prefers_path_marker_over_quant_type_intent():
    """上游档位以**实际加载的目录**为准，`quant_type` 只是意图。"""
    P.note_upstream_precision(model_path="build/keephead/qwen25-05b-f16-dequant",
                              quant_type="int4")
    assert P.relay_precision_report()["upstream"] == "f16"


def test_mismatch_warns_exactly_once_and_does_not_raise_by_default():
    logger = _SpyLogger()
    P.note_upstream_precision(quant_type="fp32")
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    for _ in range(5):
        verdict = P.check_relay_precision(logger=logger)
        assert verdict is not None and verdict.level == P.LEVEL_CRITICAL
    assert len(logger.warnings) == 1, "同一档位对只应 WARN 一次（否则每步刷屏）"
    assert "未对齐" in logger.warnings[0]


def test_aligned_pair_never_warns():
    logger = _SpyLogger()
    P.note_upstream_precision(model_path="build/keephead/qwen25-05b-f16-dequant")
    P.note_downstream_precision(metadata={"general.file_type": "1"})
    for _ in range(3):
        assert P.check_relay_precision(logger=logger).level == P.LEVEL_OK
    assert logger.warnings == []


def test_strict_mode_fails_loud_on_every_call():
    P.note_upstream_precision(quant_type="fp32")
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    for _ in range(2):
        with pytest.raises(P.RelayPrecisionMismatch) as excinfo:
            P.check_relay_precision(strict=True)
        assert "未对齐" in str(excinfo.value)
    # 严格模式**不**因为"警告过了"就放行。


def test_strict_env_enables_fail_loud(monkeypatch):
    monkeypatch.setenv(P.STRICT_ENV, "1")
    P.note_upstream_precision(quant_type="fp32")
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    with pytest.raises(P.RelayPrecisionMismatch):
        P.check_relay_precision()


def test_reset_clears_notes_and_cache():
    P.note_upstream_precision(quant_type="fp32")
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    assert P.check_relay_precision() is not None
    P.reset_relay_precision_notes()
    assert P.check_relay_precision() is None
    assert P.relay_precision_report()["pending"] == "upstream"


def test_report_shape_is_stable():
    P.note_upstream_precision(quant_type="fp32")
    report = P.relay_precision_report()
    assert set(report) == {"upstream", "downstream", "verdict", "pending"}
    assert report["pending"] == "downstream"
    P.note_downstream_precision(metadata={"general.file_type": "15"})
    report = P.relay_precision_report()
    assert report["pending"] is None
    assert report["verdict"]["level"] == P.LEVEL_CRITICAL


def test_mismatch_error_is_a_runtime_error():
    """调用方可能用 `except RuntimeError` 兜底 ⇒ 保持这个继承关系。"""
    assert issubclass(P.RelayPrecisionMismatch, RuntimeError)
