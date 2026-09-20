from scripts import llama_dependency_contract as contract


def test_llama_dependency_contract_is_consistent():
    result = contract.check_contract()

    assert result["ok"], result["failures"]
    assert result["default_cpu_gguf_version"] == "0.3.35"
    assert result["gemma_native_version"] == "0.3.28"


def test_edge_contract_has_no_torch_runtime_dependency():
    text = (contract.ROOT / "requirements-edge.txt").read_text(encoding="utf-8").lower()

    for package in contract.EDGE_FORBIDDEN:
        assert not any(
            line.strip().startswith(package)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ), package
