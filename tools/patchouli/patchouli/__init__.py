"""Patchouli — 知识库/文档库管理 TUI（只读）。

PATCH-01：数据层（catalog）已完成——docs/ 扫描与元数据解析。
后续：书架 TUI（PATCH-02）→ 检索台（PATCH-03）→ 编目/审计集成（PATCH-04）→ 流通记录（PATCH-05）→ 统计收口（PATCH-06）。
"""
from .catalog import classify, parse_document, scan, summarize

__all__ = ["classify", "parse_document", "scan", "summarize"]
__version__ = "0.1.0"
