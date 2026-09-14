"""配置编辑屏（ModalScreen）：在 TUI 内编辑 .env.docagent。

- 预填：现有文件内容；不存在时用官方模板。
- Ctrl+S 保存（写 .env.docagent，自动 .bak 备份 + chmod 600）→ 保存后自动跑
  `docagent config` 校验并显示结果（脱敏）。
- Esc 取消。**写权限仅限 .env.docagent**。
"""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from .config_check import env_path, env_template, validate_env, write_env


class ConfigEditor(ModalScreen):
    BINDINGS = [
        Binding("ctrl+s", "save", "保存", priority=True),
        Binding("escape", "cancel", "取消"),
    ]

    def __init__(self, root: Path, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.result: dict | None = None

    def compose(self) -> ComposeResult:
        path = env_path(self.root)
        if path.is_file():
            try:
                initial = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                initial = env_template(self.root)
            subtitle = f"编辑 {path}"
        else:
            initial = env_template(self.root)
            subtitle = f"新建 {path}（已预填官方模板）"
        with Vertical(id="editor-box"):
            yield Static(f"== docagent 配置 ==\n{subtitle}\nCtrl+S 保存（自动备份 .bak）· Esc 取消", id="editor-title", markup=False)
            yield TextArea(initial, id="editor-area")
            with Horizontal(id="editor-actions"):
                yield Button("保存", variant="primary", id="save")
                yield Button("取消", id="cancel")
            yield Static("", id="editor-status", markup=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        else:
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss()

    def action_save(self) -> None:
        content = self.query_one("#editor-area", TextArea).text
        status = self.query_one("#editor-status", Static)
        result = {"saved": write_env(self.root, content)}
        if result["saved"].get("ok"):
            validation = validate_env(self.root)
            result["validation"] = validation
            if validation.get("ok"):
                status.update("已保存 ✓ · 校验通过（docagent config）")
            else:
                status.update(f"已保存（备份 .bak）· 校验未通过：{validation.get('error')}")
        else:
            result["validation"] = None
            status.update(f"保存失败：{result['saved'].get('error')}")
        self.result = result
