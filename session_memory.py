"""
轻量级会话记忆模块。

教学版设计目标：
  1. 用 session_id 区分不同用户/浏览器会话
  2. 每轮只保存用户问题、最终回答、工具调用摘要，避免上下文无限膨胀
  3. 采用内存存储，方便课堂演示；生产环境应替换为 Redis / 数据库
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass
class SessionTurn:
    question: str
    answer: str
    actions: list[dict[str, Any]] = field(default_factory=list)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[已截断]"


class SessionStore:
    """线程安全的内存会话存储。"""

    def __init__(self, max_turns: int = 6, max_answer_chars: int = 800, max_obs_chars: int = 240):
        self.max_turns = max_turns
        self.max_answer_chars = max_answer_chars
        self.max_obs_chars = max_obs_chars
        self._sessions: dict[str, list[SessionTurn]] = {}
        self._lock = RLock()

    def ensure(self, session_id: str | None = None) -> str:
        """返回可用 session_id；为空时自动创建。"""
        sid = session_id.strip() if isinstance(session_id, str) else ""
        if not sid:
            sid = uuid4().hex
        with self._lock:
            self._sessions.setdefault(sid, [])
        return sid

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def build_context(self, session_id: str | None) -> str:
        """将最近几轮历史压缩成给 LLM 看的上下文文本。"""
        if not session_id:
            return ""
        with self._lock:
            turns = list(self._sessions.get(session_id, []))[-self.max_turns:]

        if not turns:
            return ""

        lines = [
            "以下是同一 session 的历史对话摘要，可用于理解代词、延续问题和已查询过的数据；",
            "如果当前问题与历史无关，请不要强行引用历史。",
        ]
        for idx, turn in enumerate(turns, 1):
            lines.append(f"\n[历史轮次 {idx}]")
            lines.append(f"用户问题: {turn.question}")
            if turn.actions:
                action_summaries = []
                for action in turn.actions:
                    args = json.dumps(action.get("input", {}), ensure_ascii=False)
                    obs = _truncate(action.get("observation", ""), self.max_obs_chars)
                    action_summaries.append(f"{action.get('tool')}({args}) => {obs}")
                lines.append("工具调用摘要: " + "；".join(action_summaries))
            lines.append(f"助手回答: {_truncate(turn.answer, self.max_answer_chars)}")

        return "\n".join(lines)

    def append_turn(self, session_id: str, question: str, answer: str, steps: list[dict[str, Any]]) -> None:
        """保存一轮完整问答。"""
        actions: list[dict[str, Any]] = []
        for step in steps:
            if step.get("type") != "action":
                continue
            actions.append({
                "tool": step.get("action", ""),
                "input": step.get("action_input", {}),
                "observation": _truncate(step.get("observation", ""), self.max_obs_chars),
            })

        turn = SessionTurn(
            question=question,
            answer=answer,
            actions=actions,
        )

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(turn)
            if len(history) > self.max_turns:
                del history[:-self.max_turns]


SESSION_STORE = SessionStore()
