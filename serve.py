"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    max_steps: int = 10
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str | None = None


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: str | None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run
    from session_memory import SESSION_STORE

    sid = SESSION_STORE.ensure(session_id)
    memory_context = SESSION_STORE.build_context(sid)

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    loop = asyncio.get_running_loop()
    steps: list[dict] = []
    final_answer: str | None = None

    def _worker():
        try:
            for step_data in react_run(
                question,
                max_steps=max_steps,
                memory_context=memory_context,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, step_data)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "error",
                "observation": f"Agent 执行出错: {e}",
            })
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    yield _sse({
        "type": "start",
        "question": question,
        "mode": mode,
        "session_id": sid,
        "has_memory": bool(memory_context),
    })

    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        steps.append(step_data)
        if step_data.get("type") == "final":
            final_answer = step_data.get("answer", "")
        yield _sse(step_data)

    if final_answer:
        SESSION_STORE.append_turn(sid, question, final_answer, steps)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/clear")
async def clear_session(req: SessionRequest):
    from session_memory import SESSION_STORE

    if req.session_id:
        SESSION_STORE.clear(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
