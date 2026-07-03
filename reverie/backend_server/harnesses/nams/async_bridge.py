"""
Sync -> async bridge for the NAMS SDK.

``neo4j-agent-memory`` is async-only by design. The reverie main loop is a
plain synchronous ``while`` loop that processes one persona at a time. We
bridge the two by owning a single asyncio event loop on a dedicated background
thread and exposing ``run(coro)`` which submits a coroutine to that loop from
any (sync) caller thread and blocks on the result.

Why a single persistent loop (rather than ``asyncio.run`` per call):
``MemoryClient`` is designed to be constructed once and reused -- it holds a
Neo4j connection pool. Creating and tearing down an event loop + client per
call would (a) be slow, (b) re-open the bolt connection every time, and
(c) fight the SDK's connection pooling. The persistent-loop design lets the
per-persona ``MemoryClient`` live for the whole simulation while keeping the
reverie loop's sync semantics untouched.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
  """Lazily start the background event loop thread and return the loop."""
  global _loop, _loop_thread
  with _loop_lock:
    if _loop is not None and not _loop.is_closed():
      return _loop

    ready = threading.Event()

    def _runner() -> None:
      global _loop
      _loop = asyncio.new_event_loop()
      asyncio.set_event_loop(_loop)
      ready.set()
      try:
        _loop.run_forever()
      finally:
        try:
          _loop.close()
        except Exception:
          pass

    _loop_thread = threading.Thread(
      name="nams-asyncio-loop", target=_runner, daemon=True
    )
    _loop_thread.start()
    ready.wait(timeout=10)
    assert _loop is not None and not _loop.is_closed(), (
      "failed to start NAMS asyncio loop"
    )
    return _loop


def run(coro: Awaitable[T]) -> T:
  """Submit ``coro`` to the persistent background loop and block on its result.

  Safe to call from any sync thread (including the main reverie thread). Re-
  raises any exception the coroutine raised.
  """
  loop = _ensure_loop()
  if not asyncio.iscoroutine(coro):
    raise TypeError(f"run() expects a coroutine, got {type(coro).__name__}")
  fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
  return fut.result()


def shutdown() -> None:
  """Stop the background loop. Only useful in tests / interpreter teardown."""
  global _loop, _loop_thread
  with _loop_lock:
    if _loop is None or _loop.is_closed():
      return
    _loop.call_soon_threadsafe(_loop.stop)
    if _loop_thread is not None:
      _loop_thread.join(timeout=5)
    _loop = None
    _loop_thread = None
