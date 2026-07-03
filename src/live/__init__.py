"""Phase 8 — Live paper-trading.

Submodules:
  * stream       – Binance WebSocket consumer pushing closed candles
                   into an asyncio queue.
  * paper_trader – Vol-scaled, confidence-gated paper-trading state
                   machine.
  * orchestrator – Async coroutine wiring the stream + inference +
                   trader together, exposed as a singleton consumed by
                   the FastAPI routes.
"""
