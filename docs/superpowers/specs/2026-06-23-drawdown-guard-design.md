# DrawdownGuard — Design Spec
**Date:** 2026-06-23  
**Scope:** `execution/live_trader.py` only  
**Status:** Approved, pending implementation

---

## Problem

The live trader has no hard floor on session losses. `RegimeGuard` scales position size down during high-ATR regimes but does not halt trading or close positions. A sustained adverse move can accumulate unbounded losses within a session.

---

## Solution

Add a `DrawdownGuard` class that tracks session equity, detects when peak-to-trough drawdown exceeds 3%, closes the position, and suspends trading for 2 hours before automatically resuming.

---

## DrawdownGuard Class

**Location:** `execution/live_trader.py`, immediately after the `RegimeGuard` class definition.

**Constructor:**
```python
DrawdownGuard(max_drawdown: float = 0.03, cooldown_hours: float = 2.0)
```

**Internal state:**
- `_peak_equity: float` — session high-water mark; initialized to `capital` on the first `update()` call, updated whenever equity exceeds the current peak
- `_halted_until: datetime | None` — `None` when active; set to `utcnow + cooldown_hours` when drawdown threshold is breached

**Public interface:**

| Method | Signature | Behaviour |
|--------|-----------|-----------|
| `update` | `(equity: float) -> bool` | Updates `_peak_equity`; computes drawdown `= (peak - equity) / peak`; if drawdown `> max_drawdown` and not already halted, sets `_halted_until` and returns `True` (halt transition). Returns `False` otherwise. |
| `is_halted` | `() -> bool` | Returns `True` if `_halted_until is not None` and `datetime.now(UTC) < _halted_until`. Automatically returns `False` after the cooldown expires — no explicit resume call required. |

**Drawdown formula:**
```
drawdown = (peak_equity - current_equity) / peak_equity
halt if drawdown > max_drawdown (0.03)
```

---

## Integration into LiveTrader

**Instantiation** in `LiveTrader.__init__`:
```python
self._drawdown_guard = DrawdownGuard(max_drawdown=0.03, cooldown_hours=2.0)
```

**Per-candle logic** in `_process_candle` (or equivalent handler), inserted after `_unrealised_pnl` is updated and before inference:

```
equity = self.capital + self._unrealised_pnl

halt_triggered = self._drawdown_guard.update(equity)
if halt_triggered:
    await self._close_position()
    logger.warning("DRAWDOWN HALT | drawdown=%.2f%% | resuming at %s", ...)
    return  # skip inference this bar

if self._drawdown_guard.is_halted():
    logger.info("Halted — bars accumulating, inference suspended")
    return  # skip inference, keep filling warm-up buffer

# normal flow continues: RegimeGuard → model.predict() → place order
```

---

## Behaviour During Halt

| Condition | Behaviour |
|-----------|-----------|
| Halt triggers | `_close_position()` called immediately; inference skipped |
| Already flat when halt triggers | `_close_position()` logs "already flat" and returns cleanly |
| During 2-hour cooldown | Bars continue accumulating; rolling norm window stays current; inference skipped |
| After 2 hours | `is_halted()` returns `False`; normal inference resumes on next candle |
| `whatif_only=True` | Halt triggers and logs; `_close_position()` skips real orders (existing behaviour) |

---

## What Is Not Changing

- `RegimeGuard` is unchanged — ATR-based position scaling continues to operate independently
- Data pipeline, training, environment, and evaluation code are untouched
- High-water mark is session-scoped: resets when the process restarts. Persistent cross-session tracking is a future enhancement.

---

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_drawdown` | 0.03 (3%) | User-specified; ~$30K on current $1M capital |
| `cooldown_hours` | 2.0 | User-specified; allows regime to stabilise before re-entry |

---

## Files Touched

| File | Change |
|------|--------|
| `execution/live_trader.py` | Add `DrawdownGuard` class; instantiate in `LiveTrader.__init__`; insert halt check in per-candle handler |

No other files are modified.
