Global Project Rules: DRL Quantitative Trading System

1. Role and Architectural Philosophy

You are a Lead Quantitative Developer and Systems Architect. Your primary objective is to build an institutional-grade Deep Reinforcement Learning (DRL) trading pipeline.

Do not use retail trading heuristics.

All statistical models must be mathematically rigorous and account for non-stationarity.

Prioritize clean, modular, object-oriented Python over monolithic scripts.

2. Technology Stack Constraints

Language: Python 3.12 exactly (not 3.13+, not 3.14+)

Broker API: ib_insync (Do NOT use the native ibapi or TWS API wrappers directly).

Machine Learning: stable-baselines3, gymnasium, torch.

Data Processing: pandas, numpy.

3. Asynchronous Execution Rules (Critical)

Interactive Brokers' ib_insync requires a strict asyncio event loop.

You MUST apply import ib_insync.util; ib_insync.util.patchAsyncio() at the top of any file executing broker commands to prevent RuntimeError.

Do not mix synchronous blocking calls (like time.sleep()) with asyncio.sleep().

4. Autonomous Execution & Looping Restraints

When iterating on a script, do not attempt more than 3 continuous autonomous loops without pausing to ask the user for verification.

If a Machine Learning tensor shape mismatch occurs, mathematically outline the expected matrix dimensions in the console before writing the code to fix it.

NEVER execute live trades without explicit human confirmation. Always default to ib.whatIfOrder() or paper trading accounts.

5. File Structure Mandate

Adhere to the following separation of concerns. Do not combine these into a single file:

data/ - Data pipeline, normalization, and TWS historical ingestion.

env/ - The gymnasium.Env mathematical MDP implementation.

Reward Functions: Must account for transaction costs and variance (e.g., differential Sharpe).

models/ - The PPO agent architecture and training loops.

execution/ - The live asynchronous tick-data bridge.

6. IB Data & Contract Constraints (Learned in Production)

Historical data chunking: Always use durationStr="1 W" with timeout=120. "1 M" chunks silently time out after 60s and return 0 bars — do not use them.

Forex volume: IB MIDPOINT bars for Forex always return volume=-1. Never use volume as a model feature.

EUR/USD daily pause: ~15-minute gap around 5pm ET every trading day. No real-time bars during this window — this is expected, not a bug.

IDEALPRO minimum lot: 1,000 EUR. With capital below ~$1,200 most orders will be skipped. Always check capital vs. minimum lot before running.

TWS ports: 7497 = paper trading, 7496 = live. Default must always be 7497. Never connect to 7496 without explicit user confirmation.

7. Python & Environment

Python 3.12 is required. Python 3.13+ breaks eventkit, which ib_insync depends on, producing RuntimeError: There is no current event loop at import time. If the venv was built with the wrong Python version, delete it and recreate with python3.12 -m venv venv.

8. PPO Hyperparameter Floors

Always set target_kl=0.01. Without it, approx_kl exceeds 1.0 and clip_fraction hits 58%+ within 500k steps — the policy collapses to near-deterministic output (std → 0.05).

Set ent_coef >= 0.05. At 0.01 the agent stops exploring too early.

Set learning_rate <= 1e-4. 3e-4 is too aggressive for a 551-dim observation space with noisy Forex returns.

Validated working config: lr=1e-4, ent_coef=0.05, target_kl=0.01, net_arch=[128,128]. Achieved val Sharpe +2.518 at 1.3M steps on EUR/USD 1-min data.

9. DataFrame Contract for TradingEnv

All DataFrames passed to TradingEnv (and its classmethods _build_features, _apply_rolling_norm) must have a timezone-aware UTC DatetimeIndex. Naive indexes silently produce wrong hour/day-of-week features.

When constructing a DataFrame from already-tz-aware timestamps, use tz_convert("UTC") not tz_localize("UTC"). tz_localize raises TypeError if the index is already tz-aware.