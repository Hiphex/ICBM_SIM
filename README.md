# ICBM Intercept Simulation

This project models a notional 2D ICBM flight and layered defense intercept attempt. It integrates the missile boost, ballistic, and re-entry phases alongside interceptor launches, guidance, and decoy discrimination so you can explore how parameter and sensor variations affect the outcome.

- Two-layer defense (GBI-style exo-atmospheric followed by THAAD-like endo-atmospheric).
- Atmospheric drag, multi-stage boost scheduling, constant wind, and optional decoy deployment.
- Monte Carlo sampling across hundreds of randomized engagements with JSON logging.
- Optional Matplotlib plots and animations for quick visual inspection of trajectories.

## Requirements

- Python 3.9 or newer.
- Optional: `matplotlib` (only needed for `--plot` or `--animate`).

No third-party packages are required for the core simulation; everything else ships with the standard library.

## Quick Start

```bash
python3 simulation.py
```

Sample console output:

```
GBI interceptor achieved lock at t= 869.8s over position (2,507,621 m, 2,494,099 m).
Sample count: 3480 | Intercept success: True
```

## CLI Options

- `--runs N` — run N simulations (default `1`). Values greater than 1 enable Monte Carlo mode.
- `--seed SEED` — fix the random seed for reproducible guidance noise, wind, and decoy samples.
- `--plot` — render a static trajectory plot (requires Matplotlib; only available when `--runs 1`).
- `--animate` — play an animation of the engagement (requires Matplotlib; only available when `--runs 1`).
- `--log-json PATH` — append JSONL entries describing each run to `PATH`. Combine with `--append-log` to avoid overwriting.
- `--gbi-salvo N` / `--thaad-salvo N` — launch `N` interceptors in parallel (or staggered) for the upper and lower layers.
- `--gbi-salvo-interval SECONDS` / `--thaad-salvo-interval SECONDS` — spacing between interceptors in the salvo (default `0` for simultaneous launches).

When Monte Carlo mode is active you receive aggregated statistics such as hit probability, decoy intercepts, and miss distances. Specifying `--seed` lets you replay a particular draw set.

## Logging Format

Each JSON line includes:

- `mode`: `single`, `monte_carlo`, or `monte_carlo_summary`.
- `success`: whether the primary warhead was destroyed.
- `interceptor_reports`: per-layer success, intercept time, and target label (primary vs decoy).
- `parameters`: the effective physical and guidance parameters used for that run.
- `min_primary_distance`: closest approach between the primary warhead and the defense stack.

This makes it straightforward to ingest the output into pandas or your preferred analysis tool.

## Working With Parameters

`simulate_icbm_intercept` exposes numerous keyword arguments to tune the scenario:

- Missile boost profile durations and lateral accelerations.
- Pitch schedule per stage.
- Wind vector, drag coefficients, and mass fractions for the warhead and decoys.
- Interceptor launch timing, speed caps, guidance gains, acceleration limits, and seeker noise.
- Decoy deployment timing, quantity, and sensor confusion mechanics.
- Salvo sizing via `salvo_count` and `salvo_interval` on each `InterceptorConfig` to model layered parallel launches.

From Python you can import the module and call `simulate_icbm_intercept` directly:

```python
from simulation import simulate_icbm_intercept

result = simulate_icbm_intercept(
    interceptor_launch_delay=90.0,
    decoy_count=5,
    wind_velocity=(120.0, 15.0),
)
print(result.intercept_success)
```

For Monte Carlo experiments, `run_monte_carlo(runs, base_kwargs=...)` accepts baseline values that will be perturbed per run.

### Layered Engagement Logic

Interceptor coordination now supports dependencies. For example, the default THAAD layer waits for the GBI layer to fail (decoy intercept, timeout, or 180 seconds with no kill) before it launches. You can control this by passing `depends_on="GBI"` and optionally `dependency_grace_period` when building an `InterceptorConfig`. Setting a grace period allows the next layer to launch even if the upstream interceptor is still in flight but has exceeded its allotted engagement window.

### Parallel Salvo Launches

Each interceptor layer can now spawn multiple vehicles and either launch them simultaneously or with a configurable cadence. Example:

```python
from simulation import simulate_icbm_intercept

result = simulate_icbm_intercept(
    gbi_salvo_count=3,
    gbi_salvo_interval=0.5,  # 500 ms spacing
    thaad_salvo_count=2,
)
```

Reports and trajectory samples use names like `GBI#1`, `GBI#2`, etc., so you can trace which vehicle achieved a kill or was spoofed by a decoy.

### Layer Geometry Defaults

The upper-tier GBI still launches from an 800 km downrange site, but the terminal THAAD battery is now staged ~3.2 Mm further along the trajectory with a shorter launch delay, a higher intercept ceiling (≈1 400 km), and a higher speed cap to sprint into the re-entry corridor. The layer also works with a 60 km lethal radius, making the engagement more precise while still giving the second layer a realistic chance to catch leak-through shots head-on.

## Interceptor Agents

See `agents.md` for a breakdown of the default interceptor layers and guidance behaviour. You can extend the `interceptors` argument to supply additional `InterceptorConfig` entries or modify the defaults inside `simulate_icbm_intercept`.

## Feature Ideas

- Add boost-phase interceptors tied to separate radar cue delays and satellite tracking errors.
- Model MIRV releases with per-warhead discrimination tied to seeker fidelity and radar cross sections.
- Introduce command-and-control latency so interceptor handovers depend on communications quality.
- Export HDF5 or parquet logs for bulk statistical analysis and integration with external tools.
- Wrap the simulation in a simple web dashboard or notebook widgets for interactive parameter sweeps.

## Project Layout

- `simulation.py` — core physics, guidance, Monte Carlo harness, and plotting helpers.
- `README.md` — overview and usage (this file).
- `agents.md` — detailed interceptor descriptions and extension notes.

## License

No license specified. Add one before publishing if you intend to share or reuse the code publicly.
