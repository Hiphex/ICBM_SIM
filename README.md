# ICBM Intercept Simulation

This project models a notional 2D ICBM flight and layered defense intercept attempt. It integrates the missile boost, ballistic, and re-entry phases alongside interceptor launches, guidance, and decoy discrimination so you can explore how parameter and sensor variations affect the outcome.

**Key Features:**
- Two-layer defense (GBI-style exo-atmospheric followed by THAAD-like endo-atmospheric).
- Atmospheric drag, multi-stage boost scheduling, constant wind, and optional decoy deployment.
- MIRV (Multiple Independently Targetable Reentry Vehicle) support with configurable warhead release.
- Multi-ICBM scenarios with multiple simultaneous threats.
- Multi-site defense architecture with distributed interceptor batteries.
- Monte Carlo sampling across hundreds of randomized engagements with JSON logging.
- Optional Matplotlib plots and animations for quick visual inspection of trajectories.
- US Standard Atmosphere 1976 density model (configurable).
- Adaptive time stepping for improved boost/reentry precision.

## Requirements

- Python 3.9 or newer.
- Optional: `matplotlib` (only needed for `--plot` or `--animate`).
- Optional: `pytest` (only needed for running tests).

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
- `--append-log` — append to existing JSON log file instead of overwriting.
- `--workers N` / `--threads N` — number of worker processes for Monte Carlo runs (`0` uses all cores).

When Monte Carlo mode is active you receive aggregated statistics such as hit probability, decoy intercepts, and miss distances. Specifying `--seed` lets you replay a particular draw set. Use `--workers` to distribute runs across CPU cores (process-based parallelism).

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
- MIRV release timing, warhead count, and spread velocity.
- Multi-ICBM scenarios with independent missile configurations.
- Multi-site defense with distributed batteries and launchers.
- Salvo sizing via `salvo_count` and `salvo_interval` on each `InterceptorConfig` to model layered parallel launches.
- Adaptive time stepping for high-precision boost and reentry phases.

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

Interceptor coordination now supports dependencies. For example, the default THAAD layer waits for the GBI layer to fail (decoy intercept, timeout, or 45 seconds with no kill) before it launches. You can control this by passing `depends_on="GBI"` and optionally `dependency_grace_period` when building an `InterceptorConfig`. Setting a grace period allows the next layer to launch even if the upstream interceptor is still in flight but has exceeded its allotted engagement window.

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

### MIRV Support

The simulation supports Multiple Independently Targetable Reentry Vehicles (MIRVs). Configure MIRV release timing and warhead count:

```python
from simulation import simulate_icbm_intercept

result = simulate_icbm_intercept(
    mirv_count=5,
    mirv_release_time=300.0,  # seconds after launch
    mirv_spread_velocity=120.0,  # m/s separation velocity
)
```

### Multi-ICBM and Multi-Site Defense

For complex scenarios with multiple simultaneous threats and distributed defense sites:

```python
from simulation import simulate_icbm_intercept, ICBMConfig, DefenseSiteConfig, BatteryConfig, LauncherConfig

# Define multiple ICBMs
icbm_configs = [
    ICBMConfig(name="ICBM-1", start_position=(0.0, 0.0), ...),
    ICBMConfig(name="ICBM-2", start_position=(100000.0, 0.0), ...),
]

# Define defense sites with batteries and launchers
defense_sites = [
    DefenseSiteConfig(
        name="Site-Alpha",
        position=(3800000.0, 0.0),
        batteries=[
            BatteryConfig(
                name="GBI-Battery-1",
                launchers=[LauncherConfig(interceptor_count=4), ...],
            ),
        ],
    ),
]

result = simulate_icbm_intercept(
    icbm_configs=icbm_configs,
    defense_sites=defense_sites,
)
```

### Radar and Discrimination

The simulation includes a physics-based radar model and ballistic coefficient discrimination for target selection.

- **Radar Horizon**: Targets are only visible if they are above the radar's line-of-sight horizon, accounting for Earth's curvature.
- **Detection Probability**: Probability of detection depends on target range and Radar Cross Section (RCS), following the radar range equation ($P \propto 1/R^4$).
- **Discrimination**: When targets are in the atmosphere, the radar estimates their ballistic coefficients (B). Since heavy warheads and light decoys decelerate differently, the system can calculate the probability that each tracked object is the actual warhead.
- **Target Selection**: Interceptors use radar discrimination data to prioritize likely warheads and deconflict with other interceptors to avoid wasting shots on identified decoys.

**New CLI Options:**
- `--no-discrimination` — disable physics-based discrimination and fall back to a simple probabilistic model.
- `--radar-range RANGE` — set the maximum radar detection range in meters (default `4000000`).
- `--radar-pos X ALT` — set the radar's horizontal position and altitude.
- `--radar-update-rate HZ` — set the tracking update frequency (default `10.0`).

### Layer Geometry Defaults

Both layers now stage from the same downrange site (~3.8 Mm) near the expected re-entry corridor. The GBI salvo leaves almost immediately, while THAAD launches only after a GBI miss and sprints into the terminal window with explicit range/altitude gates (GBI ≈0.4–6.0 Mm, THAAD ≈0–0.26 Mm). The calibration leans on public scorecards: the Missile Defense Agency’s **Ballistic Missile Defense Intercept Flight Test Record** (Jan 2019) lists 10 GMD hits in 18 attempts through 2018 (12/20 after the 2017/2019 shots), while CSIS’s **THAAD Achieves 16th Successful Intercept** brief (Aug 2019) documents THAAD’s 16/16 scripted intercepts. Monte Carlo batches (500 runs, seed 20251106) currently land at ≈53 % primary kills from GBI and ≈18 % from THAAD (≈40 % conditional success once GBI fails), highlighting the gap between developmental tests and our more pessimistic field assumptions.

## Interceptor Agents

See `agents.md` for a breakdown of the default interceptor layers and guidance behaviour. You can extend the `interceptors` argument to supply additional `InterceptorConfig` entries or modify the defaults inside `simulate_icbm_intercept`.

## Testing

The project includes comprehensive tests covering core simulation logic, Monte Carlo functionality, and physics models. Run tests with:

```bash
pytest tests/
```

Or run specific test files:

```bash
pytest test_simulation_core.py
pytest tests/test_simulation.py
pytest tests/test_new_features.py
```

Tests verify:
- Ballistic trajectory calculations
- Interceptor guidance and engagement logic
- Decoy deployment and discrimination
- MIRV spawning and tracking
- Standard atmosphere density model
- Gravity falloff with altitude
- Monte Carlo parameter sampling and JSON serialization

## Feature Ideas

- Add boost-phase interceptors tied to separate radar cue delays and satellite tracking errors.
- Enhanced MIRV discrimination with per-warhead radar cross sections and seeker fidelity variations.
- Introduce command-and-control latency so interceptor handovers depend on communications quality.
- Export HDF5 or parquet logs for bulk statistical analysis and integration with external tools.
- Wrap the simulation in a simple web dashboard or notebook widgets for interactive parameter sweeps.
- 3D trajectory visualization (currently 2D range/altitude).

## Project Layout

- `simulation.py` — core physics, guidance, Monte Carlo harness, and plotting helpers.
- `README.md` — overview and usage (this file).
- `agents.md` — detailed interceptor descriptions and extension notes.
- `test_simulation_core.py` — core simulation tests (guard conditions, ballistic impact).
- `tests/test_simulation.py` — comprehensive tests for Monte Carlo, multi-ICBM, and defense sites.
- `tests/test_new_features.py` — tests for MIRV, atmosphere, and gravity models.
- `test_log.jsonl` — example JSON log output (may be overwritten during testing).

## License

No license specified. Add one before publishing if you intend to share or reuse the code publicly.
