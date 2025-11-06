# Interceptor Agents

The simulation uses layered interceptors, each described by an `InterceptorConfig`. Every interceptor can hold different guidance gains, seeker noise, and engagement envelopes, making it easy to explore mixed-defense architectures.

## Default Layers

### GBI (Ground-Based Interceptor)

- **Site**: `(3.8 Mm, 0 m)` downrange from the launch point (co-located with THAAD in the default geometry).
- **Engagement altitude**: `120 km` to `900 km`.
- **Launch delay**: `120 s` after detection.
- **Speed cap**: `5.4 km/s`.
- **Guidance**: High proportional navigation gain (`0.7`) with light damping (`0.05`).
- **Acceleration limit**: `60 g`.
- **Interception radius**: `≥8 km` to approximate a wide-area midcourse kill vehicle footprint.
- **Sensor noise**: `~0.08°` (reduced relative to the base noise).
- **Confusion probability**: Scaled to discourage decoy lock-ons; seeker can reacquire at least `1%` per second.
- **Salvo**: Defaults to a single interceptor; increase `salvo_count` or `salvo_interval` for parallel launches.

This layer is optimized for exo-atmospheric mid-course engagements where the warhead is still above 120 km and drag is low.

### THAAD (Terminal High Altitude Area Defense analogue)

- **Site**: co-located with the GBI battery (default `3.8 Mm` downrange).
- **Engagement altitude**: sea level to `1.4 Mm`.
- **Launch delay**: `~340 s` after detection (base delay plus 220 seconds).
- **Speed cap**: `4.2 km/s`.
- **Guidance**: Moderately higher gain (`>= 0.55`) with stronger damping (`>= 0.09`) for endo-atmospheric flight.
- **Acceleration limit**: Up to `130 g`.
- **Sensor noise**: At least `0.06°`, reflecting more challenging terminal tracking.
- **Interception radius**: `45 km` lethal radius to approximate a focused terminal kill vehicle.
- **Confusion probability**: Higher susceptibility to decoys (≥0.25 base) to reflect non-ideal real-world performance.
- **Dependency**: Waits on the GBI layer; launches only after the GBI interceptor fails or 60 seconds elapse without a primary kill.
- **Salvo**: Also defaults to a single interceptor; configure `salvo_count` to fan out multiple terminal shots.

This layer catches anything that leaks through the upper tier and re-enters the denser atmosphere.

## Guidance Behaviour

All interceptors use a noisy line-of-sight guidance law limited by their maximum lateral acceleration. When confusion with a decoy occurs, an interceptor may switch to a decoy track until it reacquires the primary target based on its `reacquisition_rate`.

## Extending the Defense Stack

To add or modify layers:

1. Import `InterceptorConfig` and construct your desired list.
2. Pass the list to `simulate_icbm_intercept(interceptors=[...])` or update the default branch inside the function.
3. Tune `launch_delay`, `engage_altitude_*`, `speed_cap`, and guidance parameters to model a new capability.
4. Optionally set `salvo_count` (number of interceptors) and `salvo_interval` (seconds between launches) for each layer.

Example:

```python
from simulation import simulate_icbm_intercept, InterceptorConfig

patriot = InterceptorConfig(
    name="PATRIOT",
    site=(2_700_000.0, 0.0),
    launch_delay=900.0,
    engage_altitude_min=0.0,
    engage_altitude_max=35_000.0,
    speed_cap=1500.0,
    guidance_gain=0.55,
    damping_gain=0.18,
    intercept_distance=150.0,
    max_accel=70.0,
    guidance_noise_std_deg=0.12,
    confusion_probability=0.25,
    reacquisition_rate=0.05,
    max_flight_time=200.0,
    depends_on=None,
    dependency_grace_period=0.0,
    salvo_count=3,
    salvo_interval=0.4,
)

result = simulate_icbm_intercept(interceptors=[patriot])
```

Feel free to mix and match additional agents to explore layered or geographically distributed defense concepts.
When stacking layers, set `depends_on` (and optionally `dependency_grace_period`) on the downstream interceptor so it only launches once the specified upstream layer fails or exhausts its engagement window.
