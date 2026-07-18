"""
Q-MagNav Demo
AI-assisted magnetic anomaly navigation under GNSS denial/spoofing.

Run:
    pip install -r requirements.txt
    streamlit run app.py

This is a synthetic but technically meaningful demo:
- A vehicle follows a 2D trajectory.
- GNSS is initially reliable, then becomes spoofed.
- Dead reckoning drifts over time.
- A magnetic anomaly map is used for map-aided localization.
- A particle filter fuses odometry, magnetometer data, and GNSS only while trusted.
- A simple OODA-style decision layer detects GNSS inconsistency and switches trust mode.

Author: Milad Ghadimi
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class DemoConfig:
    seed: int = 7
    n_steps: int = 220
    n_particles: int = 900
    map_size_m: float = 10_000.0
    grid_n: int = 140

    gps_noise_m: float = 18.0
    gps_spoof_start_pct: float = 0.48
    gps_spoof_growth_m_per_step: float = 11.0
    gps_spoof_side_offset_m: float = 350.0

    odo_noise_m: float = 5.0
    odo_bias_m_per_step: float = 0.9
    process_noise_m: float = 13.0

    mag_noise_nt: float = 2.0
    classical_mag_noise_nt: float = 8.0
    quantum_mag_noise_nt: float = 2.0
    magnetic_bias_drift_nt_per_step: float = 0.015

    gps_residual_threshold_m: float = 170.0
    spoof_confirm_steps: int = 4


# -----------------------------
# Magnetic map and interpolation
# -----------------------------

def make_magnetic_map(cfg: DemoConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic magnetic anomaly map B(x,y) in nanoTesla.

    The map is a smooth background plus localized positive/negative anomalies.
    This represents a simplified 2D magnetic anomaly field.
    """
    rng = np.random.default_rng(cfg.seed)
    xs = np.linspace(0.0, cfg.map_size_m, cfg.grid_n)
    ys = np.linspace(0.0, cfg.map_size_m, cfg.grid_n)
    X, Y = np.meshgrid(xs, ys)

    # Base field around Earth magnetic-field magnitude, with a weak regional trend.
    B = 50_000.0 + 0.0015 * X - 0.0010 * Y

    # Fixed anomalies for reproducibility.
    anomalies = [
        (1800, 2500, +90, 900, 650),
        (3100, 7200, -75, 800, 950),
        (5400, 4200, +110, 700, 700),
        (7600, 6500, -95, 1000, 750),
        (8300, 2300, +65, 650, 850),
        (4700, 8200, +55, 950, 550),
        (6700, 3500, -45, 600, 600),
    ]

    for cx, cy, amp, sx, sy in anomalies:
        B += amp * np.exp(-(((X - cx) ** 2) / (2 * sx ** 2) + ((Y - cy) ** 2) / (2 * sy ** 2)))

    # Add a small smooth-ish texture.
    B += 7.0 * np.sin(X / 1100.0) * np.cos(Y / 1500.0)
    B += 3.0 * np.sin((X + Y) / 900.0)

    return xs, ys, B


def bilinear_interpolate(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation for points shaped (N,2), with columns x,y.
    Returns Z(x,y).
    """
    x = np.clip(points[:, 0], xs[0], xs[-1])
    y = np.clip(points[:, 1], ys[0], ys[-1])

    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    wx = (x - x0) / np.maximum(x1 - x0, 1e-9)
    wy = (y - y0) / np.maximum(y1 - y0, 1e-9)

    return (
        (1 - wx) * (1 - wy) * z00
        + wx * (1 - wy) * z10
        + (1 - wx) * wy * z01
        + wx * wy * z11
    )


# -----------------------------
# Simulation
# -----------------------------

def make_trajectory(cfg: DemoConfig) -> np.ndarray:
    """
    Create a smooth 2D vehicle trajectory across the magnetic map.
    """
    t = np.linspace(0.0, 1.0, cfg.n_steps)

    x = 900 + 8200 * t
    y = 5200 + 1650 * np.sin(2.3 * np.pi * t + 0.3) + 450 * np.sin(6.0 * np.pi * t)

    # Keep inside map bounds.
    x = np.clip(x, 250, cfg.map_size_m - 250)
    y = np.clip(y, 250, cfg.map_size_m - 250)

    return np.column_stack([x, y])


def simulate_sensors(
    cfg: DemoConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    Bmap: np.ndarray,
    mag_noise_nt: float,
) -> Dict[str, np.ndarray]:
    """
    Simulate ground truth, GNSS, dead reckoning, odometry, and magnetic measurement.
    """
    rng = np.random.default_rng(cfg.seed + 100)
    truth = make_trajectory(cfg)
    n = cfg.n_steps

    true_B = bilinear_interpolate(xs, ys, Bmap, truth)

    drift = np.cumsum(rng.normal(0.0, cfg.magnetic_bias_drift_nt_per_step, size=n))
    mag_meas = true_B + drift + rng.normal(0.0, mag_noise_nt, size=n)

    # GPS / GNSS: reliable first, then spoofed with a growing false offset.
    gps = truth + rng.normal(0.0, cfg.gps_noise_m, size=(n, 2))
    spoof_start = int(cfg.gps_spoof_start_pct * n)

    for k in range(spoof_start, n):
        g = k - spoof_start
        gps[k, 0] += cfg.gps_spoof_growth_m_per_step * g
        gps[k, 1] += cfg.gps_spoof_side_offset_m * np.sin(g / 19.0)

    # Odometry increments: approximate local motion with noise and small bias.
    true_delta = np.diff(truth, axis=0, prepend=truth[:1])
    heading_bias = np.deg2rad(4.0)
    R = np.array(
        [
            [np.cos(heading_bias), -np.sin(heading_bias)],
            [np.sin(heading_bias), np.cos(heading_bias)],
        ]
    )
    odo_delta = true_delta @ R.T
    odo_delta += rng.normal(0.0, cfg.odo_noise_m, size=(n, 2))
    odo_delta += np.array([cfg.odo_bias_m_per_step, -0.35 * cfg.odo_bias_m_per_step])

    # Dead reckoning starts at the first GPS point, then integrates odometry.
    dead = np.zeros_like(truth)
    dead[0] = gps[0]
    for k in range(1, n):
        dead[k] = dead[k - 1] + odo_delta[k]

    return {
        "truth": truth,
        "gps": gps,
        "dead": dead,
        "odo_delta": odo_delta,
        "true_B": true_B,
        "mag_meas": mag_meas,
        "spoof_start": np.array([spoof_start]),
    }


# -----------------------------
# Particle filter
# -----------------------------

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    indexes = np.zeros(n, dtype=np.int64)

    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def run_particle_filter(
    cfg: DemoConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    Bmap: np.ndarray,
    data: Dict[str, np.ndarray],
    mag_sigma: float,
) -> Dict[str, np.ndarray]:
    """
    Particle-filter localization.

    State: 2D position.
    Prediction: odometry increments + process noise.
    Measurement: magnetic anomaly likelihood.
    GNSS is used only while it is consistent with the fused estimate.
    """
    rng = np.random.default_rng(cfg.seed + 500)

    n = cfg.n_steps
    N = cfg.n_particles
    map_size = cfg.map_size_m

    gps = data["gps"]
    odo = data["odo_delta"]
    mag = data["mag_meas"]

    particles = gps[0] + rng.normal(0.0, 70.0, size=(N, 2))
    weights = np.ones(N) / N

    estimates = np.zeros((n, 2))
    uncertainty = np.zeros(n)
    gps_residual = np.zeros(n)
    gps_trusted = np.ones(n, dtype=bool)
    ooda_state = np.empty(n, dtype=object)

    suspicious_counter = 0
    trust_gps = True

    for k in range(n):
        if k > 0:
            particles += odo[k] + rng.normal(0.0, cfg.process_noise_m, size=(N, 2))
            particles[:, 0] = np.clip(particles[:, 0], 0.0, map_size)
            particles[:, 1] = np.clip(particles[:, 1], 0.0, map_size)

        # Magnetic likelihood
        predicted_B = bilinear_interpolate(xs, ys, Bmap, particles)
        mag_err = predicted_B - mag[k]
        mag_likelihood = np.exp(-0.5 * (mag_err / max(mag_sigma, 1e-6)) ** 2)

        weights *= mag_likelihood + 1e-300
        weights_sum = np.sum(weights)
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            weights = np.ones(N) / N
        else:
            weights /= weights_sum

        # Preliminary estimate before deciding GPS trust at this step.
        preliminary = np.average(particles, weights=weights, axis=0)
        gps_residual[k] = float(np.linalg.norm(gps[k] - preliminary))

        # Simple GNSS spoofing detector:
        # if GNSS disagrees strongly with the magnetic/odometry fusion for multiple steps,
        # stop using GNSS as a trusted measurement.
        if gps_residual[k] > cfg.gps_residual_threshold_m:
            suspicious_counter += 1
        else:
            suspicious_counter = max(0, suspicious_counter - 1)

        if suspicious_counter >= cfg.spoof_confirm_steps:
            trust_gps = False

        gps_trusted[k] = trust_gps

        # If still trusted, apply GPS likelihood too.
        if trust_gps:
            gps_err = np.linalg.norm(particles - gps[k], axis=1)
            gps_likelihood = np.exp(-0.5 * (gps_err / max(cfg.gps_noise_m, 1e-6)) ** 2)
            weights *= gps_likelihood + 1e-300
            weights /= np.sum(weights)

        # Estimate and uncertainty.
        estimates[k] = np.average(particles, weights=weights, axis=0)
        diffs = particles - estimates[k]
        uncertainty[k] = float(np.sqrt(np.average(np.sum(diffs ** 2, axis=1), weights=weights)))

        # Resample when effective sample size is low.
        ess = 1.0 / np.sum(weights ** 2)
        if ess < 0.55 * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.ones(N) / N

        if trust_gps:
            ooda_state[k] = "GNSS trusted: fuse GPS + odometry + magnetic map"
        else:
            ooda_state[k] = "GNSS denied/spoofed: switch to magnetic-map-aided navigation"

    return {
        "estimate": estimates,
        "uncertainty": uncertainty,
        "gps_residual": gps_residual,
        "gps_trusted": gps_trusted,
        "ooda_state": ooda_state,
    }


def compute_metrics(truth: np.ndarray, gps: np.ndarray, dead: np.ndarray, magnav: np.ndarray, spoof_start: int) -> pd.DataFrame:
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    rows = [
        {
            "method": "GNSS / GPS",
            "RMSE all [m]": rmse(gps, truth),
            "RMSE after spoof [m]": rmse(gps[spoof_start:], truth[spoof_start:]),
            "final error [m]": float(np.linalg.norm(gps[-1] - truth[-1])),
        },
        {
            "method": "Dead reckoning / INS",
            "RMSE all [m]": rmse(dead, truth),
            "RMSE after spoof [m]": rmse(dead[spoof_start:], truth[spoof_start:]),
            "final error [m]": float(np.linalg.norm(dead[-1] - truth[-1])),
        },
        {
            "method": "Q-MagNav particle filter",
            "RMSE all [m]": rmse(magnav, truth),
            "RMSE after spoof [m]": rmse(magnav[spoof_start:], truth[spoof_start:]),
            "final error [m]": float(np.linalg.norm(magnav[-1] - truth[-1])),
        },
    ]
    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Plotting
# -----------------------------

def make_map_figure(
    xs: np.ndarray,
    ys: np.ndarray,
    Bmap: np.ndarray,
    data: Dict[str, np.ndarray],
    pf: Dict[str, np.ndarray],
    k_view: int,
) -> go.Figure:
    truth = data["truth"]
    gps = data["gps"]
    dead = data["dead"]
    est = pf["estimate"]
    unc = pf["uncertainty"]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=Bmap,
            colorscale="Viridis",
            colorbar=dict(title="nT"),
            name="Magnetic anomaly map",
            opacity=0.88,
        )
    )

    def add_path(arr, name, dash=None, width=3):
        fig.add_trace(
            go.Scatter(
                x=arr[: k_view + 1, 0],
                y=arr[: k_view + 1, 1],
                mode="lines",
                name=name,
                line=dict(width=width, dash=dash),
            )
        )

    add_path(truth, "Ground truth", width=4)
    add_path(gps, "GNSS / GPS", dash="dot", width=2)
    add_path(dead, "Dead reckoning / INS", dash="dash", width=2)
    add_path(est, "Q-MagNav estimate", width=4)

    # Current markers
    markers = [
        (truth, "Truth now"),
        (gps, "GPS now"),
        (dead, "INS now"),
        (est, "Q-MagNav now"),
    ]
    for arr, name in markers:
        fig.add_trace(
            go.Scatter(
                x=[arr[k_view, 0]],
                y=[arr[k_view, 1]],
                mode="markers",
                name=name,
                marker=dict(size=11),
            )
        )

    # Uncertainty circle around current estimate.
    theta = np.linspace(0, 2 * np.pi, 80)
    radius = max(unc[k_view], 25)
    circle_x = est[k_view, 0] + radius * np.cos(theta)
    circle_y = est[k_view, 1] + radius * np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="Q-MagNav uncertainty",
            line=dict(width=2, dash="dash"),
        )
    )

    fig.update_layout(
        height=690,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Q-MagNav demo: GNSS spoofing vs magnetic-map-aided localization",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def make_timeseries_figure(data: Dict[str, np.ndarray], pf: Dict[str, np.ndarray], spoof_start: int, k_view: int) -> go.Figure:
    t = np.arange(len(data["truth"]))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t[: k_view + 1],
            y=data["mag_meas"][: k_view + 1],
            mode="lines",
            name="Measured magnetic field [nT]",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t[: k_view + 1],
            y=data["true_B"][: k_view + 1],
            mode="lines",
            name="True map field at real position [nT]",
            line=dict(dash="dash"),
        )
    )

    fig.add_vline(x=spoof_start, line_dash="dot", annotation_text="GNSS spoof starts")
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Magnetometer stream",
        xaxis_title="time step",
        yaxis_title="magnetic field [nT]",
        legend=dict(orientation="h"),
    )
    return fig


def make_error_figure(data: Dict[str, np.ndarray], pf: Dict[str, np.ndarray], spoof_start: int, k_view: int) -> go.Figure:
    truth = data["truth"]
    gps = data["gps"]
    dead = data["dead"]
    est = pf["estimate"]
    t = np.arange(len(truth))

    gps_error = np.linalg.norm(gps - truth, axis=1)
    dead_error = np.linalg.norm(dead - truth, axis=1)
    mag_error = np.linalg.norm(est - truth, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[: k_view + 1], y=gps_error[: k_view + 1], mode="lines", name="GPS error [m]"))
    fig.add_trace(go.Scatter(x=t[: k_view + 1], y=dead_error[: k_view + 1], mode="lines", name="Dead reckoning error [m]"))
    fig.add_trace(go.Scatter(x=t[: k_view + 1], y=mag_error[: k_view + 1], mode="lines", name="Q-MagNav error [m]"))
    fig.add_trace(go.Scatter(x=t[: k_view + 1], y=pf["uncertainty"][: k_view + 1], mode="lines", name="Q-MagNav uncertainty [m]", line=dict(dash="dash")))
    fig.add_vline(x=spoof_start, line_dash="dot", annotation_text="GNSS spoof starts")
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Position error and uncertainty",
        xaxis_title="time step",
        yaxis_title="meters",
        legend=dict(orientation="h"),
    )
    return fig


def make_residual_figure(pf: Dict[str, np.ndarray], cfg: DemoConfig, spoof_start: int, k_view: int) -> go.Figure:
    t = np.arange(len(pf["gps_residual"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t[: k_view + 1],
            y=pf["gps_residual"][: k_view + 1],
            mode="lines",
            name="GNSS residual vs fused estimate [m]",
        )
    )
    fig.add_hline(y=cfg.gps_residual_threshold_m, line_dash="dash", annotation_text="spoof threshold")
    fig.add_vline(x=spoof_start, line_dash="dot", annotation_text="actual spoof starts")
    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=35, b=10),
        title="GNSS trust monitor",
        xaxis_title="time step",
        yaxis_title="residual [m]",
        legend=dict(orientation="h"),
    )
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Q-MagNav Demo",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Q-MagNav: AI-Assisted Magnetic Anomaly Navigation")
st.caption(
    "A prototype demo for GPS-denied / GPS-spoofed autonomous navigation using magnetic anomaly maps, "
    "sensor fusion, particle filtering, and an OODA-style decision layer."
)

with st.sidebar:
    st.header("Demo controls")

    sensor_mode = st.radio(
        "Magnetometer mode",
        ["Quantum-grade magnetometer", "Classical magnetometer"],
        index=0,
        help="This changes simulated magnetometer noise. It does not claim access to a real quantum sensor.",
    )

    n_particles = st.slider("Particle count", 300, 2500, 900, step=100)
    spoof_pct = st.slider("GNSS spoof start [% of mission]", 20, 80, 48, step=2)
    gps_threshold = st.slider("GNSS residual threshold [m]", 80, 400, 170, step=10)
    process_noise = st.slider("Process noise [m]", 3, 45, 13, step=1)

    st.divider()
    st.markdown("**Mission playback**")
    playback = st.slider("Time step", 1, 219, 219, step=1)

    st.divider()
    st.info(
        "Pitch line: The vehicle detects GNSS inconsistency and switches from GPS trust to "
        "magnetic-map-aided navigation."
    )

cfg = DemoConfig(
    n_particles=int(n_particles),
    gps_spoof_start_pct=float(spoof_pct) / 100.0,
    gps_residual_threshold_m=float(gps_threshold),
    process_noise_m=float(process_noise),
)

mag_noise = cfg.quantum_mag_noise_nt if sensor_mode == "Quantum-grade magnetometer" else cfg.classical_mag_noise_nt

@st.cache_data(show_spinner=False)
def cached_run(cfg_dict, mag_noise_cached):
    local_cfg = DemoConfig(**cfg_dict)
    xs_local, ys_local, Bmap_local = make_magnetic_map(local_cfg)
    data_local = simulate_sensors(local_cfg, xs_local, ys_local, Bmap_local, mag_noise_cached)
    pf_local = run_particle_filter(local_cfg, xs_local, ys_local, Bmap_local, data_local, mag_sigma=max(mag_noise_cached, 1.5))
    return xs_local, ys_local, Bmap_local, data_local, pf_local

cfg_dict = cfg.__dict__.copy()
xs, ys, Bmap, data, pf = cached_run(cfg_dict, mag_noise)
spoof_start = int(data["spoof_start"][0])
k_view = min(int(playback), cfg.n_steps - 1)

truth = data["truth"]
gps = data["gps"]
dead = data["dead"]
est = pf["estimate"]

current_gps_error = float(np.linalg.norm(gps[k_view] - truth[k_view]))
current_dead_error = float(np.linalg.norm(dead[k_view] - truth[k_view]))
current_mag_error = float(np.linalg.norm(est[k_view] - truth[k_view]))

# Top status cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("GNSS trust state", "TRUSTED" if pf["gps_trusted"][k_view] else "DENIED / SPOOFED")
col2.metric("GPS error now", f"{current_gps_error:,.1f} m")
col3.metric("Dead reckoning error now", f"{current_dead_error:,.1f} m")
col4.metric("Q-MagNav error now", f"{current_mag_error:,.1f} m")

# OODA layer
st.subheader("OODA-style decision layer")
ooda_cols = st.columns(4)
ooda_cols[0].markdown("**Observe**  \nMagnetometer + odometry + GNSS")
ooda_cols[1].markdown("**Orient**  \nMap matching + particle filter")
ooda_cols[2].markdown(
    f"**Decide**  \nGNSS residual = `{pf['gps_residual'][k_view]:.1f} m`"
)
ooda_cols[3].markdown(f"**Act**  \n{pf['ooda_state'][k_view]}")

st.plotly_chart(make_map_figure(xs, ys, Bmap, data, pf, k_view), use_container_width=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(make_timeseries_figure(data, pf, spoof_start, k_view), use_container_width=True)
with right:
    st.plotly_chart(make_residual_figure(pf, cfg, spoof_start, k_view), use_container_width=True)

st.plotly_chart(make_error_figure(data, pf, spoof_start, k_view), use_container_width=True)

metrics = compute_metrics(truth, gps, dead, est, spoof_start)
st.subheader("Performance summary")
st.dataframe(metrics.style.format({"RMSE all [m]": "{:.1f}", "RMSE after spoof [m]": "{:.1f}", "final error [m]": "{:.1f}"}), use_container_width=True)

with st.expander("Technical notes"):
    st.markdown(
        """
        **What this demo is:**  
        A synthetic but realistic prototype showing the algorithmic concept of magnetic-map-aided navigation under GNSS denial.

        **What this demo is not:**  
        It is not claiming access to a real quantum magnetometer or proprietary QOODA data.

        **Core idea:**  
        A particle filter propagates candidate positions using noisy odometry. Each candidate is scored by comparing the measured magnetic field to a 2D magnetic anomaly map. GNSS is fused only while it is consistent with the fused estimate. Once the GNSS residual stays above a threshold, the system marks GNSS as spoofed/denied and switches to magnetic-map-aided navigation.

        **Why this is relevant:**  
        It demonstrates robust localization under uncertainty, GPS spoofing detection, sensor fusion, and an OODA-style decision loop.
        """
    )

st.success("Demo ready. Record a 45–60 second screen video and send it with the GitHub link.")
