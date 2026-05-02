"""Streamlit dashboard for the stochastic simulator."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import streamlit as st

from stochastic_simulator import (
    SimulationResult,
    StochasticProcessSimulator,
    benchmark,
)


def _parse_matrix(text: str) -> Optional[np.ndarray]:
    if not text.strip():
        return None
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split(",")])
    matrix = np.array(rows, dtype=np.float64)
    return matrix


def _build_simulator(
    template_name: str,
    transition_text: str,
    rate_text: str,
    use_gpu: bool,
) -> StochasticProcessSimulator:
    templates = StochasticProcessSimulator.templates()
    transition = _parse_matrix(transition_text)
    rates = _parse_matrix(rate_text)

    if transition is not None:
        return StochasticProcessSimulator(
            transition_matrix=transition,
            labels=tuple(f"State {i}" for i in range(transition.shape[0])),
            use_gpu=use_gpu,
        )
    if rates is not None:
        return StochasticProcessSimulator(
            rate_matrix=rates,
            labels=tuple(f"State {i}" for i in range(rates.shape[0])),
            use_gpu=use_gpu,
        )

    template = templates[template_name]
    return StochasticProcessSimulator(
        rate_matrix=template["rate_matrix"],
        labels=template["labels"],
        use_gpu=use_gpu,
    )


def _plot_result(result: SimulationResult, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, label in enumerate(result.labels):
        ax.plot(result.times, result.populations[:, idx], lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Stochastic Process Simulator", layout="wide")
    st.title("Stochastic Process Simulator")
    st.write(
        "Vectorized Monte Carlo engine with optional GPU acceleration, "
        "custom transition matrices, and sensitivity analysis."
    )

    with st.sidebar:
        st.header("Simulation Controls")
        templates = StochasticProcessSimulator.templates()
        template_name = st.selectbox("Template", list(templates.keys()))
        n_particles = st.slider("Particles (N)", 10_000, 1_000_000, 200_000, step=10_000)
        n_steps = st.slider("Total steps", 50, 800, 200, step=10)
        dt = st.number_input("dt", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
        use_gpu = st.checkbox("Use GPU (CuPy)", value=False)

        st.subheader("Optional Custom Input")
        transition_text = st.text_area(
            "Transition Matrix (rows sum to 1, CSV rows)",
            value="",
            placeholder="0.9,0.1\n0.0,1.0",
        )
        rate_text = st.text_area(
            "Rate Matrix (CSV rows, off-diagonal rates)",
            value="",
            placeholder="0.0,0.4\n0.0,0.0",
        )

        st.subheader("Sensitivity Analysis")
        run_sensitivity = st.checkbox("Run sensitivity analysis", value=False)
        n_runs = st.slider("Runs", 5, 30, 10)
        perturb = st.slider("Rate perturbation (%)", 1, 20, 5)

    simulator = _build_simulator(template_name, transition_text, rate_text, use_gpu)
    result = simulator.simulate(n_particles, n_steps, dt)

    col1, col2 = st.columns([2, 1])
    with col1:
        _plot_result(result, f"Population Trends ({result.backend})")
    with col2:
        st.subheader("Summary")
        st.write(f"States: {', '.join(result.labels)}")
        st.write(f"Backend: {result.backend}")
        st.write(f"Final populations: {result.populations[-1].tolist()}")

        csv_data = result.to_csv()
        st.download_button(
            "Export to .CSV",
            data=csv_data,
            file_name="simulation_data.csv",
            mime="text/csv",
        )

    if run_sensitivity:
        st.subheader("Sensitivity Analysis")
        sensitivity = simulator.sensitivity_analysis(
            n_particles,
            n_steps,
            dt,
            n_runs=n_runs,
            perturb_fraction=perturb / 100.0,
        )
        mean = sensitivity["mean"]
        std = sensitivity["std"]

        st.write("Final mean and std (per state):")
        final_mean = mean[-1]
        final_std = std[-1]
        for idx, label in enumerate(result.labels):
            st.write(f"{label}: mean={final_mean[idx]:.2f}, std={final_std[idx]:.2f}")

    st.subheader("Performance Benchmark")
    if st.button("Run CPU vs GPU Benchmark"):
        timings = benchmark(simulator, n_particles, n_steps, dt, compare_gpu=True)
        st.json(timings)


if __name__ == "__main__":
    main()
