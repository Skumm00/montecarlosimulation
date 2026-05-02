"""
stochastic_simulator.py - Generalized Monte Carlo stochastic process simulator.

Provides a vectorized engine for Markov state transitions using either a
transition probability matrix or event-rate matrix, with optional CuPy backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


@dataclass
class SimulationResult:
    times: np.ndarray
    populations: np.ndarray
    labels: Tuple[str, ...]
    backend: str

    def to_csv(self) -> str:
        header = "time," + ",".join(self.labels)
        rows = [header]
        for i, t in enumerate(self.times):
            values = ",".join(str(int(v)) for v in self.populations[i])
            rows.append(f"{t:.6f},{values}")
        return "\n".join(rows) + "\n"


class StochasticProcessSimulator:
    """Generalized Monte Carlo simulator for Markov state transitions."""

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        rate_matrix: Optional[np.ndarray] = None,
        labels: Optional[Iterable[str]] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        if transition_matrix is None and rate_matrix is None:
            raise ValueError("Provide transition_matrix or rate_matrix.")

        if transition_matrix is not None and rate_matrix is not None:
            raise ValueError("Provide only one of transition_matrix or rate_matrix.")

        if use_gpu is None:
            use_gpu = _CUPY_AVAILABLE
        if use_gpu and not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not installed. Set use_gpu=False or install cupy.")

        self.xp = cp if use_gpu else np  # type: ignore[assignment]
        self.backend = "CuPy" if use_gpu else "NumPy"

        if transition_matrix is not None:
            self.transition_matrix = self._validate_transition_matrix(
                np.asarray(transition_matrix, dtype=np.float64)
            )
            self.rate_matrix = None
        else:
            self.rate_matrix = self._validate_rate_matrix(
                np.asarray(rate_matrix, dtype=np.float64)
            )
            self.transition_matrix = None

        if labels is None:
            n_states = self.num_states
            self.labels = tuple(f"State {i}" for i in range(n_states))
        else:
            self.labels = tuple(labels)

    @property
    def num_states(self) -> int:
        matrix = self.transition_matrix if self.transition_matrix is not None else self.rate_matrix
        return int(matrix.shape[0])

    @staticmethod
    def _validate_transition_matrix(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition_matrix must be square.")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Rows of transition_matrix must sum to 1.")
        if np.any(matrix < 0):
            raise ValueError("transition_matrix cannot contain negative values.")
        return matrix

    @staticmethod
    def _validate_rate_matrix(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("rate_matrix must be square.")
        if np.any(matrix < 0):
            raise ValueError("rate_matrix cannot contain negative values.")
        return matrix

    def _rates_to_transition(self, dt: float) -> np.ndarray:
        rates = self.rate_matrix
        if rates is None:
            raise RuntimeError("rate_matrix not set.")
        n_states = rates.shape[0]
        trans = np.zeros_like(rates)
        for i in range(n_states):
            row = rates[i]
            total_rate = float(row.sum())
            if total_rate == 0:
                trans[i, i] = 1.0
                continue
            leave_prob = 1.0 - np.exp(-total_rate * dt)
            trans[i] = (row / total_rate) * leave_prob
            trans[i, i] = 1.0 - leave_prob
        return trans

    def simulate(
        self,
        n_particles: int,
        n_steps: int,
        dt: float,
        initial_state: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """Run Monte Carlo simulation.

        Parameters
        ----------
        n_particles : int
            Number of particles to simulate.
        n_steps : int
            Number of time steps.
        dt : float
            Time step size.
        initial_state : ndarray, optional
            Vector of initial counts per state (length = n_states).
        """
        n_states = self.num_states
        xp = self.xp

        if initial_state is None:
            counts = np.zeros(n_states, dtype=np.int64)
            counts[0] = n_particles
        else:
            counts = np.asarray(initial_state, dtype=np.int64)
            if counts.shape[0] != n_states:
                raise ValueError("initial_state length must match number of states.")
            if counts.sum() != n_particles:
                raise ValueError("initial_state must sum to n_particles.")

        transition = (
            self.transition_matrix
            if self.transition_matrix is not None
            else self._rates_to_transition(dt)
        )
        if self.transition_matrix is not None and dt <= 0:
            raise ValueError("dt must be positive.")

        # Move to backend
        transition_xp = xp.asarray(transition)
        counts_xp = xp.asarray(counts)

        populations = np.zeros((n_steps + 1, n_states), dtype=np.int64)
        populations[0] = counts

        rng = xp.random.default_rng()

        for step in range(n_steps):
            next_counts = xp.zeros_like(counts_xp)
            for state in range(n_states):
                count = int(counts_xp[state])
                if count == 0:
                    continue
                probs = transition_xp[state]
                draws = rng.multinomial(count, probs)
                next_counts += draws
            counts_xp = next_counts
            populations[step + 1] = np.asarray(counts_xp)

        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        return SimulationResult(times, populations, self.labels, self.backend)

    def sensitivity_analysis(
        self,
        n_particles: int,
        n_steps: int,
        dt: float,
        n_runs: int = 10,
        perturb_fraction: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run sensitivity analysis by perturbing rates or transitions."""
        rng = np.random.default_rng(seed)
        results = []

        for _ in range(n_runs):
            if self.rate_matrix is not None:
                noise = rng.normal(0.0, perturb_fraction, self.rate_matrix.shape)
                perturbed = np.clip(self.rate_matrix * (1.0 + noise), 0.0, None)
                simulator = StochasticProcessSimulator(
                    rate_matrix=perturbed,
                    labels=self.labels,
                    use_gpu=self.backend == "CuPy",
                )
            else:
                noise = rng.normal(0.0, perturb_fraction, self.transition_matrix.shape)
                perturbed = np.clip(self.transition_matrix * (1.0 + noise), 0.0, None)
                perturbed = perturbed / perturbed.sum(axis=1, keepdims=True)
                simulator = StochasticProcessSimulator(
                    transition_matrix=perturbed,
                    labels=self.labels,
                    use_gpu=self.backend == "CuPy",
                )

            result = simulator.simulate(n_particles, n_steps, dt)
            results.append(result.populations)

        stacked = np.stack(results, axis=0)
        return {
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
        }

    @staticmethod
    def templates() -> Dict[str, Dict[str, object]]:
        """Preset templates for common stochastic processes."""
        return {
            "Chemistry/Physics": {
                "labels": ("A", "B", "C", "Stable"),
                "rate_matrix": np.array(
                    [
                        [0.0, 0.45, 0.0, 0.0],
                        [0.0, 0.0, 0.25, 0.0],
                        [0.0, 0.0, 0.0, 0.12],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "Epidemiology (SIR)": {
                "labels": ("S", "I", "R"),
                "rate_matrix": np.array(
                    [
                        [0.0, 0.35, 0.0],
                        [0.0, 0.0, 0.18],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            },
            "Infrastructure (Packets)": {
                "labels": ("OK", "Congested", "Dropped", "Recovered"),
                "rate_matrix": np.array(
                    [
                        [0.0, 0.18, 0.02, 0.0],
                        [0.0, 0.0, 0.12, 0.16],
                        [0.0, 0.0, 0.0, 0.08],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
        }


def benchmark(
    simulator: StochasticProcessSimulator,
    n_particles: int,
    n_steps: int,
    dt: float,
    compare_gpu: bool = True,
) -> Dict[str, float]:
    """Benchmark CPU vs GPU performance if available."""
    import time

    timings = {}

    cpu_sim = StochasticProcessSimulator(
        transition_matrix=simulator.transition_matrix,
        rate_matrix=simulator.rate_matrix,
        labels=simulator.labels,
        use_gpu=False,
    )
    t0 = time.perf_counter()
    cpu_sim.simulate(n_particles, n_steps, dt)
    timings["cpu_seconds"] = time.perf_counter() - t0

    if compare_gpu and _CUPY_AVAILABLE:
        gpu_sim = StochasticProcessSimulator(
            transition_matrix=simulator.transition_matrix,
            rate_matrix=simulator.rate_matrix,
            labels=simulator.labels,
            use_gpu=True,
        )
        t0 = time.perf_counter()
        gpu_sim.simulate(n_particles, n_steps, dt)
        timings["gpu_seconds"] = time.perf_counter() - t0

    return timings
