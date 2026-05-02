"""
decay_chain_mc.py – Monte Carlo simulation of a multi-step radioactive decay chain.

Chain:  Isotope A → Isotope B → Isotope C → Stable (D)

Scientific Summary
------------------
Each of the N simulated nuclei is represented as one element of a compact integer
array whose value encodes its current state (0 = A, 1 = B, 2 = C, 3 = stable).
At every discrete time step Δt a single vectorised draw of N uniform random values
is compared against the exact per-step decay probabilities 1 − exp(−λΔt); Boolean
masks select the decaying nuclei for each active isotope and advance their states
in-place, eliminating all per-atom Python loops.  As N → ∞ the ensemble-mean
trajectories converge to the deterministic Bateman-equation solutions, while
finite-N runs faithfully reproduce the intrinsic Poissonian fluctuations that are
the quantum-mechanical signature of radioactive decay.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")  # headless-safe; remove for interactive display
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Simulation engine
# ──────────────────────────────────────────────────────────────────────────────

class RadioactiveDecayChain:
    """Monte Carlo engine for the three-step decay chain  A → B → C → Stable.

    Every nucleus is stored as one element of an integer ``states`` array with
    values {0 = A, 1 = B, 2 = C, 3 = stable}.  At each time step a single
    vectorised random draw of N uniform values is compared against the exact
    per-step decay probability p = 1 − exp(−λΔt); Boolean masks select the
    decaying nuclei and update their states in-place — no per-atom Python loop
    is ever executed.  The computational back-end switches transparently between
    NumPy (CPU) and CuPy (GPU).

    Parameters
    ----------
    N_A0 : int
        Initial number of nuclei of isotope A.
    lambda_A, lambda_B, lambda_C : float
        Decay constants λ (s⁻¹) for the steps A→B, B→C, and C→stable.
    N_B0, N_C0 : int, optional
        Initial populations of isotopes B and C (default 0).
    use_gpu : bool or None
        ``True`` forces CuPy (raises ``RuntimeError`` if unavailable);
        ``False`` forces NumPy; ``None`` (default) auto-detects GPU availability.

    Attributes
    ----------
    xp : module
        Active array namespace — ``numpy`` or ``cupy``.
    """

    def __init__(
        self,
        N_A0: int,
        lambda_A: float,
        lambda_B: float,
        lambda_C: float,
        N_B0: int = 0,
        N_C0: int = 0,
        use_gpu: Optional[bool] = None,
    ) -> None:
        if use_gpu is None:
            use_gpu = _CUPY_AVAILABLE
        if use_gpu and not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Set use_gpu=False or install cupy."
            )
        self.xp = cp if use_gpu else np  # type: ignore[assignment]
        self.N_A0, self.N_B0, self.N_C0 = int(N_A0), int(N_B0), int(N_C0)
        self.lambda_A = float(lambda_A)
        self.lambda_B = float(lambda_B)
        self.lambda_C = float(lambda_C)

    # ------------------------------------------------------------------
    def simulate(
        self, t_max: float, dt: float
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run the stochastic decay-chain simulation.

        Parameters
        ----------
        t_max : float
            Total simulation duration (s).
        dt : float
            Time-step size (s).  For accuracy choose λ · dt ≪ 1.

        Returns
        -------
        times : ndarray, shape (n_steps + 1,)
            Uniformly-spaced time points from 0 to t_max.
        populations : tuple of three ndarray
            ``(pop_A, pop_B, pop_C)`` — nucleus counts at every time point.
        """
        xp = self.xp
        N_total = self.N_A0 + self.N_B0 + self.N_C0
        n_steps = int(t_max / dt)

        # Exact per-step decay probabilities
        decay_probs = [
            1.0 - float(np.exp(-self.lambda_A * dt)),
            1.0 - float(np.exp(-self.lambda_B * dt)),
            1.0 - float(np.exp(-self.lambda_C * dt)),
        ]

        # Build state array: 0=A, 1=B, 2=C, 3=stable
        states = xp.zeros(N_total, dtype=xp.int8)
        if self.N_B0:
            states[self.N_A0 : self.N_A0 + self.N_B0] = 1
        if self.N_C0:
            states[self.N_A0 + self.N_B0 :] = 2

        # Pre-allocate output arrays on CPU
        pop_A = np.empty(n_steps + 1, dtype=np.int64)
        pop_B = np.empty(n_steps + 1, dtype=np.int64)
        pop_C = np.empty(n_steps + 1, dtype=np.int64)
        pop_A[0], pop_B[0], pop_C[0] = self.N_A0, self.N_B0, self.N_C0

        # ── Time integration ─────────────────────────────────────────
        for step in range(n_steps):
            # One vectorised random draw shared across all isotopes
            rand = xp.random.random(N_total)

            # Advance each active isotope: mutually exclusive Boolean masks
            for state_idx, prob in enumerate(decay_probs):
                mask = (states == state_idx) & (rand < prob)
                states[mask] = state_idx + 1

            pop_A[step + 1] = int((states == 0).sum())
            pop_B[step + 1] = int((states == 1).sum())
            pop_C[step + 1] = int((states == 2).sum())

        times = np.linspace(0.0, t_max, n_steps + 1)
        return times, (pop_A, pop_B, pop_C)

    # ------------------------------------------------------------------
    @staticmethod
    def plot(
        times: np.ndarray,
        populations: tuple[np.ndarray, np.ndarray, np.ndarray],
        title: str = "Monte Carlo Radioactive Decay Chain  (A → B → C → Stable)",
        save_path: Optional[str] = "decay_chain.png",
    ) -> plt.Figure:
        """Plot population curves and optionally save the figure to disk.

        Parameters
        ----------
        times : ndarray
            Time axis returned by :meth:`simulate`.
        populations : tuple
            ``(pop_A, pop_B, pop_C)`` returned by :meth:`simulate`.
        title : str
            Figure title.
        save_path : str or None
            File path for the PNG output; ``None`` skips saving.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        pop_A, pop_B, pop_C = populations
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(times, pop_A, lw=2, label="A (parent)")
        ax.plot(times, pop_B, lw=2, label="B (intermediate 1)")
        ax.plot(times, pop_C, lw=2, label="C (intermediate 2)")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Population (nuclei)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    @staticmethod
    def benchmark(
        N_values: Sequence[int],
        lambda_A: float = 0.5,
        lambda_B: float = 0.2,
        lambda_C: float = 0.1,
        t_max: float = 10.0,
        dt: float = 0.1,
        use_gpu: Optional[bool] = None,
        save_path: Optional[str] = "benchmark.png",
    ) -> dict:
        """Benchmark simulation wall-clock time for several particle counts.

        Parameters
        ----------
        N_values : sequence of int
            Particle counts to profile.
        lambda_A, lambda_B, lambda_C : float
            Decay constants used for all benchmark runs.
        t_max, dt : float
            Simulation window and step size (kept short for rapid profiling).
        use_gpu : bool or None
            Backend selection (see :class:`RadioactiveDecayChain`).
        save_path : str or None
            Output path for the timing chart; ``None`` skips saving.

        Returns
        -------
        timings : dict
            Mapping ``{N: elapsed_seconds}``.
        """
        timings: dict = {}
        print(f"{'N':>12}  {'Time (s)':>10}")
        print("-" * 26)
        for N in N_values:
            sim = RadioactiveDecayChain(
                N, lambda_A, lambda_B, lambda_C, use_gpu=use_gpu
            )
            t0 = time.perf_counter()
            sim.simulate(t_max, dt)
            timings[N] = time.perf_counter() - t0
            print(f"{N:>12,}  {timings[N]:>10.3f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(list(timings.keys()), list(timings.values()), "o-", lw=2, ms=7)
        ax.set_xlabel("Particle count N", fontsize=12)
        ax.set_ylabel("Wall-clock time (s)", fontsize=12)
        ax.set_title("Monte Carlo Decay Simulation — Performance Benchmark", fontsize=12)
        ax.grid(which="both", alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return timings


# ──────────────────────────────────────────────────────────────────────────────
# Execution block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── User-configurable parameters ──────────────────────────────────
    N_INITIAL = 1_000_000   # initial nuclei in isotope A
    LAMBDA_A  = 0.5         # decay constant A → B  (s⁻¹)
    LAMBDA_B  = 0.2         # decay constant B → C  (s⁻¹)
    LAMBDA_C  = 0.05        # decay constant C → stable (s⁻¹)
    T_MAX     = 40.0        # total simulation time (s)
    DT        = 0.05        # time-step size (s)
    USE_GPU   = None        # None = auto-detect CuPy

    backend = (
        "CuPy (GPU)" if (_CUPY_AVAILABLE and USE_GPU is not False) else "NumPy (CPU)"
    )
    print(f"Backend : {backend}")
    print(f"N       : {N_INITIAL:,}")

    # ── Run main simulation ────────────────────────────────────────────
    sim = RadioactiveDecayChain(
        N_INITIAL, LAMBDA_A, LAMBDA_B, LAMBDA_C, use_gpu=USE_GPU
    )
    t0 = time.perf_counter()
    times, populations = sim.simulate(T_MAX, DT)
    elapsed = time.perf_counter() - t0
    print(f"Simulation complete in {elapsed:.2f} s")

    # ── Plot population curves ─────────────────────────────────────────
    fig = RadioactiveDecayChain.plot(times, populations, save_path="decay_chain.png")
    plt.close(fig)
    print("Population plot  →  decay_chain.png")

    # ── Performance benchmark ──────────────────────────────────────────
    print("\nRunning benchmark …")
    RadioactiveDecayChain.benchmark(
        N_values=[1_000, 10_000, 100_000, 1_000_000],
        use_gpu=USE_GPU,
        save_path="benchmark.png",
    )
    print("Benchmark plot   →  benchmark.png")
