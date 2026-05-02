# montecarlosimulation

## Web app (GitHub Pages friendly)

This repository includes a static web app that runs fully in the browser and is
ready for GitHub Pages.

Files:
- index.html
- styles.css
- app.js

What the website can do:
- Run a Monte Carlo simulation for a two-step decay chain (A -> B -> stable)
- Adjust initial populations (N_A0, N_B0), decay constants (lambda_A, lambda_B), and total steps
- Update plots instantly as sliders move (no server required)
- Toggle Bateman equation overlays to compare deterministic vs stochastic results
- Show summary stats (total time, final A, final B)
- Export all time-series data to CSV (MC + Bateman)

### Run locally

Open index.html in a browser, or use a simple local server:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.


## Python simulation

The original Python Monte Carlo simulation remains in decay_chain_mc.py.

## Stochastic Process Simulator (modular)

Core engine: stochastic_simulator.py

Example usage:

```python
from stochastic_simulator import StochasticProcessSimulator

sim = StochasticProcessSimulator(
	rate_matrix=[[0.0, 0.4], [0.0, 0.0]],
	labels=["A", "B"],
)
result = sim.simulate(n_particles=1_000_000, n_steps=200, dt=0.05)
print(result.populations[-1])
```

### Streamlit dashboard

Run locally:

```bash
streamlit run streamlit_app.py
```

Features:
- Custom transition matrix or rate matrix input
- Sensitivity analysis (perturbed parameters)
- CPU vs GPU benchmark (when CuPy is installed)
- CSV export