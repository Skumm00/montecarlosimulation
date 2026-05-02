const controls = {
  nA: document.getElementById("nA"),
  nB: document.getElementById("nB"),
  lambdaA: document.getElementById("lambdaA"),
  lambdaB: document.getElementById("lambdaB"),
  steps: document.getElementById("steps"),
  showBateman: document.getElementById("showBateman"),
};

const outputs = {
  nAValue: document.getElementById("nAValue"),
  nBValue: document.getElementById("nBValue"),
  lambdaAValue: document.getElementById("lambdaAValue"),
  lambdaBValue: document.getElementById("lambdaBValue"),
  stepsValue: document.getElementById("stepsValue"),
  stats: document.getElementById("stats"),
  downloadCsv: document.getElementById("downloadCsv"),
};

const plotTarget = document.getElementById("plot");
const dt = 0.05;
let latestData = null;
let debounceTimer = null;

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function updateLabels() {
  outputs.nAValue.textContent = formatNumber(parseInt(controls.nA.value, 10));
  outputs.nBValue.textContent = formatNumber(parseInt(controls.nB.value, 10));
  outputs.lambdaAValue.textContent = Number(controls.lambdaA.value).toFixed(2);
  outputs.lambdaBValue.textContent = Number(controls.lambdaB.value).toFixed(2);
  outputs.stepsValue.textContent = controls.steps.value;
}

function batemanAB(times, nA0, nB0, lambdaA, lambdaB) {
  const nA = new Float64Array(times.length);
  const nB = new Float64Array(times.length);
  const la = lambdaA;
  const lb = lambdaB;

  for (let i = 0; i < times.length; i += 1) {
    const t = times[i];
    const expA = Math.exp(-la * t);
    const expB = Math.exp(-lb * t);
    nA[i] = nA0 * expA;
    nB[i] = nB0 * expB + (la * nA0 / (lb - la)) * (expA - expB);
  }

  return { nA, nB };
}

function simulateMonteCarlo(params) {
  const { nA0, nB0, lambdaA, lambdaB, nSteps } = params;
  const total = nA0 + nB0;
  const states = new Uint8Array(total);

  for (let i = 0; i < nB0; i += 1) {
    states[nA0 + i] = 1;
  }

  const pA = 1 - Math.exp(-lambdaA * dt);
  const pB = 1 - Math.exp(-lambdaB * dt);

  const times = new Float64Array(nSteps + 1);
  const popA = new Float64Array(nSteps + 1);
  const popB = new Float64Array(nSteps + 1);

  popA[0] = nA0;
  popB[0] = nB0;

  for (let step = 0; step < nSteps; step += 1) {
    let countA = 0;
    let countB = 0;

    for (let i = 0; i < total; i += 1) {
      const state = states[i];
      if (state === 0) {
        if (Math.random() < pA) {
          states[i] = 1;
          countB += 1;
        } else {
          countA += 1;
        }
      } else if (state === 1) {
        if (Math.random() < pB) {
          states[i] = 2;
        } else {
          countB += 1;
        }
      }
    }

    times[step + 1] = (step + 1) * dt;
    popA[step + 1] = countA;
    popB[step + 1] = countB;
  }

  return { times, popA, popB };
}

function updateStats(data) {
  const lastIndex = data.times.length - 1;
  const stats = [
    { title: "Total time (s)", value: data.times[lastIndex].toFixed(2) },
    { title: "Final A", value: formatNumber(Math.round(data.popA[lastIndex])) },
    { title: "Final B", value: formatNumber(Math.round(data.popB[lastIndex])) },
  ];

  outputs.stats.innerHTML = stats
    .map(
      (item) => `
        <div class="stat-card">
          <div class="stat-title">${item.title}</div>
          <div class="stat-value">${item.value}</div>
        </div>
      `
    )
    .join("");
}

function plotData(data) {
  const showBateman = controls.showBateman.checked;
  const traces = [
    {
      x: Array.from(data.times),
      y: Array.from(data.popA),
      name: "MC A",
      mode: "lines",
      line: { color: "#1e6f63", width: 2.5 },
    },
    {
      x: Array.from(data.times),
      y: Array.from(data.popB),
      name: "MC B",
      mode: "lines",
      line: { color: "#c55d1f", width: 2.5 },
    },
  ];

  if (showBateman) {
    traces.push(
      {
        x: Array.from(data.times),
        y: Array.from(data.batemanA),
        name: "Bateman A",
        mode: "lines",
        line: { color: "#1f2b24", width: 2, dash: "dash" },
      },
      {
        x: Array.from(data.times),
        y: Array.from(data.batemanB),
        name: "Bateman B",
        mode: "lines",
        line: { color: "#444f48", width: 2, dash: "dash" },
      }
    );
  }

  const layout = {
    margin: { t: 30, r: 20, b: 50, l: 60 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: "Time (s)", gridcolor: "rgba(31,43,36,0.12)" },
    yaxis: { title: "Population", gridcolor: "rgba(31,43,36,0.12)" },
    legend: { orientation: "h", y: -0.2 },
  };

  Plotly.react(plotTarget, traces, layout, { displayModeBar: false });
}

function buildCsv(data) {
  const header = "time,mc_a,mc_b,bateman_a,bateman_b\n";
  let rows = "";
  for (let i = 0; i < data.times.length; i += 1) {
    rows += `${data.times[i].toFixed(4)},${data.popA[i]},${data.popB[i]},${data.batemanA[i]},${data.batemanB[i]}\n`;
  }
  return header + rows;
}

function runSimulation() {
  updateLabels();

  const nA0 = parseInt(controls.nA.value, 10);
  const nB0 = parseInt(controls.nB.value, 10);
  const lambdaA = Number(controls.lambdaA.value);
  let lambdaB = Number(controls.lambdaB.value);
  const nSteps = parseInt(controls.steps.value, 10);

  if (Math.abs(lambdaA - lambdaB) < 1e-6) {
    lambdaB += 0.01;
    outputs.lambdaBValue.textContent = `${lambdaB.toFixed(2)} (adjusted)`;
  }

  const sim = simulateMonteCarlo({ nA0, nB0, lambdaA, lambdaB, nSteps });
  const bateman = batemanAB(sim.times, nA0, nB0, lambdaA, lambdaB);

  latestData = {
    ...sim,
    batemanA: bateman.nA,
    batemanB: bateman.nB,
  };

  plotData(latestData);
  updateStats(latestData);
}

function scheduleUpdate() {
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(runSimulation, 150);
}

Object.values(controls).forEach((input) => {
  input.addEventListener("input", scheduleUpdate);
});

outputs.downloadCsv.addEventListener("click", () => {
  if (!latestData) {
    return;
  }

  const csv = buildCsv(latestData);
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "decay_simulation.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
});

runSimulation();
