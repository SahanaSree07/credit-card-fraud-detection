const state = { sample: null };

// ---------- Navigation ----------
document.querySelectorAll(".rail-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".rail-btn").forEach((b) => b.classList.remove("is-active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    btn.classList.add("is-active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("is-active");
    if (btn.dataset.view === "dashboard") loadDashboard();
    if (btn.dataset.view === "history") loadHistory();
  });
});

// ---------- Helpers ----------
function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }
function fmtNum(x, d = 3) { return Number(x).toFixed(d); }

// ---------- Dashboard ----------
let modelChart, balanceChart;

async function loadDashboard() {
  const res = await fetch("/api/stats");
  const data = await res.json();

  document.getElementById("model-name").textContent = data.best_model.best_model;
  document.getElementById("stat-rows").textContent = data.dataset.n_rows;
  document.getElementById("stat-fraud-pct").textContent = data.dataset.fraud_pct + "%";
  document.getElementById("stat-f1").textContent = fmtNum(data.best_model.metrics["F1-Score"]);
  document.getElementById("stat-logged").textContent = data.app_usage.total_predictions_logged;

  const models = data.model_comparison;
  const labels = models.map((m) => m.Model);

  const ctx1 = document.getElementById("chart-models");
  if (modelChart) modelChart.destroy();
  modelChart = new Chart(ctx1, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Precision", data: models.map((m) => m.Precision), backgroundColor: "#33D6E0" },
        { label: "Recall", data: models.map((m) => m.Recall), backgroundColor: "#F0B94A" },
        { label: "F1-Score", data: models.map((m) => m["F1-Score"]), backgroundColor: "#FB6E7C" },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#8891A8" } } },
      scales: {
        x: { ticks: { color: "#8891A8", maxRotation: 20 }, grid: { color: "#1A2440" } },
        y: { ticks: { color: "#8891A8" }, grid: { color: "#1A2440" }, min: 0, max: 1 },
      },
    },
  });

  const ctx2 = document.getElementById("chart-balance");
  if (balanceChart) balanceChart.destroy();
  balanceChart = new Chart(ctx2, {
    type: "doughnut",
    data: {
      labels: ["Legitimate", "Fraud"],
      datasets: [{
        data: [data.dataset.n_legit, data.dataset.n_fraud],
        backgroundColor: ["#37D399", "#FB6E7C"],
        borderWidth: 0,
      }],
    },
    options: {
      plugins: { legend: { labels: { color: "#8891A8" } } },
    },
  });

  const tbody = document.querySelector("#table-models tbody");
  tbody.innerHTML = "";
  models.forEach((m) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td style="font-family:var(--sans)">${m.Model}</td>
      <td>${fmtNum(m.Accuracy)}</td><td>${fmtNum(m.Precision)}</td>
      <td>${fmtNum(m.Recall)}</td><td>${fmtNum(m["F1-Score"])}</td>
      <td>${fmtNum(m["ROC-AUC"])}</td>`;
    tbody.appendChild(tr);
  });
}

// ---------- Predict ----------
function showResult({ prediction, prediction_label, probability, model, note }, actualLabel) {
  const panel = document.getElementById("result-panel");
  panel.style.display = "block";

  const verdict = document.getElementById("result-verdict");
  verdict.textContent = prediction_label;
  verdict.className = "result-verdict " + (prediction === 1 ? "fraud" : "legit");

  document.getElementById("result-proba").textContent = fmtPct(probability);
  document.getElementById("result-model").textContent = model;

  const actualRow = document.getElementById("result-actual-row");
  if (actualLabel !== undefined && actualLabel !== null) {
    actualRow.style.display = "flex";
    document.getElementById("result-actual").textContent = actualLabel === 1 ? "Fraud" : "Legitimate";
  } else {
    actualRow.style.display = "none";
  }

  document.getElementById("result-note").textContent = note || "";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("quick-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const amount = parseFloat(document.getElementById("quick-amount").value);
  const time = parseFloat(document.getElementById("quick-time").value);
  const res = await fetch("/api/predict/quick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ amount, time }),
  });
  const data = await res.json();
  showResult(data);
});

document.getElementById("btn-sample").addEventListener("click", async () => {
  const res = await fetch("/api/sample");
  const data = await res.json();
  state.sample = data;

  const preview = document.getElementById("sample-preview");
  preview.style.display = "block";
  preview.textContent = `Amount: $${data.features.Amount.toFixed(2)}  |  Time: ${data.features.Time.toFixed(
    0
  )}  |  True label: ${data.actual_label === 1 ? "FRAUD" : "legitimate"}\n\n` +
    Object.entries(data.features)
      .filter(([k]) => k.startsWith("V"))
      .map(([k, v]) => `${k}=${v.toFixed(3)}`)
      .join("  ");

  document.getElementById("btn-sample-predict").style.display = "inline-block";
});

document.getElementById("btn-sample-predict").addEventListener("click", async () => {
  if (!state.sample) return;
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features: state.sample.features,
      actual_label: state.sample.actual_label,
      source: "sample",
    }),
  });
  const data = await res.json();
  showResult(data, state.sample.actual_label);
});

// ---------- History ----------
async function loadHistory() {
  const res = await fetch("/api/history?limit=100");
  const rows = await res.json();
  const tbody = document.querySelector("#table-history tbody");
  const empty = document.getElementById("history-empty");
  tbody.innerHTML = "";

  if (rows.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const predBadge = r.prediction === 1
      ? '<span class="badge badge-fraud">Fraud</span>'
      : '<span class="badge badge-legit">Legitimate</span>';
    const actual = r.actual_label === null || r.actual_label === undefined
      ? "—"
      : (r.actual_label === 1 ? "Fraud" : "Legitimate");
    tr.innerHTML = `<td>${r.created_at.replace("T", " ")}</td>
      <td>$${Number(r.amount).toFixed(2)}</td>
      <td>${predBadge}</td>
      <td>${fmtPct(r.probability)}</td>
      <td>${actual}</td>
      <td>${r.source}</td>`;
    tbody.appendChild(tr);
  });
}

document.getElementById("btn-refresh-history").addEventListener("click", loadHistory);
document.getElementById("btn-clear-history").addEventListener("click", async () => {
  if (!confirm("Clear all logged predictions?")) return;
  await fetch("/api/history", { method: "DELETE" });
  loadHistory();
});

// ---------- Init ----------
loadDashboard();
