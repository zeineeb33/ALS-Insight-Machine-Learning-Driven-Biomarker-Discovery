"use strict";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : String(child));
  return node;
}

async function getJSON(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw Object.assign(new Error(body.detail || response.statusText), { body });
  return body;
}

const fmt = (value, digits = 3) => {
  if (value === null || value === undefined) return "NA";
  if (value !== 0 && Math.abs(value) < 1e-3) return value.toExponential(2);
  return value.toFixed(digits);
};
const pct = (value) => `${(value * 100).toFixed(0)}%`;

function plotLayout(xTitle, yTitle) {
  const axis = (title) => ({
    title: { text: title, font: { size: 12, color: css("--ink-2") } },
    gridcolor: css("--grid"),
    zerolinecolor: css("--axis"),
    linecolor: css("--axis"),
    tickfont: { color: css("--muted"), size: 11 },
  });
  return {
    margin: { l: 56, r: 12, t: 8, b: 48 },
    paper_bgcolor: css("--surface"),
    plot_bgcolor: css("--surface"),
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: css("--ink") },
    xaxis: axis(xTitle),
    yaxis: axis(yTitle),
    legend: { orientation: "h", y: -0.2, font: { color: css("--ink-2"), size: 11 } },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink") } },
  };
}
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function renderTiles(summary) {
  const de = summary.differential_expression;
  const cls = summary.classification;
  const samples = summary.samples;
  const perm = cls.permutation_test;
  const auc = (name) => cls.models[name].metrics.roc_auc;
  const tiles = [
    ["Samples analysed", samples.n_samples_analyzed,
      Object.entries(samples.samples_per_condition).map(([k, v]) => `${v} ${k}`).join(" / ")],
    [`DE features (FDR < ${de.alpha})`, de.n_significant.toLocaleString(),
      `of ${de.n_tested.toLocaleString()} tested`],
    ["Elastic net ROC AUC", fmt(auc("elastic_net").mean, 2), `± ${fmt(auc("elastic_net").std, 2)} across outer folds`],
    ["XGBoost ROC AUC", fmt(auc("xgboost").mean, 2), `± ${fmt(auc("xgboost").std, 2)} across outer folds`],
  ];
  if (perm && perm.n_permutations) {
    tiles.push(["Permutation p-value", fmt(perm.p_value, 3), `${perm.n_permutations} label permutations`]);
  }
  const container = document.getElementById("tiles");
  container.replaceChildren(
    ...tiles.map(([label, value, detail]) =>
      el("div", { class: "tile" }, el("div", { class: "label" }, label),
        el("div", { class: "value" }, value), el("div", { class: "detail" }, detail)))
  );
  document.getElementById("disclaimer").textContent = summary.disclaimer;
  document.getElementById("subtitle").textContent =
    `${summary.contrast.case} vs ${summary.contrast.control} · GEO ${summary.dataset.accession}`;
  const footer = document.getElementById("footer");
  footer.replaceChildren(
    `Results ${summary.results_version} · generated ${summary.generated_at} · data: `,
    el("a", { href: summary.dataset.url }, summary.dataset.accession),
    summary.dataset.citation ? ` (${summary.dataset.citation})` : ""
  );
}

function geneCard(gene) {
  const de = gene.differential_expression;
  const block = (title, rows) =>
    el("div", {}, el("div", { class: "group-title" }, title),
      el("dl", {}, ...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));
  return el("div", { class: "gene-card" },
    el("h3", {}, `${gene.symbol} `, el("small", {}, `(${gene.feature_type})`)),
    block("Differential expression", [
      ["log2FC (shrunk)", fmt(de.log2_fold_change, 3)],
      ["log2FC (MLE)", fmt(de.log2_fold_change_mle, 3)],
      ["padj", fmt(de.padj)],
      ["Significant", de.significant ? "yes" : "no"],
      ["DE rank", de.rank.toLocaleString()],
    ]),
    block("Elastic net (nested CV)", [
      ["Selected in", pct(gene.elastic_net.selection_frequency) + " of folds"],
      ["Mean coefficient", fmt(gene.elastic_net.mean_coefficient, 4)],
      ["Rank", gene.elastic_net.rank.toLocaleString()],
    ]),
    block("XGBoost (nested CV)", [
      ["Used in", pct(gene.xgboost.selection_frequency) + " of folds"],
      ["Mean gain", fmt(gene.xgboost.mean_gain, 3)],
      ["Rank", gene.xgboost.rank.toLocaleString()],
    ]),
    block("Consensus", [
      ["Rank", gene.consensus.rank.toLocaleString()],
      ["Score", fmt(gene.consensus.score, 4)],
    ])
  );
}

let volcanoData = null;

async function showGene(symbol) {
  const target = document.getElementById("gene-result");
  try {
    const gene = await getJSON(`/genes/${encodeURIComponent(symbol)}`);
    target.replaceChildren(geneCard(gene));
    highlightVolcano(gene.symbol);
  } catch (err) {
    const suggestions = err.body && err.body.suggestions ? err.body.suggestions : [];
    target.replaceChildren(
      el("p", { class: "error" }, err.message),
      suggestions.length ? el("p", {}, `Did you mean: ${suggestions.join(", ")}?`) : ""
    );
  }
}

function highlightVolcano(symbol) {
  if (!volcanoData) return;
  const i = volcanoData.symbol.indexOf(symbol);
  const plot = document.getElementById("volcano");
  const update = i < 0 ? { x: [[]], y: [[]], text: [[]] } : {
    x: [[volcanoData.log2_fold_change[i]]], y: [[volcanoData.neg_log10_padj[i]]], text: [[symbol]],
  };
  Plotly.restyle(plot, update, [3]);
}

function renderVolcano(data) {
  volcanoData = data;
  const groups = { up: [], down: [], ns: [] };
  data.symbol.forEach((_, i) => {
    const key = !data.significant[i] ? "ns" : data.log2_fold_change[i] > 0 ? "up" : "down";
    groups[key].push(i);
  });
  const trace = (idx, name, color, size) => ({
    type: "scattergl", mode: "markers", name: `${name} (${idx.length.toLocaleString()})`,
    x: idx.map((i) => data.log2_fold_change[i]), y: idx.map((i) => data.neg_log10_padj[i]),
    text: idx.map((i) => data.symbol[i]),
    hovertemplate: "<b>%{text}</b><br>log2FC %{x:.3f}<br>-log10 padj %{y:.2f}<extra></extra>",
    marker: { color, size, opacity: 0.75 },
  });
  const traces = [
    trace(groups.ns, "not significant", css("--neutral"), 4),
    trace(groups.down, "lower in ALS", css("--down"), 5),
    trace(groups.up, "higher in ALS", css("--up"), 5),
    { type: "scatter", mode: "markers+text", name: "selected", x: [], y: [], text: [],
      textposition: "top center", textfont: { color: css("--ink") }, hoverinfo: "skip",
      marker: { size: 14, color: "rgba(0,0,0,0)", line: { color: css("--ink"), width: 2 } } },
  ];
  const layout = plotLayout("log2 fold change (ALS vs control, shrunk)", "-log10 adjusted p-value");
  layout.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: -Math.log10(data.alpha),
    y1: -Math.log10(data.alpha), line: { color: css("--muted"), width: 1 } }];
  const plot = document.getElementById("volcano");
  Plotly.newPlot(plot, traces, layout, plotConfig);
  plot.on("plotly_click", (ev) => ev.points[0] && ev.points[0].text && showGene(ev.points[0].text));
}

function renderPCA(data) {
  const conditions = [...new Set(data.samples.map((s) => s.condition))];
  const colors = [css("--series-1"), css("--series-2")];
  const traces = conditions.map((condition, k) => {
    const pts = data.samples.filter((s) => s.condition === condition);
    return {
      type: "scatter", mode: "markers", name: `${condition} (${pts.length})`,
      x: pts.map((s) => s.pc1), y: pts.map((s) => s.pc2),
      text: pts.map((s) => `${s.sample_id} · ${s.region}${s.outlier ? " · outlier" : ""}`),
      hovertemplate: "%{text}<br>PC1 %{x:.1f}, PC2 %{y:.1f}<extra></extra>",
      marker: {
        color: colors[k % colors.length], size: 9, opacity: 0.85,
        symbol: pts.map((s) => (s.region === "motor" ? "triangle-up" : "circle")),
        line: { color: pts.map((s) => (s.outlier ? css("--ink") : css("--surface"))),
          width: pts.map((s) => (s.outlier ? 2.5 : 1)) },
      },
    };
  });
  const v = data.explained_variance_ratio;
  const layout = plotLayout(`PC1 (${(v[0] * 100).toFixed(1)}%)`, `PC2 (${(v[1] * 100).toFixed(1)}%)`);
  Plotly.newPlot(document.getElementById("pca"), traces, layout, plotConfig);
}

function renderTable(biomarkers) {
  const tbody = document.querySelector("#top-table tbody");
  tbody.replaceChildren(...biomarkers.items.map((g) => {
    const row = el("tr", { tabindex: "0" },
      el("td", {}, g.consensus.rank), el("td", {}, g.symbol),
      el("td", {}, fmt(g.differential_expression.log2_fold_change, 2)),
      el("td", {}, fmt(g.differential_expression.padj)),
      el("td", {}, pct(g.elastic_net.selection_frequency)),
      el("td", {}, pct(g.xgboost.selection_frequency)));
    const open = () => { document.getElementById("gene-input").value = g.symbol; showGene(g.symbol); };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => e.key === "Enter" && open());
    return row;
  }));
}

function wireSearch() {
  const input = document.getElementById("gene-input");
  const options = document.getElementById("gene-options");
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (!q) return;
    timer = setTimeout(async () => {
      const matches = await getJSON(`/genes?q=${encodeURIComponent(q)}&limit=10`);
      options.replaceChildren(...matches.map((m) => el("option", { value: m.symbol })));
    }, 150);
  });
  document.getElementById("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (input.value.trim()) showGene(input.value.trim());
  });
}

async function main() {
  wireSearch();
  const [summary, biomarkers, volcano, pca] = await Promise.all([
    getJSON("/summary"), getJSON("/biomarkers?top=25"), getJSON("/plots/volcano"), getJSON("/plots/pca"),
  ]);
  renderTiles(summary);
  renderTable(biomarkers);
  renderVolcano(volcano);
  renderPCA(pca);
  if (biomarkers.items.length) showGene(biomarkers.items[0].symbol);
}

main().catch((err) => {
  document.getElementById("tiles").replaceChildren(el("p", { class: "error" }, `Failed to load results: ${err.message}`));
});
