let evidence;
let campaigns = [];
let totals = {};

function prepareEvidence(payload) {
  evidence = payload;
  campaigns = payload.campaigns.map((item) => ({
    ...item,
    audience: item.id,
    ctr: (item.clicks / item.impressions) * 100,
    roas: item.revenue / item.spend,
    cpa: item.spend / item.conversions,
    utilisation: item.spend / item.planned,
  }));
  totals = campaigns.reduce(
  (result, item) => ({
    spend: result.spend + item.spend,
    revenue: result.revenue + item.revenue,
    impressions: result.impressions + item.impressions,
    clicks: result.clicks + item.clicks,
    conversions: result.conversions + item.conversions,
  }),
  { spend: 0, revenue: 0, impressions: 0, clicks: 0, conversions: 0 },
  );
}

const state = { route: location.hash.slice(1) || "overview", campaignQuery: "", sort: "roas", shift: 15 };
const workspace = document.querySelector("#workspace");
const toast = document.querySelector(".toast");
const money = (value, digits = 0) => new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP", maximumFractionDigits: digits }).format(value);
const number = (value) => new Intl.NumberFormat("en-GB", { notation: value > 999999 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
const pct = (value) => `${value.toFixed(1)}%`;
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 2200);
}

function metric(label, value, detail, tone = "") {
  return `<article class="metric ${tone}"><span>${label}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function overview() {
  const blendedRoas = totals.revenue / totals.spend;
  const atRisk = campaigns.filter((item) => item.under > 10);
  return `
    <section class="page-head">
      <div><p class="eyebrow">SAMPLE DATA · ${campaigns[0].days} DAYS</p><h1>Spend, revenue and pacing for six sample campaigns</h1></div>
      <p class="lede">The pipeline joins the delivery, conversion and pacing CSV files committed to the repository. Every figure on this page comes from those fixtures.</p>
    </section>
    <section class="metric-grid">
      ${metric("Attributed revenue", money(totals.revenue), `${totals.conversions} conversion rows in the fixtures`, "positive")}
      ${metric("Recorded spend", money(totals.spend), `${number(totals.impressions)} impressions`)}
      ${metric("Blended ROAS · fixture", `${blendedRoas.toFixed(2)}×`, "Fixture revenue divided by fixture spend", "positive")}
      ${metric("Pacing watchlist", String(atRisk.length), "Campaigns under pace on >10 days", "warning")}
    </section>
    <section class="split">
      <article class="panel action-panel">
        <header><div><h2>Campaigns to review</h2></div><span class="count">${atRisk.length + 1}</span></header>
        ${atRisk.map((item, index) => `
          <button class="action-row" data-open-campaign="${item.id}">
            <span class="priority">${String(index + 1).padStart(2, "0")}</span>
            <span><strong>${item.name}</strong><small>Under pace on ${item.under} of ${item.days} observed days</small></span>
            <span class="action-value">${pct(item.utilisation * 100)}<small>utilisation</small></span>
          </button>`).join("")}
        <button class="action-row" data-route-link="campaigns">
          <span class="priority">03</span>
          <span><strong>Adventure Seekers</strong><small>Most impressions and the lowest ROAS (${campaigns[3].roas.toFixed(2)}×)</small></span>
          <span class="action-value">${money(campaigns[3].spend)}<small>spend</small></span>
        </button>
      </article>
      <article class="panel">
        <header><div><h2>Revenue by audience</h2></div><span class="stamp">Fixture data</span></header>
        <div class="bars">
          ${[...campaigns].sort((a, b) => b.revenue - a.revenue).map((item) => `
            <div class="bar-row"><span>${item.name}</span><div><i style="width:${(item.revenue / Math.max(...campaigns.map((row) => row.revenue))) * 100}%"></i></div><strong>${money(item.revenue)}</strong></div>
          `).join("")}
        </div>
      </article>
    </section>
    <section class="decision-strip">
      <div><strong>Check pacing and attribution before you move budget.</strong></div>
      <p>In the fixture data, Luxury Honeymoon Couples brings in 46% of attributed revenue and Premium Travelers ran under pace on ${campaigns[0].under} of ${campaigns[0].days} days.</p>
      <button data-route-link="pacing">Open pacing lab →</button>
    </section>`;
}

function campaignsView() {
  const query = state.campaignQuery.toLowerCase();
  const rows = campaigns
    .filter((item) => `${item.name} ${item.audience}`.toLowerCase().includes(query))
    .sort((a, b) => state.sort === "roas" ? b.roas - a.roas : state.sort === "spend" ? b.spend - a.spend : b.conversions - a.conversions);
  return `
    <section class="page-head compact">
      <div><p class="eyebrow">GOLD LAYER</p><h1>Six sample campaigns by audience</h1></div>
      <p class="lede">Values come from the committed performance and conversion fixtures.</p>
    </section>
    <section class="toolbar">
      <label><span>Search campaigns</span><input id="campaign-search" value="${escapeHtml(state.campaignQuery)}" placeholder="Audience or segment" /></label>
      <label><span>Order by</span><select id="campaign-sort"><option value="roas" ${state.sort === "roas" ? "selected" : ""}>ROAS</option><option value="spend" ${state.sort === "spend" ? "selected" : ""}>Spend</option><option value="conversions" ${state.sort === "conversions" ? "selected" : ""}>Conversions</option></select></label>
    </section>
    <section class="campaign-grid">
      ${rows.map((item) => `
        <article class="campaign-card">
          <header><span class="campaign-id">${item.id}</span><span class="health ${item.roas < 2 ? "risk" : ""}">${item.roas < 2 ? "ROAS below 2×" : "ROAS 2× or more"}</span></header>
          <h2>${item.name}</h2><p>${item.audience}</p>
          <div class="campaign-kpis"><div><span>ROAS</span><strong>${item.roas.toFixed(2)}×</strong></div><div><span>CTR</span><strong>${pct(item.ctr)}</strong></div><div><span>CPA</span><strong>${money(item.cpa)}</strong></div></div>
          <div class="spend-line"><span>Spend ${money(item.spend)}</span><span>Revenue ${money(item.revenue)}</span></div>
          <button data-open-campaign="${item.id}">Inspect evidence</button>
        </article>`).join("") || `<p class="empty">No campaigns match that search.</p>`}
    </section>`;
}

function pacingView() {
  const shift = state.shift;
  const source = campaigns[0];
  const extra = source.planned * (shift / 100);
  const guardedReturn = extra * source.roas * 0.7;
  return `
    <section class="page-head compact">
      <div><p class="eyebrow">SCENARIO</p><h1>Try a budget change before you make it</h1></div>
      <p class="lede">The calculation uses the assumptions listed below. It does not change any budget.</p>
    </section>
    <section class="scenario-layout">
      <article class="panel control-panel">
        <h2>Premium Travelers pacing</h2>
        <label class="range-label" for="shift"><span>Planned budget adjustment</span><strong>+${shift}%</strong></label>
        <input id="shift" type="range" min="0" max="40" step="5" value="${shift}" />
        <div class="range-scale"><span>No change</span><span>+40%</span></div>
        <div class="guardrails">
          <label><input type="checkbox" checked disabled /> Retain 20% cash guardrail</label>
          <label><input type="checkbox" checked disabled /> Discount observed ROAS by 30%</label>
          <label><input type="checkbox" checked disabled /> Require operator approval</label>
        </div>
      </article>
      <article class="panel outcome-panel">
        <p class="eyebrow">MODELLED OUTCOME</p>
        <div class="outcome-number"><span>Additional planned spend</span><strong>${money(extra)}</strong></div>
        <div class="outcome-number"><span>Attributed revenue after 30% haircut</span><strong>${money(guardedReturn)}</strong></div>
        <div class="outcome-number"><span>Assumption</span><strong>${(source.roas * 0.7).toFixed(2)}× ROAS</strong></div>
        <button class="primary-action" id="record-scenario">Record review scenario</button>
        <small>This action saves nothing to an ad platform.</small>
      </article>
    </section>
    <section class="panel assumption-table"><header><div><p class="eyebrow">NOT A FORECAST</p><h2>Assumptions</h2></div></header>
      <table><thead><tr><th>Input</th><th>Observed</th><th>Scenario treatment</th></tr></thead><tbody>
        <tr><td>ROAS</td><td>${source.roas.toFixed(2)}× in fixture</td><td>30% haircut</td></tr>
        <tr><td>Pacing</td><td>${source.under}/${source.days} days under pace</td><td>Budget capacity only</td></tr>
        <tr><td>Attribution</td><td>Mixed 7-day click / 1-day view</td><td>No incrementality claim</td></tr>
      </tbody></table>
    </section>`;
}

function qualityView() {
  const checks = evidence.quality.checks.map((check) => [check.name.replaceAll("_", " "), `observed ${check.observed}; expected ${check.expected}`, check.status === "pass" ? "Pass" : "Fail"]);
  return `
    <section class="page-head compact"><div><p class="eyebrow">RULE-BASED CHECKS</p><h1>Data quality checks</h1></div><p class="lede">Key, reconciliation and provenance checks from the last pipeline run.</p></section>
    <section class="quality-summary"><div class="quality-score"><span>${checks.filter((row) => row[2] === "Pass").length} / ${checks.length}</span><strong>checks pass</strong><small>Generated by the local pipeline, with source hashes recorded.</small></div>
      <div class="quality-copy"><h2>${evidence.quality.status === "pass" ? "Integrity checks pass" : "Blocked"}</h2><p>The pipeline writes these results to docs/generated/evidence.json, which is committed with the code.</p></div></section>
    <section class="panel check-list">${checks.map(([name, evidence, result]) => `<div class="check-row"><span class="check-mark ${result === "Review" ? "review" : ""}">${result === "Review" ? "!" : "✓"}</span><span><strong>${name}</strong><small>${evidence}</small></span><b class="${result.toLowerCase()}">${result}</b></div>`).join("")}</section>`;
}

function lineageView() {
  const layers = [
    ["RAW", "4 CSV inputs", "Audience, pacing, conversions and delivery"],
    ["BRONZE", "Typed landing tables", "Immutable source-shaped records"],
    ["SILVER", "Clean campaign facts", "Dates, joins and quality flags"],
    ["GOLD", "Decision features", "ROAS, CPA, pacing and model inputs"],
    ["MODEL", "XGBoost artefacts", "Performance and under-pacing risk"],
    ["REVIEW", "Streamlit + agents", "Dashboard and rule-based review"],
  ];
  return `
    <section class="page-head compact"><div><h1>From raw CSV to model output</h1></div><p class="lede">The Python project rebuilds each layer locally. This page shows the output of the last run.</p></section>
    <section class="lineage">${layers.map(([tag, title, description], index) => `<article><span>${tag}</span><div><strong>${title}</strong><p>${description}</p></div>${index < layers.length - 1 ? `<i aria-hidden="true">→</i>` : ""}</article>`).join("")}</section>
    <section class="split lineage-detail"><article class="panel"><h2>Input contract</h2><ul>${evidence.contracts.sources.map((source) => `<li>${source.rows} rows · ${source.name}</li>`).join("")}</ul></article><article class="panel"><h2>Generated by the pipeline</h2><ul><li>Contract ${evidence.contracts.contract_version}</li><li>Quality status ${evidence.quality.status}</li><li>Models include dated holdout metadata</li></ul><a class="inline-link" href="https://github.com/MatthewPaver/marketing-ml-lakehouse#canonical-setup">Run the engine locally →</a></article></section>
    ${modelEvidence()}`;
}

function modelEvidence() {
  const model = evidence.models?.bookings;
  if (!model) return "";
  const m = model.metrics;
  const ci = (pair) => `[${pair[0].toFixed(2)}, ${pair[1].toFixed(2)}]`;
  const overlap = m.mae_95pct_bootstrap[0] <= m.baseline_mae_95pct_bootstrap[1] && m.baseline_mae_95pct_bootstrap[0] <= m.mae_95pct_bootstrap[1];
  return `<section class="panel" id="model-evidence"><p class="eyebrow">NEXT-DAY HOLDOUT · GENERATED ${escapeHtml(evidence.generated_at.slice(0, 10))}</p><h2>${overlap ? "No demonstrated skill over the prior-day baseline" : "Model interval separates from baseline"}</h2>
    <ul><li>Holdout: ${model.split.test_rows} rows (${model.split.test_start} to ${model.split.test_end}), trained on ${model.split.train_rows}</li>
    <li>Bookings MAE ${m.mae.toFixed(3)} ${ci(m.mae_95pct_bootstrap)} vs persistence baseline ${m.baseline_mae.toFixed(3)} ${ci(m.baseline_mae_95pct_bootstrap)}</li>
    <li>Point skill ${(m.skill_over_baseline * 100).toFixed(0)}%${overlap ? ", but the 95% bootstrap intervals overlap, so this fixture cannot distinguish the model from the baseline" : ""}</li></ul></section>`;
}

function render() {
  const allowed = ["overview", "campaigns", "pacing", "quality", "lineage"];
  if (!allowed.includes(state.route)) state.route = "overview";
  document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.route === state.route));
  workspace.innerHTML = ({ overview, campaigns: campaignsView, pacing: pacingView, quality: qualityView, lineage: lineageView })[state.route]();
  bind();
}

function bind() {
  document.querySelectorAll("[data-route-link]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.routeLink)));
  document.querySelectorAll("[data-open-campaign]").forEach((button) => button.addEventListener("click", () => { state.campaignQuery = button.dataset.openCampaign; navigate("campaigns"); }));
  document.querySelector("#campaign-search")?.addEventListener("input", (event) => { state.campaignQuery = event.target.value; render(); document.querySelector("#campaign-search")?.focus(); });
  document.querySelector("#campaign-sort")?.addEventListener("change", (event) => { state.sort = event.target.value; render(); });
  document.querySelector("#shift")?.addEventListener("input", (event) => { state.shift = Number(event.target.value); render(); });
  document.querySelector("#record-scenario")?.addEventListener("click", () => showToast("Scenario recorded in this demo session"));
}

function navigate(route) {
  state.route = route;
  history.replaceState(null, "", `#${route}`);
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.route)));
window.addEventListener("hashchange", () => { state.route = location.hash.slice(1) || "overview"; render(); });
fetch("generated/evidence.json")
  .then((response) => { if (!response.ok) throw new Error(`Evidence load failed: ${response.status}`); return response.json(); })
  .then((payload) => { prepareEvidence(payload); render(); })
  .catch((error) => { workspace.innerHTML = `<p class="empty">Generated pipeline evidence is unavailable: ${escapeHtml(error.message)}</p>`; });
