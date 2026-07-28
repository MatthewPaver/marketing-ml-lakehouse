const campaigns = [
  { id: "AS_001", name: "Premium Travelers", audience: "25–45 · high-value customers", impressions: 981730, clicks: 19948, spend: 6537, planned: 10500, revenue: 73373.75, conversions: 62, under: 21, days: 22 },
  { id: "AS_002", name: "Budget Families", audience: "28–50 · family bookers", impressions: 1288910, clicks: 24473, spend: 4208.73, planned: 6300, revenue: 21001.75, conversions: 51, under: 18, days: 22 },
  { id: "AS_003", name: "Business Travel", audience: "30–55 · corporate travel", impressions: 867540, clicks: 18423, spend: 8422.84, planned: 9450, revenue: 55324.75, conversions: 63, under: 0, days: 22 },
  { id: "AS_004", name: "Adventure Seekers", audience: "18–30 · adventure travel", impressions: 1448980, clicks: 29526, spend: 5098.12, planned: 5250, revenue: 7924.5, conversions: 48, under: 0, days: 22 },
  { id: "AS_005", name: "Honeymoon Couples", audience: "25–40 · luxury couples", impressions: 638040, clicks: 12510, spend: 7869.42, planned: 8400, revenue: 157095.75, conversions: 59, under: 6, days: 22 },
  { id: "AS_006", name: "Senior Leisure", audience: "55+ · leisure travellers", impressions: 1013310, clicks: 18588, spend: 6371.12, planned: 7350, revenue: 23513.25, conversions: 36, under: 3, days: 21 },
].map((item) => ({
  ...item,
  ctr: (item.clicks / item.impressions) * 100,
  roas: item.revenue / item.spend,
  cpa: item.spend / item.conversions,
  utilisation: item.spend / item.planned,
}));

const totals = campaigns.reduce(
  (result, item) => ({
    spend: result.spend + item.spend,
    revenue: result.revenue + item.revenue,
    impressions: result.impressions + item.impressions,
    clicks: result.clicks + item.clicks,
    conversions: result.conversions + item.conversions,
  }),
  { spend: 0, revenue: 0, impressions: 0, clicks: 0, conversions: 0 },
);

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
      <div><p class="eyebrow">EXECUTIVE REVIEW · 22-DAY SNAPSHOT</p><h1>Spend has a pacing problem,<br />not a revenue problem.</h1></div>
      <p class="lede">The lakehouse connects raw delivery, conversion value and pacing evidence before a campaign recommendation reaches an operator.</p>
    </section>
    <section class="metric-grid">
      ${metric("Attributed revenue", money(totals.revenue), `${totals.conversions} committed conversion rows`, "positive")}
      ${metric("Recorded spend", money(totals.spend), `${number(totals.impressions)} impressions`)}
      ${metric("Blended ROAS", `${blendedRoas.toFixed(2)}×`, "Revenue divided by recorded spend", "positive")}
      ${metric("Pacing watchlist", String(atRisk.length), "Campaigns under pace on >10 days", "warning")}
    </section>
    <section class="split">
      <article class="panel action-panel">
        <header><div><p class="eyebrow">OPERATOR QUEUE</p><h2>What needs a decision</h2></div><span class="count">${atRisk.length + 1}</span></header>
        ${atRisk.map((item, index) => `
          <button class="action-row" data-open-campaign="${item.id}">
            <span class="priority">${String(index + 1).padStart(2, "0")}</span>
            <span><strong>${item.name}</strong><small>Under pace on ${item.under} of ${item.days} observed days</small></span>
            <span class="action-value">${pct(item.utilisation * 100)}<small>utilisation</small></span>
          </button>`).join("")}
        <button class="action-row" data-route-link="campaigns">
          <span class="priority">03</span>
          <span><strong>Adventure Seekers</strong><small>Highest reach, but only ${campaigns[3].roas.toFixed(2)}× ROAS</small></span>
          <span class="action-value">${money(campaigns[3].spend)}<small>spend</small></span>
        </button>
      </article>
      <article class="panel">
        <header><div><p class="eyebrow">VALUE CONCENTRATION</p><h2>Revenue by audience</h2></div><span class="stamp">CSV snapshot</span></header>
        <div class="bars">
          ${[...campaigns].sort((a, b) => b.revenue - a.revenue).map((item) => `
            <div class="bar-row"><span>${item.name}</span><div><i style="width:${(item.revenue / Math.max(...campaigns.map((row) => row.revenue))) * 100}%"></i></div><strong>${money(item.revenue)}</strong></div>
          `).join("")}
        </div>
      </article>
    </section>
    <section class="decision-strip">
      <div><p class="eyebrow">REVIEWER VERDICT</p><strong>Reallocate only after pacing and attribution checks.</strong></div>
      <p>Honeymoon Couples contributes 46% of attributed value. Premium Travelers is materially under pace. The demo keeps those facts separate from the recommendation.</p>
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
      <div><p class="eyebrow">GOLD · CAMPAIGN PERFORMANCE</p><h1>Compare the six audience strategies.</h1></div>
      <p class="lede">Every value traces back to the committed performance and conversion fixtures.</p>
    </section>
    <section class="toolbar">
      <label><span>Search campaigns</span><input id="campaign-search" value="${escapeHtml(state.campaignQuery)}" placeholder="Audience or segment" /></label>
      <label><span>Order by</span><select id="campaign-sort"><option value="roas" ${state.sort === "roas" ? "selected" : ""}>ROAS</option><option value="spend" ${state.sort === "spend" ? "selected" : ""}>Spend</option><option value="conversions" ${state.sort === "conversions" ? "selected" : ""}>Conversions</option></select></label>
    </section>
    <section class="campaign-grid">
      ${rows.map((item) => `
        <article class="campaign-card">
          <header><span class="campaign-id">${item.id}</span><span class="health ${item.roas < 2 ? "risk" : ""}">${item.roas < 2 ? "Review" : "Healthy"}</span></header>
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
      <div><p class="eyebrow">DECISION SANDBOX</p><h1>Test a budget move before making it.</h1></div>
      <p class="lede">A transparent scenario, not an automated spend instruction.</p>
    </section>
    <section class="scenario-layout">
      <article class="panel control-panel">
        <p class="eyebrow">SCENARIO INPUT</p><h2>Recover Premium Travelers pacing</h2>
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
        <div class="outcome-number"><span>Guarded attributed value</span><strong>${money(guardedReturn)}</strong></div>
        <div class="outcome-number"><span>Assumption</span><strong>${(source.roas * 0.7).toFixed(2)}× ROAS</strong></div>
        <button class="primary-action" id="record-scenario">Record review scenario</button>
        <small>This action saves nothing to an ad platform.</small>
      </article>
    </section>
    <section class="panel assumption-table"><header><div><p class="eyebrow">WHY THIS IS NOT A FORECAST</p><h2>Assumption register</h2></div></header>
      <table><thead><tr><th>Input</th><th>Observed</th><th>Scenario treatment</th></tr></thead><tbody>
        <tr><td>ROAS</td><td>${source.roas.toFixed(2)}× in fixture</td><td>30% haircut</td></tr>
        <tr><td>Pacing</td><td>${source.under}/${source.days} days under pace</td><td>Budget capacity only</td></tr>
        <tr><td>Attribution</td><td>Mixed 7-day click / 1-day view</td><td>No incrementality claim</td></tr>
      </tbody></table>
    </section>`;
}

function qualityView() {
  const checks = [
    ["Campaign keys resolve", "131/131 performance rows", "Pass"],
    ["Pacing keys resolve", "126/126 pacing rows", "Pass"],
    ["Conversion keys resolve", "319/319 conversion rows", "Pass"],
    ["Spend is non-negative", "6 campaign aggregates", "Pass"],
    ["Under-pacing concentration", "AS_001 and AS_002", "Review"],
    ["Attribution comparability", "Two windows in source data", "Review"],
  ];
  return `
    <section class="page-head compact"><div><p class="eyebrow">DETERMINISTIC REVIEW AGENTS</p><h1>Quality before commentary.</h1></div><p class="lede">The public console distinguishes structural checks from analyst judgement.</p></section>
    <section class="quality-summary"><div class="quality-score"><span>4 / 6</span><strong>checks pass</strong><small>Two issues need interpretation, not silent repair.</small></div>
      <div class="quality-copy"><p class="eyebrow">RELEASE STATE</p><h2>Usable with caveats</h2><p>The fixture is internally complete enough for the demo. Mixed attribution windows and persistent under-pacing still prevent a clean “reallocate” recommendation.</p></div></section>
    <section class="panel check-list">${checks.map(([name, evidence, result]) => `<div class="check-row"><span class="check-mark ${result === "Review" ? "review" : ""}">${result === "Review" ? "!" : "✓"}</span><span><strong>${name}</strong><small>${evidence}</small></span><b class="${result.toLowerCase()}">${result}</b></div>`).join("")}</section>`;
}

function lineageView() {
  const layers = [
    ["RAW", "4 CSV inputs", "Audience, pacing, conversions and delivery"],
    ["BRONZE", "Typed landing tables", "Immutable source-shaped records"],
    ["SILVER", "Clean campaign facts", "Dates, joins and quality flags"],
    ["GOLD", "Decision features", "ROAS, CPA, pacing and model inputs"],
    ["MODEL", "XGBoost artefacts", "Performance and under-pacing risk"],
    ["REVIEW", "Streamlit + agents", "Evidence before recommendation"],
  ];
  return `
    <section class="page-head compact"><div><p class="eyebrow">SYSTEM SHAPE</p><h1>Trace a recommendation to its row.</h1></div><p class="lede">The runnable Python project rebuilds every layer locally; this browser edition exposes the review surface.</p></section>
    <section class="lineage">${layers.map(([tag, title, description], index) => `<article><span>${tag}</span><div><strong>${title}</strong><p>${description}</p></div>${index < layers.length - 1 ? `<i aria-hidden="true">→</i>` : ""}</article>`).join("")}</section>
    <section class="split lineage-detail"><article class="panel"><p class="eyebrow">INPUT CONTRACT</p><h2>Committed evidence</h2><ul><li>131 delivery rows</li><li>126 pacing rows</li><li>319 conversion rows</li><li>6 audience definitions</li></ul></article><article class="panel"><p class="eyebrow">PRODUCTION BOUNDARY</p><h2>What Pages does not run</h2><ul><li>DuckDB rebuilds</li><li>XGBoost training</li><li>Live ad-platform ingestion</li><li>Streamlit server sessions</li></ul><a class="inline-link" href="https://github.com/MatthewPaver/marketing-ml-lakehouse#canonical-setup">Run the engine locally →</a></article></section>`;
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
render();
