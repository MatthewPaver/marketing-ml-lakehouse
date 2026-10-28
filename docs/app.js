let evidence;
let campaigns = [];
let totals = {};

// Readable display names for the sample ad sets. Raw names and IDs stay searchable.
const CAMPAIGN_LABELS = {
  "Premium_Travelers_25-45": "Premium travellers 25–45",
  "Budget_Conscious_Families": "Budget-conscious families",
  "Business_Travel_Professionals": "Business travel professionals",
  "Young_Adventure_Seekers": "Young adventure seekers",
  "Luxury_Honeymoon_Couples": "Luxury honeymoon couples",
  "Senior_Leisure_Travelers": "Senior leisure travellers",
};

// Plain names for the four raw files checked by the input contract.
const SOURCE_INFO = {
  audience_segments: ["Audiences", "One row per campaign: target age, interests and daily budget"],
  budget_pacing: ["Daily budget plan", "Planned and actual spend per campaign per day, with a pacing status"],
  conversion_events: ["Conversions", "Bookings and other actions, with their value and attribution window"],
  meta_campaign_performance: ["Ad delivery", "Impressions, clicks and spend per campaign per day"],
};

// One-line meaning for each check on the final daily table.
const CHECK_INFO = {
  daily_key_unique: ["One row per campaign per day", () => "No campaign has two rows for the same day.", (value) => `${value} repeated`],
  spend_reconciles: ["Spend adds up", (check) => `Spend from the ad delivery file matches actual spend in the budget file, to within ${money(check.tolerance, 2)}.`, (value) => money(value, 2)],
  conversion_revenue_reconciles: ["Booking revenue adds up", (check) => `Revenue in the final table matches completed bookings in the conversions file, to within ${money(check.tolerance, 2)}.`, (value) => money(value, 2)],
};

const LEAKAGE_INFO = {
  target_date_after_as_of_date: "Every predicted day comes after the day the prediction is made",
  target_columns_excluded: "The answers are not used as inputs",
  dates_disjoint: "Training and test dates do not overlap",
};

const STATUS = {
  pass: { label: "Pass", tone: "success" },
  warn: { label: "Warning", tone: "attention" },
  fail: { label: "Fail", tone: "danger" },
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function readableName(raw) {
  if (CAMPAIGN_LABELS[raw]) return CAMPAIGN_LABELS[raw];
  const text = String(raw).replaceAll("_", " ").replace(/(\d)-(\d)/g, "$1–$2").toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function prepareEvidence(payload) {
  evidence = payload;
  campaigns = payload.campaigns.map((item) => ({
    ...item,
    label: readableName(item.name),
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
const money = (value, digits = 0) => new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const number = (value) => new Intl.NumberFormat("en-GB", { notation: value > 999999 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
const pct = (value) => `${value.toFixed(1)}%`;
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function formatDate(iso) {
  const [year, month, day] = String(iso).slice(0, 10).split("-").map(Number);
  return `${day} ${MONTHS[month - 1]} ${year}`;
}

function formatRange(start, end) {
  const [startYear, startMonth, startDay] = String(start).slice(0, 10).split("-").map(Number);
  const [endYear, endMonth] = String(end).slice(0, 10).split("-").map(Number);
  if (startYear === endYear && startMonth === endMonth) return `${startDay} to ${formatDate(end)}`;
  return `${formatDate(start)} to ${formatDate(end)}`;
}

const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const inWords = (value) => WORDS[value] ?? String(value);

// Lets long file paths wrap at slashes and underscores; copied text is unchanged.
const pathHtml = (path) => escapeHtml(path).replace(/([/_])/g, "$1<wbr>");

function pill(tone, text) {
  return `<span class="pill ${tone}">${text}</span>`;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 2400);
}

function sourceName(source) {
  return (SOURCE_INFO[source.name] || [readableName(source.name)])[0];
}

function keyLabel(key = []) {
  const joined = key.join(",");
  if (joined === "date,ad_set_id" || joined === "ad_set_id,date") return "campaign and day";
  if (joined === "ad_set_id") return "campaign ID";
  if (joined === "conversion_id") return "conversion ID";
  return key.map((column) => `<code>${escapeHtml(column)}</code>`).join(" and ");
}

function describeIssue(code, source) {
  const [kind, first, second] = String(code).split(":");
  switch (kind) {
    case "duplicate_key_rows": return `${first} ${first === "1" ? "row repeats" : "rows repeat"} the same ${keyLabel(source.key)}`;
    case "null_required": return `${second} ${second === "1" ? "row has" : "rows have"} a blank <code>${escapeHtml(first)}</code>`;
    case "missing_column": return `Column <code>${escapeHtml(first)}</code> is missing`;
    case "duplicate_column": return `Column <code>${escapeHtml(first)}</code> appears twice`;
    case "malformed_row": return `Row ${escapeHtml(first)} has the wrong number of fields`;
    case "empty_source": return "The file has no rows";
    case "file_missing": return "The file was not found";
    default: return escapeHtml(code);
  }
}

function sourceIssues(source) {
  return [...(source.fatal || []), ...(source.warnings || [])];
}

function checkSummary() {
  const statuses = [
    ...evidence.contracts.sources.map((source) => source.status),
    ...evidence.quality.checks.map((check) => check.status),
  ];
  const count = (status) => statuses.filter((item) => item === status).length;
  return { total: statuses.length, pass: count("pass"), warn: count("warn"), fail: count("fail") };
}

function forecastResult() {
  const model = evidence.models?.bookings;
  if (!model) return null;
  const m = model.metrics;
  const overlap = m.mae_95pct_bootstrap[0] <= m.baseline_mae_95pct_bootstrap[1] && m.baseline_mae_95pct_bootstrap[0] <= m.mae_95pct_bootstrap[1];
  return { model, m, overlap };
}

function overview() {
  const blendedRoas = totals.revenue / totals.spend;
  const atRisk = campaigns.filter((item) => item.under > 10);
  const lowest = [...campaigns].sort((a, b) => a.roas - b.roas)[0];
  const mostShown = [...campaigns].sort((a, b) => b.impressions - a.impressions)[0];
  const byRevenue = [...campaigns].sort((a, b) => b.revenue - a.revenue);
  const topRevenue = byRevenue[0].revenue;
  const days = campaigns[0].days;
  const checks = checkSummary();
  const flagged = evidence.contracts.sources.filter((source) => sourceIssues(source).length);
  const forecast = forecastResult();
  return `
    <section class="page-head">
      <h1>Spend, revenue and data checks for six sample campaigns</h1>
      <p class="lede">Marketing ML Lakehouse is a Python pipeline. It loads four campaign CSV files into a local database, checks them, and tests a forecast of next-day bookings. This page shows its output for six made-up travel campaigns over ${days} days in August 2024. It is for marketing analysts and data engineers.</p>
      <div class="actions">
        <button class="button primary" data-route-link="quality">See the data checks</button>
        <button class="button" data-route-link="campaigns">Compare the campaigns</button>
      </div>
    </section>

    <section class="section">
      <h2>Summary</h2>
      <p>Over ${days} days the ${inWords(campaigns.length)} campaigns spent ${money(totals.spend)}. Their ads were shown ${number(totals.impressions)} times, and the sample data credits them with ${totals.conversions} bookings worth ${money(totals.revenue)}.</p>
      <p>That is a return on ad spend (ROAS) of ${blendedRoas.toFixed(2)}×. Return on ad spend is revenue divided by spend, so each £1 spent was matched by £${blendedRoas.toFixed(2)} of booking revenue.</p>
    </section>

    <section class="section">
      <h2>Booking revenue by campaign</h2>
      <ul class="bars">
        ${byRevenue.map((item) => `
          <li class="bar-row"><span class="bar-name">${escapeHtml(item.label)}</span><span class="bar-track" aria-hidden="true"><i style="width:${(item.revenue / topRevenue) * 100}%"></i></span><strong>${money(item.revenue)}</strong></li>`).join("")}
      </ul>
    </section>

    <section class="section">
      <h2>What to look at next</h2>
      <ul class="next-steps">
        ${atRisk.length ? `<li>
          <div>
            <h3>Check the ${inWords(atRisk.length)} ${atRisk.length === 1 ? "campaign that was" : "campaigns that were"} under pace on most days</h3>
            <p>${atRisk.map((item) => `<button class="text-link" data-open-campaign="${escapeHtml(item.id)}">${escapeHtml(item.label)}</button> was under pace on ${item.under} of ${item.days} days and spent ${pct(item.utilisation * 100)} of its planned budget.`).join(" ")} Under pace means the budget file marks the day as spending behind plan.</p>
          </div>
          <button class="button" data-route-link="pacing">Try a budget change</button>
        </li>` : ""}
        <li>
          <div>
            <h3>Look at ${escapeHtml(lowest.label)}</h3>
            <p>It has the lowest return on ad spend of the six (${lowest.roas.toFixed(2)}× on ${money(lowest.spend)} of spend)${mostShown.id === lowest.id ? ", even though its ads were shown the most" : ""}.</p>
          </div>
          <button class="button" data-open-campaign="${escapeHtml(lowest.id)}">View this campaign</button>
        </li>
        <li>
          <div>
            ${flagged.length
              ? `<h3>${checks.warn + checks.fail} data ${checks.warn + checks.fail === 1 ? "check needs" : "checks need"} a look</h3>
                 <p>${checks.pass} of ${checks.total} checks pass. ${flagged.map((source) => `${sourceName(source)} file: ${sourceIssues(source).map((code) => describeIssue(code, source)).join("; ")}.`).join(" ")}</p>`
              : `<h3>All ${checks.total} data checks pass</h3><p>The input files and the final table passed every check on the last run.</p>`}
          </div>
          <button class="button" data-route-link="quality">See the data checks</button>
        </li>
        ${forecast ? `<li>
          <div>
            ${forecast.overlap
              ? `<h3>Do not rely on the bookings forecast yet</h3>
                 <p>On ${forecast.model.split.test_rows} held-back campaign-days, the forecast was not shown to beat a simple guess that tomorrow's bookings equal today's.</p>`
              : `<h3>Read the bookings forecast test</h3>
                 <p>On ${forecast.model.split.test_rows} held-back campaign-days, the forecast's error range sits below that of a simple guess that tomorrow's bookings equal today's.</p>`}
          </div>
          <button class="button" data-route-link="lineage" data-anchor="model-evidence">See the forecast test</button>
        </li>` : ""}
      </ul>
    </section>

    <section class="section">
      <h2>About this data</h2>
      <p>All figures come from synthetic sample data: four CSV files committed with the code, covering six travel campaigns over ${days} days in August 2024. No ad account is connected, and nothing on this site changes a budget. Revenue is the booking value the sample data credits to each campaign. It is not an estimate of sales the ads caused. The figures show that the pipeline works. They say nothing about a real advertiser.</p>
    </section>`;
}

function campaignsView() {
  const query = state.campaignQuery.trim().toLowerCase();
  const rows = campaigns
    .filter((item) => `${item.label} ${item.name} ${item.id}`.toLowerCase().includes(query))
    .sort((a, b) => state.sort === "roas" ? b.roas - a.roas : state.sort === "spend" ? b.spend - a.spend : b.conversions - a.conversions);
  return `
    <section class="page-head">
      <h1>Six sample campaigns</h1>
      <p class="lede">Each campaign targets one audience. Search by name or ID, or change the order.</p>
    </section>
    <section class="toolbar">
      <label><span>Search campaigns</span><input id="campaign-search" type="search" value="${escapeHtml(state.campaignQuery)}" placeholder="Name or ID, for example AS_001" autocomplete="off" /></label>
      <label><span>Order by</span><select id="campaign-sort"><option value="roas" ${state.sort === "roas" ? "selected" : ""}>Return on ad spend</option><option value="spend" ${state.sort === "spend" ? "selected" : ""}>Spend</option><option value="conversions" ${state.sort === "conversions" ? "selected" : ""}>Bookings</option></select></label>
    </section>
    ${query ? `<p class="filter-note">Showing ${rows.length} of ${campaigns.length} campaigns. <button class="text-link" id="clear-search">Show all campaigns</button></p>` : ""}
    ${rows.length ? `<div class="table-wrap">
      <table class="data-table stack-table">
        <caption class="visually-hidden">Results for each sample campaign</caption>
        <thead><tr><th scope="col">Campaign</th><th scope="col" class="num">Spend</th><th scope="col" class="num">Booking revenue</th><th scope="col" class="num">Return on ad spend</th><th scope="col" class="num">Bookings</th><th scope="col" class="num">Cost per booking</th><th scope="col" class="num">Click-through rate</th></tr></thead>
        <tbody>
          ${rows.map((item) => `
          <tr class="campaign-row">
            <th scope="row"><span class="cell-title">${escapeHtml(item.label)}</span><span class="cell-sub">ID <code>${escapeHtml(item.id)}</code></span></th>
            <td class="num" data-label="Spend">${money(item.spend)}</td>
            <td class="num" data-label="Booking revenue">${money(item.revenue)}</td>
            <td class="num" data-label="Return on ad spend"><span class="value-with-pill">${item.roas < 2 ? pill("attention", "Below 2×") : ""}${item.roas.toFixed(2)}×</span></td>
            <td class="num" data-label="Bookings">${item.conversions}</td>
            <td class="num" data-label="Cost per booking">${money(item.cpa)}</td>
            <td class="num" data-label="Click-through rate">${pct(item.ctr)}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>` : `<p class="empty">No campaigns match that search.</p>`}
    <p class="note">Return on ad spend is booking revenue divided by spend. Cost per booking is spend divided by bookings. Click-through rate is clicks divided by the number of times the ads were shown.</p>`;
}

function pacingView() {
  const shift = state.shift;
  const source = campaigns[0];
  const extra = source.planned * (shift / 100);
  const guardedReturn = extra * source.roas * 0.7;
  return `
    <section class="page-head">
      <h1>Try a budget change before you make it</h1>
      <p class="lede">${escapeHtml(source.label)} was under pace on ${source.under} of ${source.days} days: the budget file marks those days as spending behind plan. Move the slider to see what a larger planned budget might bring in, using cautious assumptions. Nothing here changes a real budget.</p>
    </section>
    <section class="scenario-layout">
      <div class="scenario-control">
        <h2>${escapeHtml(source.label)}</h2>
        <label class="range-label" for="shift"><span>Increase the planned budget by</span><strong>+${shift}%</strong></label>
        <input id="shift" type="range" min="0" max="40" step="5" value="${shift}" aria-valuetext="${shift} percent" />
        <div class="range-scale" aria-hidden="true"><span>No change</span><span>+40%</span></div>
        <p class="note">Its planned budget over the ${source.days} days was ${money(source.planned)}.</p>
      </div>
      <div class="scenario-result">
        <h2>Estimate</h2>
        <dl class="result-list">
          <div><dt>Extra planned spend</dt><dd>${money(extra)}</dd></div>
          <div><dt>Booking revenue from the extra spend</dt><dd>${money(guardedReturn)}</dd></div>
          <div><dt>Return on ad spend used</dt><dd>${(source.roas * 0.7).toFixed(2)}×</dd></div>
        </dl>
        <button class="button primary" id="record-scenario">Mark this scenario as reviewed</button>
        <p class="note">Nothing is saved or sent anywhere.</p>
      </div>
    </section>
    <section class="section">
      <h2>Assumptions</h2>
      <p>This is simple arithmetic on the sample data, not a forecast.</p>
      <div class="table-wrap">
        <table class="data-table stack-table">
          <thead><tr><th scope="col">Input</th><th scope="col">In the sample data</th><th scope="col">How the estimate uses it</th></tr></thead>
          <tbody>
            <tr><th scope="row">Return on ad spend</th><td data-label="In the sample data">${source.roas.toFixed(2)}×</td><td data-label="How the estimate uses it">Cut by 30% to ${(source.roas * 0.7).toFixed(2)}×, to stay cautious</td></tr>
            <tr><th scope="row">Pacing</th><td data-label="In the sample data">Under pace on ${source.under} of ${source.days} days</td><td data-label="How the estimate uses it">Taken only as room to spend more</td></tr>
            <tr><th scope="row">Attribution</th><td data-label="In the sample data">Mixed 7-day click and 1-day view</td><td data-label="How the estimate uses it">Used as recorded</td></tr>
          </tbody>
        </table>
      </div>
    </section>`;
}

function qualityView() {
  const summary = checkSummary();
  const contracts = evidence.contracts;
  const quality = evidence.quality;
  const flagged = contracts.sources.filter((source) => sourceIssues(source).length);
  const countPill = (tone, count, text) => pill(count ? tone : "neutral", `${count} ${text}`);
  return `
    <section class="page-head">
      <h1>Data checks</h1>
      <p class="lede">The pipeline checks the four input files before loading them, then checks the final daily table after building it. A failed check stops the build. A warning lets it continue but needs a look. These results come from the run on ${formatDate(quality.generated_at)}.</p>
      <p class="check-summary">${summary.total} checks: ${countPill("success", summary.pass, "pass")} ${countPill("attention", summary.warn, summary.warn === 1 ? "warning" : "warnings")} ${countPill("danger", summary.fail, "fail")}</p>
    </section>

    <section class="section">
      <h2>Input files</h2>
      <p>Checked before loading: the expected columns are there, no row is broken, required fields are filled in, and no key repeats.</p>
      <div class="table-wrap">
        <table class="data-table stack-table">
          <thead><tr><th scope="col">File</th><th scope="col">What it holds</th><th scope="col" class="num">Rows</th><th scope="col">Result</th></tr></thead>
          <tbody>
            ${contracts.sources.map((source) => {
              const status = STATUS[source.status] || STATUS.fail;
              const issues = sourceIssues(source);
              return `<tr class="source-row">
                <th scope="row"><span class="cell-title">${sourceName(source)}</span><span class="cell-sub"><code>${pathHtml(source.path)}</code></span></th>
                <td data-label="What it holds">${(SOURCE_INFO[source.name] || [, ""])[1]}</td>
                <td class="num" data-label="Rows">${source.rows ?? ""}</td>
                <td data-label="Result">${pill(status.tone, status.label)}<span class="cell-sub">${issues.length ? issues.map((code) => describeIssue(code, source)).join("; ") : "No issues"}</span></td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Final daily table</h2>
      <p>Checked after building one row per campaign per day (${quality.rows} rows).</p>
      <div class="table-wrap">
        <table class="data-table stack-table">
          <thead><tr><th scope="col">Check</th><th scope="col">What it means</th><th scope="col" class="num">Found</th><th scope="col" class="num">Expected</th><th scope="col">Result</th></tr></thead>
          <tbody>
            ${quality.checks.map((check) => {
              const [title, meaning, format] = CHECK_INFO[check.name] || [readableName(check.name), () => "", (value) => escapeHtml(value)];
              const status = STATUS[check.status] || STATUS.fail;
              return `<tr class="check-row">
                <th scope="row"><span class="cell-title">${title}</span></th>
                <td data-label="What it means">${meaning(check)}</td>
                <td class="num" data-label="Found">${format(check.observed)}</td>
                <td class="num" data-label="Expected">${format(check.expected)}</td>
                <td data-label="Result">${pill(status.tone, status.label)}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Where the data came from</h2>
      <p>Four synthetic CSV files in the repository's <code>data/raw/</code> folder. They are checked against version ${escapeHtml(contracts.contract_version)} of the file contract, which lists the columns and keys each file must have. Each file's SHA-256 fingerprint is recorded with the results.</p>
      <details>
        <summary>Show file fingerprints</summary>
        <ul class="fingerprints">${contracts.sources.map((source) => `<li><code>${escapeHtml(source.path)}</code><code class="hash">${escapeHtml(source.sha256 || "")}</code></li>`).join("")}</ul>
      </details>
    </section>

    <section class="section">
      <h2>What to do next</h2>
      ${flagged.length ? `<ul class="plain-list">${flagged.map((source) => `<li>${sourceIssues(source).map((code) => issueAdvice(code, source)).join(" ")}</li>`).join("")}</ul>`
        : `<p>Nothing to fix. Every check passed on the last run.</p>`}
    </section>`;
}

function issueAdvice(code, source) {
  const path = `<code>${pathHtml(source.path)}</code>`;
  if (code.startsWith("duplicate_key_rows") && source.name === "meta_campaign_performance") {
    return `Review the repeated rows in ${path}: ${describeIssue(code, source)}. When two rows share a campaign and day, the pipeline keeps the most complete one, so the totals above still add up. Fix the source file if the repeats are a mistake.`;
  }
  return `Review ${path}: ${describeIssue(code, source)}.`;
}

function lineageView() {
  const layers = [
    ["Raw files", "4 CSV files: audiences, daily budget plan, conversions and ad delivery"],
    ["Bronze tables", "Each file loaded into DuckDB as it arrives, unchanged"],
    ["Silver tables", "Cleaned copies: numbers typed, repeated rows and rows missing an ID removed"],
    ["Gold table", "One row per campaign per day, with return on ad spend, cost per booking, pacing and model inputs"],
    ["Models", "XGBoost forecasts of next-day bookings and next-day under-pacing"],
    ["Review", "A local Streamlit dashboard and rule-based review checks"],
  ];
  return `
    <section class="page-head">
      <h1>From raw CSV files to model output</h1>
      <p class="lede">The Python project rebuilds every step on your own computer with one command. This page shows the output of the last run.</p>
    </section>
    <section class="section">
      <h2>Steps</h2>
      <ol class="lineage">${layers.map(([title, description]) => `<li><article><strong>${title}</strong><p>${description}</p></article></li>`).join("")}</ol>
      <p><a class="inline-link" href="https://github.com/MatthewPaver/marketing-ml-lakehouse#quickstart">Run the pipeline on your own computer</a></p>
    </section>
    ${modelEvidence()}`;
}

function modelEvidence() {
  const forecast = forecastResult();
  if (!forecast) return "";
  const { model, m, overlap } = forecast;
  const split = model.split;
  const range = (pair) => `${pair[0].toFixed(2)} to ${pair[1].toFixed(2)}`;
  const skill = (m.skill_over_baseline * 100).toFixed(0);
  const leakage = Object.entries(model.leakage_checks || {});
  return `<section class="section" id="model-evidence" tabindex="-1">
    <h2>Next-day bookings forecast test</h2>
    <p>The model predicts each campaign's bookings for the next day, using only what was known at the end of the day before. It is compared with a simple guess: tomorrow's bookings equal today's. It learned from ${split.train_rows} campaign-days (${formatRange(split.train_start, split.train_end)}) and was tested on the next ${split.test_rows} (${formatRange(split.test_start, split.test_end)}), which it had not seen.</p>
    <div class="table-wrap">
      <table class="data-table stack-table">
        <thead><tr><th scope="col">Method</th><th scope="col" class="num">Average error, in bookings</th><th scope="col" class="num">Likely range (95%)</th></tr></thead>
        <tbody>
          <tr><th scope="row">XGBoost model</th><td class="num" data-label="Average error, in bookings">${m.mae.toFixed(3)}</td><td class="num" data-label="Likely range (95%)">${range(m.mae_95pct_bootstrap)}</td></tr>
          <tr><th scope="row">Simple guess: tomorrow equals today</th><td class="num" data-label="Average error, in bookings">${m.baseline_mae.toFixed(3)}</td><td class="num" data-label="Likely range (95%)">${range(m.baseline_mae_95pct_bootstrap)}</td></tr>
        </tbody>
      </table>
    </div>
    <p class="result-line">${overlap
      ? `${pill("attention", "Not shown to be better")} The model's average error is ${skill}% lower, but the two ranges overlap. With ${split.test_rows} test rows, this sample cannot tell the model apart from the simple guess.`
      : `${pill("success", "Better than the simple guess")} The model's average error is ${skill}% lower, and the two ranges do not overlap.`}</p>
    <details>
      <summary>Method details</summary>
      <ul class="plain-list">
        <li>Average error is the mean absolute error (MAE): the average gap between predicted and actual bookings. Lower is better.</li>
        <li>The likely range is a 95% bootstrap interval over the ${split.test_rows} test rows. The simple guess is also called a persistence baseline.</li>
        ${leakage.map(([name, passed]) => `<li>${pill(passed ? "success" : "danger", passed ? "Pass" : "Fail")} ${LEAKAGE_INFO[name] || escapeHtml(readableName(name))}</li>`).join("")}
      </ul>
    </details>
  </section>`;
}

function render() {
  const allowed = ["overview", "campaigns", "pacing", "quality", "lineage"];
  if (!allowed.includes(state.route)) state.route = "overview";
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.dataset.route === state.route;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  workspace.innerHTML = ({ overview, campaigns: campaignsView, pacing: pacingView, quality: qualityView, lineage: lineageView })[state.route]();
  bind();
}

function bind() {
  document.querySelectorAll("[data-route-link]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.routeLink, button.dataset.anchor)));
  document.querySelectorAll("[data-open-campaign]").forEach((button) => button.addEventListener("click", () => { state.campaignQuery = button.dataset.openCampaign; navigate("campaigns"); }));
  document.querySelector("#campaign-search")?.addEventListener("input", (event) => { state.campaignQuery = event.target.value; render(); document.querySelector("#campaign-search")?.focus(); });
  document.querySelector("#campaign-sort")?.addEventListener("change", (event) => { state.sort = event.target.value; render(); document.querySelector("#campaign-sort")?.focus(); });
  document.querySelector("#clear-search")?.addEventListener("click", () => { state.campaignQuery = ""; render(); document.querySelector("#campaign-search")?.focus(); });
  document.querySelector("#shift")?.addEventListener("input", (event) => { state.shift = Number(event.target.value); render(); document.querySelector("#shift")?.focus(); });
  document.querySelector("#record-scenario")?.addEventListener("click", () => showToast("Marked as reviewed for this visit only. Nothing was saved."));
}

function navigate(route, anchor) {
  state.route = route;
  history.replaceState(null, "", `#${route}`);
  render();
  const target = anchor && document.getElementById(anchor);
  if (target) {
    target.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
    target.focus({ preventScroll: true });
  } else {
    window.scrollTo({ top: 0, behavior: reduceMotion() ? "auto" : "smooth" });
  }
}

document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.route)));
window.addEventListener("hashchange", () => { state.route = location.hash.slice(1) || "overview"; render(); });
fetch("generated/evidence.json")
  .then((response) => { if (!response.ok) throw new Error(`Evidence load failed: ${response.status}`); return response.json(); })
  .then((payload) => {
    prepareEvidence(payload);
    const footerRun = document.querySelector("#footer-run");
    if (footerRun && payload.generated_at) footerRun.textContent = `Made-up sample data. Pipeline last run on ${formatDate(payload.generated_at)}`;
    render();
  })
  .catch((error) => { workspace.innerHTML = `<p class="empty">The pipeline results could not be loaded: ${escapeHtml(error.message)}</p>`; });
