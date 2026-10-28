import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const html = readFileSync(new URL("../docs/index.html", import.meta.url), "utf8");
const app = readFileSync(new URL("../docs/app.js", import.meta.url), "utf8");
const evidence = JSON.parse(readFileSync(new URL("../docs/generated/evidence.json", import.meta.url), "utf8"));

test("Pages console exposes the complete evidence workflow", () => {
  for (const route of ["overview", "campaigns", "pacing", "quality", "lineage"]) {
    assert.match(html, new RegExp(`data-route="${route}"`));
  }
  for (const label of ["Overview", "Campaigns", "Budget scenario", "Data checks", "Pipeline"]) {
    assert.match(html, new RegExp(`>${label}<`));
  }
  assert.match(html, /<title>Marketing ML Lakehouse/);
  assert.doesNotMatch(html, /Marketing Lakehouse</);
  assert.ok(evidence.contracts.sources.every((source) => source.rows > 0));
  assert.ok(evidence.campaigns.length > 0);
  assert.match(app, /No ad account is connected/);
  assert.match(app, /return on ad spend \(ROAS\)/i);
});

test("Pages uses one accent and trackless revenue bars", () => {
  const css = readFileSync(new URL("../docs/styles.css", import.meta.url), "utf8");
  assert.doesNotMatch(css, /--lime/);
  assert.match(css, /\.button\.primary \{[^}]*background: var\(--accent\)/);
  assert.doesNotMatch(css, /\.bar-track \{[^}]*background/);
  assert.match(css, /\.bar-track i \{[^}]*border-radius: 0 4px 4px 0/);
  assert.match(css, /\.result-list dd \{[^}]*font-variant-numeric: tabular-nums/);
  assert.doesNotMatch(css, /\.result-list dd \{[^}]*Georgia/);
  assert.match(app, /<h1>Six sample campaigns: spend and data checks<\/h1>/);
});

test("Pages reads generated, versioned pipeline evidence", () => {
  assert.match(app, /generated\/evidence\.json/);
  assert.equal(evidence.schema_version, 1);
  assert.equal(evidence.provenance.generator, "lakehouse.publish_pages");
  assert.ok(evidence.quality.checks.some((check) => check.name === "daily_key_unique"));
});
