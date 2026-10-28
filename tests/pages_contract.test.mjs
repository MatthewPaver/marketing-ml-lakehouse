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

test("Pages reads generated, versioned pipeline evidence", () => {
  assert.match(app, /generated\/evidence\.json/);
  assert.equal(evidence.schema_version, 1);
  assert.equal(evidence.provenance.generator, "lakehouse.publish_pages");
  assert.ok(evidence.quality.checks.some((check) => check.name === "daily_key_unique"));
});
