import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const html = readFileSync(new URL("../docs/index.html", import.meta.url), "utf8");
const app = readFileSync(new URL("../docs/app.js", import.meta.url), "utf8");

test("Pages console exposes the complete evidence workflow", () => {
  for (const label of ["Overview", "Campaigns", "Pacing lab", "Data quality", "Lineage"]) {
    assert.match(html, new RegExp(label));
  }
  assert.match(app, /131 delivery rows/);
  assert.match(app, /319 conversion rows/);
  assert.match(app, /No ad account is connected|No incrementality claim/);
});
