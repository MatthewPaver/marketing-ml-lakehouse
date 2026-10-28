---
name: Marketing ML Lakehouse demo
description: A quiet, readable console for six made-up travel campaigns, their data checks and a forecast test.
colors:
  ink: "#15221f"
  muted: "#5e6c67"
  line: "#ced8d2"
  line-soft: "#e1e8e3"
  surface: "#ffffff"
  canvas: "#f2f5f1"
  table-head: "#f7f9f7"
  accent: "#0c695e"
  accent-hover: "#09544b"
  accent-soft: "#d9eee8"
  focus-ring: "#6fbcb2"
  neutral-soft: "#e8ede9"
  attention-ink: "#8f3f19"
  attention-soft: "#ffeadc"
  danger-ink: "#a3241b"
  danger-soft: "#fde6e3"
typography:
  display:
    fontFamily: "Georgia, Times New Roman, serif"
    fontSize: "clamp(30px, 4vw, 42px)"
    fontWeight: 500
    lineHeight: 1.12
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Georgia, Times New Roman, serif"
    fontSize: "24px"
    fontWeight: 500
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "16px"
    fontWeight: 650
    lineHeight: 1.4
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  lede:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.6
  small:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.6
  caption:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.3
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "13px"
    fontWeight: 650
    lineHeight: 1.5
  figure:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    fontFeature: "tnum"
  code:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "max(12px, 0.92em)"
rounded:
  bar-end: "4px"
  skip-link: "6px"
  tab: "7px"
  control: "8px"
  container: "10px"
  pill: "999px"
spacing:
  hairline: "2px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  section: "48px"
  page-top: "48px"
  page-bottom: "72px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    height: "44px"
    padding: "0 16px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    height: "44px"
    padding: "0 16px"
  nav-tab:
    textColor: "{colors.muted}"
    rounded: "{rounded.tab}"
    height: "44px"
    padding: "0 13px"
  nav-tab-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
  pill-neutral:
    backgroundColor: "{colors.neutral-soft}"
    textColor: "{colors.muted}"
    rounded: "{rounded.pill}"
    padding: "2px 9px"
  pill-success:
    backgroundColor: "{colors.accent-soft}"
    textColor: "{colors.accent}"
  pill-attention:
    backgroundColor: "{colors.attention-soft}"
    textColor: "{colors.attention-ink}"
  pill-danger:
    backgroundColor: "{colors.danger-soft}"
    textColor: "{colors.danger-ink}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    height: "46px"
    padding: "0 14px"
  revenue-bar:
    backgroundColor: "{colors.accent}"
    height: "10px"
---

# Design System: Marketing ML Lakehouse demo

This is the implementation contract for the GitHub Pages console in `docs/` (`index.html`, `app.js`, `styles.css`). It is derived from the shipped CSS. When the CSS and this file disagree, fix one of them in the same change.

## Overview

The console shows made-up campaign data to marketing analysts and data engineers. Pages are read top to bottom: a heading, a short lede, one primary action, then plain sections separated by space and thin rules. Numbers sit in sentences, simple tables or one bar list, never in KPI tiles.

**Key characteristics:**
- Pale green-grey canvas, white tables, dark green-black ink.
- One accent, teal, for every action and the data mark.
- Georgia for page and section headings; the system sans for everything else, including all figures.
- Flat surfaces. Depth comes from a tone step (canvas to white), not shadows.

## Colors

A calm, low-saturation palette with one teal accent.

### Primary
- **Teal** (`#0c695e`, `--accent`): primary buttons, links, text links, the selected range thumb (`accent-color`) and the revenue bars. White text on teal is 6.5:1.
- **Deep teal** (`#09544b`, `--accent-hover`): hover state of primary buttons.
- **Teal wash** (`#d9eee8`, `--accent-soft`): background of the success pill only.

### Neutral
- **Ink** (`#15221f`, `--ink`): body text, headings, figures, the active page tab background.
- **Muted** (`#5e6c67`, `--muted`): ledes, notes, table headers, secondary cell text. 5.0:1 on canvas.
- **Canvas** (`#f2f5f1`, `--canvas`): page and top bar background.
- **Surface** (`#ffffff`, `--surface`): tables, secondary buttons, inputs.
- **Table head** (`#f7f9f7`): table header row only.
- **Line** (`#ced8d2`, `--line`) and **soft line** (`#e1e8e3`, `--line-soft`): 1px rules between rows, around tables and controls.
- **Neutral wash** (`#e8ede9`, `--neutral-soft`): page-tab rail and the neutral pill.
- **Focus ring** (`#6fbcb2`): 3px outline, 2px offset, on every focusable element.

### Status tones
Four tones only: neutral, success, attention, danger. Each is a pill with a soft background and a dark text colour from the same hue (`--attention-ink` on `--attention-soft`, `--danger-ink` on `--danger-soft`). Status colour is never the only signal: the pill always carries a word ("Pass", "Warning", "Fail").

### Named rules
**The one accent rule.** Teal is the only accent. There is no second highlight colour for calls to action; the earlier lime button colour was removed.
**The blue-means-clickable rule.** Accent-coloured text is always clickable. Data values are set in ink.

## Typography

**Heading font:** Georgia (with Times New Roman, serif).
**Body and figure font:** the system UI sans (`system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`).
**Code font:** `ui-monospace`, only for IDs, file names and hashes people copy.

### Hierarchy
- **Display** (500, `clamp(30px, 4vw, 42px)`, 1.12, `text-wrap: balance`): the one `h1` per page.
- **Headline** (500, 24px, 1.25): section `h2`.
- **Title** (650, 16px, 1.4): `h3` inside lists such as "What to look at next".
- **Lede** (400, 17px, 1.6; 16px under 720px): the sentence under the `h1`, muted.
- **Body** (400, 15px, 1.6): paragraphs, max 760px wide.
- **Small** (400, 14px): table cells, bar labels, notes, page tabs and the range labels.
- **Caption** (400, 12px): the wordmark subtitle and file hashes. The smallest size on the page.
- **Label** (650, 13px): table headers, field labels, pills.
- **Figure** (600, 22px, tabular figures): budget estimate values and the slider value.
- **Code** (`max(12px, 0.92em)`): never below 12px.

### Named rules
**The two-line rule.** Every heading fits in two lines at 1440px and at 390px. Shorten the copy first, then adjust size.
**The figures-in-sans rule.** Every number is set in the sans with `font-variant-numeric: tabular-nums` (tables, bar values, budget figures). Georgia’s old-style numerals are not used for figures.
**The plain-statement rule.** Headings are one plain statement in sentence case, with no full stop and no two-beat "X. Y." cadence.

## Layout

- Content column: `min(1120px, 100% - 48px)`, centred; `100% - 32px` under 720px.
- Page padding: 48px top, 72px bottom (32px top on phones).
- Prose width: 760px for ledes and paragraphs; lists and details up to 900px.
- Sections: 48px apart (40px on phones). More space above a heading than below it.
- Overview at 1000px and wider: two columns (1.1fr / 0.9fr, 64px gap): summary and revenue bars on the left, "What to look at next" on the right.
- Budget scenario: two columns (1.2fr / 0.8fr, 48px gap), one column under 960px.
- Top bar: wordmark, page tabs, source link. Sticky on desktop with a solid canvas background; static on phones.
- Breakpoints: 960px (top bar wraps to two rows, scenario stacks) and 720px (phone layout).
- Phones: page tabs sit in one horizontally scrollable row. A 48px fade on the edge with more tabs (classes `has-more-start` / `has-more-end`, set by `app.js`) shows that the row scrolls, and the active tab is scrolled into view on each route change. Tables become one block per row, with each value labelled by `data-label`.
- No horizontal page scroll at 320px or wider.

## Elevation & Depth

Flat. There are no shadows, glows or blurred glass. Depth is one tone step: white tables and controls on the pale canvas, separated by 1px rules.

## Shapes

- 8px: buttons, inputs, selects.
- 10px: table wrapper and the page-tab rail.
- 7px: individual page tabs.
- 999px: status pills only.
- 50%: the circular "ML" wordmark badge only.
- Revenue bars: a square baseline end and a 4px rounded data end (`border-radius: 0 4px 4px 0`). There is no grey background track.

## Components

- **Primary button** (`.button.primary`): teal fill, white text, 700 weight, 44px tall. At most one per screen section. Hover darkens to deep teal.
- **Secondary button** (`.button`): white fill, ink text, 1px line border. Hover turns the border ink.
- **Text link** (`.text-link`, `.inline-link`): teal, underlined with a 3px offset, 650 weight; 44px tap target where it stands alone.
- **Page tabs** (`.nav-button`): muted text on the neutral rail; the active tab is ink with white text and `aria-current="page"`.
- **Status pill** (`.pill`): 13px, 650 weight, one of the four tones, always with a word.
- **Data table** (`.data-table`): white, 1px soft rules, 13px muted header row; numeric columns right-aligned with tabular figures; stacked rows under 720px.
- **Revenue bars** (`.bars`): one row per campaign: label, a teal bar scaled to the largest value, and the value in ink with tabular figures. No track, no gridlines, no legend.
- **Budget slider**: native range input in teal (`accent-color`), with the current value as a 22px figure and "No change" / "+40%" end labels.
- **Disclosures** (`details`/`summary`): teal summary text; used for raw details such as hashes and warnings.
- **Form fields**: label above the field (13px, 650); every input has `name`; the search input has `autocomplete="off"` and a placeholder ending in "…".
- **Focus**: 3px `#6fbcb2` outline with 2px offset on buttons, links, inputs, selects and summaries.

### Motion
- Pressed buttons, tabs and text links scale to 0.98 over 100ms (`cubic-bezier(.16, 1, .3, 1)`), with colour changes over 120ms.
- Under `prefers-reduced-motion: reduce` there is no press scale, no transitions and no smooth scrolling.
- No entrance animations, scroll effects or animated charts.

## Do's and Don'ts

**Do**
- Use real figures from `docs/generated/evidence.json`; say plainly that the data is made up.
- Put each number in a sentence, a table or the one bar list.
- Keep the lede to two or three lines on a phone; put background (what the pipeline is, who it is for) in "About this data".
- Use typographic apostrophes (’) in visible copy.
- Say each limit once where it applies; the general data limits live in "About this data".

**Don't**
- KPI tiles or stat-card rows.
- A second accent colour, gradients, glass or backdrop blur, shadows on resting surfaces.
- Grey tracks behind bars (they read as progress meters), or bars rounded at both ends.
- Old-style or proportional figures in columns of numbers.
- Uppercase eyebrows or small labels above headings, index numbering (01 / 02), icon tiles.
- Monospace outside IDs, file names and hashes.
- Copy under 12px.
