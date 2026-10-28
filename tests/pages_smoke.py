import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / "docs" / "assets" / "console.png"
EVIDENCE = json.loads((ROOT / "docs" / "generated" / "evidence.json").read_text())
QUALITY_CHECKS = len(EVIDENCE["quality"]["checks"])
SOURCES = len(EVIDENCE["contracts"]["sources"])


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto("http://127.0.0.1:8766", wait_until="networkidle")
    assert "Marketing ML Lakehouse" in page.title()
    assert page.get_by_role("heading", level=1, name="Six sample campaigns: spend and data checks").is_visible()
    page.screenshot(path=str(SCREENSHOT), animations="disabled")

    page.get_by_role("button", name="Campaigns", exact=True).click()
    assert page.locator(".campaign-row").count() == 6
    page.get_by_role("button", name="Budget scenario").click()
    assert page.get_by_text("Try a budget change before you make it").is_visible()
    page.get_by_role("button", name="Data checks", exact=True).click()
    assert page.locator(".check-row").count() == QUALITY_CHECKS
    assert page.locator(".source-row").count() == SOURCES
    page.get_by_role("button", name="Pipeline").click()
    assert page.locator(".lineage article").count() == 6
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")

    mobile = browser.new_page(viewport={"width": 375, "height": 812})
    mobile.goto("http://127.0.0.1:8766", wait_until="networkidle")
    mobile.get_by_role("button", name="Campaigns", exact=True).click()
    assert mobile.locator(".campaign-row").count() == 6
    assert mobile.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    browser.close()
