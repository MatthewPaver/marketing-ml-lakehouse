from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / "docs" / "showcase.png"


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1200, "height": 675})
    page.goto("http://127.0.0.1:8766", wait_until="networkidle")
    assert "Evidence console" in page.title()
    assert page.get_by_text("Spend has a pacing problem").is_visible()
    page.screenshot(path=str(SCREENSHOT), animations="disabled")

    page.get_by_role("button", name="Campaigns").click()
    assert page.locator(".campaign-card").count() == 6
    page.get_by_role("button", name="Pacing lab").click()
    assert page.get_by_text("Test a budget move before making it.").is_visible()
    page.get_by_role("button", name="Data quality").click()
    assert page.locator(".check-row").count() == 6
    page.get_by_role("button", name="Lineage").click()
    assert page.locator(".lineage article").count() == 6
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto("http://127.0.0.1:8766", wait_until="networkidle")
    mobile.get_by_role("button", name="Campaigns").click()
    assert mobile.locator(".campaign-card").count() == 6
    assert mobile.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    browser.close()
