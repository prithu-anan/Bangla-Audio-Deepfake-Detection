from playwright.sync_api import sync_playwright
import os

# Use a SEPARATE user data directory for Playwright (not Chrome's default)
PLAYWRIGHT_USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "playwright-chrome-data")

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir=PLAYWRIGHT_USER_DATA_DIR,
        channel="chrome",
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
        ],
        timeout=60_000,
    )

    print("Context created, getting page...")
    
    page = context.new_page()
    
    print("Navigating to page...")
    page.goto("https://crikk.com/app", wait_until="domcontentloaded", timeout=60_000)

    print("Page loaded")
    
    # Keep browser open for interaction - remove this line when done
    input("Press Enter to close the browser...")

    context.close()
