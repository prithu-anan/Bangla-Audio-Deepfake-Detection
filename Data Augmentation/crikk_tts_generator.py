from playwright.sync_api import sync_playwright
import os
import csv
import json
import time
from datetime import datetime

# Use the same user data directory as store-google-creds.py
PLAYWRIGHT_USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "playwright-chrome-data")

# Path to the metadata CSV file
METADATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "metadata.csv")

# Path to checkpoint file for tracking progress
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "tts_checkpoint.json")

# Voice to use for all TTS generation
VOICE = "Nabanita"


def load_sentences_from_csv(csv_path: str) -> list[str]:
    """Load Bangla sentences from the metadata CSV file."""
    sentences = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) >= 2:
                # The text is in the second column, strip leading/trailing whitespace
                text = row[1].strip()
                if text:
                    sentences.append(text)
    return sentences


def load_checkpoint() -> dict:
    """Load checkpoint data from file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "last_completed_index": -1,
        "successful": [],
        "failed": [],
        "start_time": None,
        "last_update": None
    }


def save_checkpoint(checkpoint: dict):
    """Save checkpoint data to file."""
    checkpoint["last_update"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def wait_for_overlay_to_disappear(page, max_wait_seconds=120):
    """Wait for any blocking overlay to disappear before proceeding."""
    print("Waiting for any loading overlay to disappear...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        # Check if blockUI overlay exists and is visible
        overlay = page.locator("div.blockUI.blockOverlay")
        if overlay.count() == 0:
            return True
        
        # Check if it's visible
        try:
            if not overlay.first.is_visible():
                return True
        except:
            return True
        
        # Wait a bit before checking again
        page.wait_for_timeout(500)
    
    print(f"Warning: Overlay still present after {max_wait_seconds}s, proceeding anyway...")
    return False


def generate_tts(page, title: str, text: str):
    """Generate a single TTS audio on crikk.com using Nabanita voice"""
    
    print(f"\n--- Generating TTS: '{title}' ---")
    
    # Wait for any overlay to disappear before clicking
    wait_for_overlay_to_disappear(page)
    
    # Click on "Text To Speech" button - no timeout, wait indefinitely
    print("Clicking 'Text To Speech' button...")
    tts_button = page.locator("button:has-text('Text To Speech')")
    tts_button.wait_for(state="visible", timeout=0)  # Wait indefinitely for button to be visible
    
    # Wait for button to be clickable (no overlay blocking)
    while True:
        try:
            tts_button.click(timeout=5000)
            break
        except Exception as e:
            if "intercepts pointer events" in str(e):
                print("  Waiting for overlay to clear...")
                page.wait_for_timeout(1000)
            else:
                raise e
    
    # Wait for the modal to appear - no timeout
    print("Waiting for modal to appear...")
    page.wait_for_selector("#fileName", timeout=0)
    time.sleep(0.5)  # Small delay for modal animation
    
    # Fill in the Title (use specific ID: fileName)
    print(f"Entering title: {title}")
    title_input = page.locator("#fileName")
    title_input.fill(title)
    
    # Fill in the Text
    print(f"Entering text: {text[:50]}...")
    text_area = page.locator("textarea[placeholder='Enter text here...']")
    text_area.fill(text)
    
    # Change language from English(US) to Bangla (Bangladesh)
    print("Selecting Bangla (Bangladesh) language...")
    language_select = page.locator("select").first
    language_select.select_option(value="bn-BD")
    page.wait_for_timeout(500)
    
    # Nabanita is the default voice, no need to change it
    print(f"Using default voice: {VOICE}")
    
    # Click "Create Audio" button
    print("Clicking 'Create Audio'...")
    create_button = page.get_by_role("button", name="Create Audio")
    create_button.click()
    
    # Wait for the modal to close - no timeout, wait indefinitely
    print("Waiting for audio to be created...")
    page.wait_for_selector("#fileName", state="hidden", timeout=0)
    print(f"✓ TTS '{title}' creation initiated successfully!")
    
    # Give some time for the page to update
    page.wait_for_timeout(2000)


def login_with_google(page):
    """Handle Google login on crikk.com"""
    print("\n--- Logging in with Google ---")
    
    # Click on "Google, Help Me Out!" button
    google_button = page.get_by_text("Google, Help Me Out!")
    if google_button.count() > 0:
        print("Clicking Google login button...")
        google_button.click()
        
        # Wait for Google OAuth or redirect
        print("Waiting for Google authentication...")
        
        # Wait for either dashboard to load or for manual intervention
        try:
            page.wait_for_selector("text=My Projects", timeout=120_000)
            print("✓ Login successful! Dashboard loaded.")
            return True
        except:
            print("⚠️ Login taking longer than expected.")
            print("If a Google popup appeared, please complete the login.")
            print("Press Enter once you're on the dashboard...")
            input()
            return True
    else:
        print("Google login button not found")
        return False


def navigate_to_project(page, project_name="Deepfake"):
    """Navigate to a specific project from the dashboard"""
    print(f"\n--- Navigating to project: {project_name} ---")
    
    # Wait for projects to load
    page.wait_for_timeout(2000)
    
    # Look for the project card/link
    project_selectors = [
        f"text={project_name}",
        f"div:has-text('{project_name}')",
        f"[class*='project']:has-text('{project_name}')",
    ]
    
    for selector in project_selectors:
        project_element = page.locator(selector).first
        if project_element.count() > 0:
            print(f"Found project '{project_name}', clicking...")
            project_element.click()
            page.wait_for_timeout(2000)
            
            # Verify we're in the project (should see Text To Speech button)
            if page.get_by_text("Text To Speech").count() > 0:
                print(f"✓ Successfully entered project '{project_name}'")
                return True
    
    print(f"⚠️ Could not find project '{project_name}'")
    print("Please click on the project manually, then press Enter...")
    input()
    return True


def main():
    # Load sentences from CSV
    print("Loading sentences from metadata.csv...")
    sentences = load_sentences_from_csv(METADATA_CSV_PATH)
    total_sentences = len(sentences)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Get indices that have been successfully processed
    successful_indices = set(f["index"] for f in checkpoint.get("successful", []))
    
    # Get failed indices that need to be retried (failed is just a list of indices)
    failed_indices = checkpoint.get("failed", [])
    
    # Build list of indices to process
    indices_to_process = []
    
    # First, add failed indices (retry them first)
    if failed_indices:
        indices_to_process.extend(failed_indices)
        print(f"\n🔄 Will RETRY {len(failed_indices)} previously failed sentences first")
    
    # Then, add remaining indices that haven't been processed yet
    start_index = checkpoint["last_completed_index"] + 1
    for idx in range(start_index, total_sentences):
        if idx not in successful_indices and idx not in failed_indices:
            indices_to_process.append(idx)
    
    print("=" * 70)
    print("Crikk TTS Generator - Bangla Text to Speech (Batch Mode)")
    print("=" * 70)
    print(f"\nLoaded {total_sentences} sentences from metadata.csv")
    print(f"Voice: {VOICE} (for all sentences)")
    print(f"\nProgress:")
    print(f"  - Already successful: {len(successful_indices)}")
    print(f"  - Previously failed (will retry): {len(failed_indices)}")
    print(f"  - Remaining to process: {len(indices_to_process)}")
    
    if failed_indices:
        print(f"\n📌 RETRYING FAILED INDICES FIRST: {failed_indices}")
    
    print("=" * 70)
    
    if total_sentences == 0:
        print("ERROR: No sentences found in metadata.csv!")
        return
    
    if len(indices_to_process) == 0:
        print("✓ All sentences have already been processed successfully!")
        print(f"  Total successful: {len(successful_indices)}")
        print("\nTo start fresh, delete the file: tts_checkpoint.json")
        return
    
    # Clear the failed list since we're retrying them
    checkpoint["failed"] = []
    save_checkpoint(checkpoint)
    
    with sync_playwright() as p:
        # Launch browser maximized/fullscreen
        context = p.chromium.launch_persistent_context(
            user_data_dir=PLAYWRIGHT_USER_DATA_DIR,
            channel="chrome",
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ],
            no_viewport=True,
            timeout=0,  # No timeout for browser launch
        )
        
        print("\nBrowser launched successfully!")
        
        page = context.new_page()
        
        # Set default timeout to 0 (no timeout) for all operations
        page.set_default_timeout(0)
        
        # Step 1: Go to login page
        login_url = "https://crikk.com/app/login"
        print(f"\nNavigating to: {login_url}")
        page.goto(login_url, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(2000)
        
        # Step 2: Check if already logged in (redirected to dashboard)
        if "login" in page.url.lower():
            login_with_google(page)
        else:
            print("✓ Already logged in!")
        
        # Step 3: Wait for dashboard to fully load
        print("\nWaiting for dashboard to load...")
        page.wait_for_timeout(5000)
        
        # Step 4: Navigate to the Deepfake Detection project
        navigate_to_project(page, "Deepfake")
        
        # Wait for project page to fully load
        page.wait_for_timeout(2000)
        
        # Step 5: Generate TTS for all indices in the list
        processed_count = 0
        try:
            for idx in indices_to_process:
                sentence = sentences[idx]
                
                # Create a title with "crikk" and index for easy identification
                title = f"crikk_{idx:04d}_{VOICE}"
                
                processed_count += 1
                print(f"\n{'='*70}")
                print(f"Processing {processed_count}/{len(indices_to_process)} (index: {idx})")
                print(f"{'='*70}")
                
                try:
                    generate_tts(
                        page=page,
                        title=title,
                        text=sentence
                    )
                    
                    # Mark as successful
                    checkpoint["successful"].append({
                        "index": idx,
                        "title": title,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update last_completed_index if this is beyond it
                    if idx > checkpoint["last_completed_index"]:
                        checkpoint["last_completed_index"] = idx
                    
                    save_checkpoint(checkpoint)
                    
                    print(f"✓ Successfully queued: {title}")
                    
                except Exception as e:
                    # Mark as failed (just store the index)
                    if idx not in checkpoint["failed"]:
                        checkpoint["failed"].append(idx)
                    save_checkpoint(checkpoint)
                    
                    print(f"✗ Failed: {title}")
                    print(f"  Error: {str(e)}")
                    
                    # Try to recover by closing any open modal
                    try:
                        cancel_btn = page.get_by_role("button", name="Cancel")
                        if cancel_btn.count() > 0:
                            cancel_btn.click()
                            page.wait_for_timeout(1000)
                    except:
                        pass
                
                # Small delay between generations
                page.wait_for_timeout(1000)
                
                # Progress update every 50 sentences
                if processed_count % 50 == 0:
                    print(f"\n{'#'*70}")
                    print(f"# PROGRESS: {processed_count}/{len(indices_to_process)} processed this session")
                    print(f"# Total successful: {len(checkpoint['successful'])}")
                    print(f"# Total failed: {len(checkpoint['failed'])}")
                    print(f"# Checkpoint saved to: {CHECKPOINT_FILE}")
                    print(f"{'#'*70}\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ INTERRUPTED BY USER!")
            print(f"Checkpoint saved. Resume by running the script again.")
            save_checkpoint(checkpoint)
        
        # Final summary
        print("\n" + "=" * 70)
        print("TTS GENERATION SESSION COMPLETE!")
        print("=" * 70)
        print(f"Total sentences: {total_sentences}")
        print(f"Processed this session: {processed_count}")
        print(f"Total successful: {len(checkpoint['successful'])}")
        print(f"Total failed: {len(checkpoint['failed'])}")
        print(f"\nCheckpoint file: {CHECKPOINT_FILE}")
        
        if checkpoint["failed"]:
            print(f"\nFailed indices: {checkpoint['failed']}")
            print("Run the script again to retry failed sentences.")
        else:
            print(f"\n✓ All sentences have been processed successfully!")
        
        print("=" * 70)
        print("\nThe audio files are being processed on crikk.com")
        print("You can see them in your project dashboard.")
        print("\nPress Enter to close the browser...")
        input()
        
        context.close()


if __name__ == "__main__":
    main()
