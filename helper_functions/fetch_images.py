import os
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# === Load JSON with panoId and camera settings ===
with open("../../Geogusser_Learn_AI/data/json/slowenia/slowenia_bollards.json", "r", encoding="utf-8") as f:
    data = json.load(f)

coords = data["customCoordinates"]

# === Setup Chrome WebDriver ===
options = Options()
options.add_argument("--headless=new")  # modern headless mode (fewer bugs)
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)  # Selenium auto-manages ChromeDriver if up-to-date

# === Accept Google Maps cookie banner once ===
print("🔄 Opening Google Maps to accept cookies...")
driver.get("https://www.google.com/maps")
time.sleep(3)
try:
    # Try inside iframes first (common for cookie banners)
    found = False
    for iframe in driver.find_elements(By.TAG_NAME, "iframe"):
        driver.switch_to.frame(iframe)
        try:
            btn = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//button[contains(., "Accept") or contains(., "Alle akzeptieren")]')
                )
            )
            btn.click()
            print("Accepted cookies inside iframe.")
            found = True
            break
        except:
            driver.switch_to.default_content()

    if not found:
        driver.switch_to.default_content()
        # Try directly in main page (no iframe)
        btn = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//button[contains(., "Accept") or contains(., "Alle akzeptieren")]')
            )
        )
        btn.click()
        print("Accepted cookies on main page.")

except Exception:
    print("ℹ️ No cookie banner found or already accepted.")
# === Prepare output directory ===
output_folder = "data/images_unlabeled/slowenia/images"
os.makedirs(output_folder, exist_ok=True)

# === Loop through each germany_bollard panoId entry ===
entry = 1
for idx,   loc in enumerate(coords):

    pano_id = (loc.get("panoId") or loc.get("extra", {}).get("panoId"))
    if not pano_id:
        print(f"Skipping entry - no panoID {entry} (line: {entry+1}")
    heading = loc["heading"]
    pitch = loc["pitch"]
    zoom = loc["zoom"]

    # Approximate conversion from zoom to FOV
    fov = max(10, min(120, 90 - (zoom) * 15))
    fov-=30
    # Build the Google Maps Street View URL
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}&heading={heading}&pitch={pitch}&fov={fov}&hl=de"
    driver.get(url)
    time.sleep(1)

    # Save screenshot
    screenshot_path = os.path.join(output_folder, f"slowenia_bollard_{idx + 1}.png")
    driver.save_screenshot(screenshot_path)
    print(f"Saved screenshot: {screenshot_path}")

# === Clean up ===
driver.quit()
print("Done.")
