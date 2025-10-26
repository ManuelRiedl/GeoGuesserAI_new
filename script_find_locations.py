import csv
import math
import os
import random
import threading
import tensorflow as tf
import time
import warnings
import sys
from collections import Counter
import pytesseract
from langdetect import detect, DetectorFactory
import readchar
from rich.align import Align
from rich.text import Text

from script_UI import estimated_time_per_loc, metas_selected_from_user, start_screen_loop
from variables import germany_bollard_queries
from rich.rule import Rule

from Demos.SystemParametersInfo import new_w
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import subprocess
import os
from rich.console import Console, Group, RenderableType

import os
import time
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from ultralytics import YOLO
from variables import *
import json
import random
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from PIL import Image
import io
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
prog = None
class ProgressManager:
    def __init__(self, metas_to_fetch,driver):
        self.current_step = "Setup..."
        self.metas_to_fetch = metas_to_fetch
        self.driver = driver
        self.total_to_fetch = sum(metas_to_fetch.values())
        self.total_found = 0
        self.running = True
        self.pause_request = False
        self.pause_started_time = None
        self.paused = False
        self.pause_condition = threading.Condition()
        self.total_paused_time = 0
        self.cancel_request = False
        self.cancel = False
        self.last_key_pressed = None
        # Per-meta tracking
        self.meta_states = {
            meta: {
                "total": count,
                "checked": 0,
                "found": 0,
                "faulty": 0,
                "no_street_view": 0,
                "start_time": None,
                "time": None,
                "done": False,
                "paused_time_total": 0,
                "pause_started": None,
            }
            for meta, count in metas_to_fetch.items()
        }

        # Progress objects for each meta
        self.progress_bars = {}
        for meta, state in self.meta_states.items():
            self.progress_bars[meta] = Progress(
                BarColumn(bar_width=None),
                TextColumn("{task.completed}/{task.total}"),
                expand=True
            )
            self.progress_bars[meta].add_task("", total=state["total"], completed=0)

        self.start_time = time.time()
        os.system('cls' if os.name == 'nt' else 'clear')
        self.listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.listener_thread.start()



    def save_progress(self):
        global current_save_file
        state = {
            "save_file": current_save_file,
            "metas_to_fetch": self.metas_to_fetch,
            "meta_states": self.meta_states,
            "total_to_fetch": self.total_to_fetch,
            "total_found": self.total_found,
            "total_paused_time": self.total_paused_time,
            "paused": self.paused,
            "pause_request": self.pause_request,
            "cancel_request": self.cancel_request,
            "cancel": self.cancel,
            "current_step": self.current_step,
            "global_start_time": getattr(self, "start_time", None),
            "saved_timestamp": time.time(),
        }

        with open(SAVE_RESUME_FETCHING_FILE, "w") as f:
            json.dump(state, f, indent=2)



    def keyboard_listener(self):
        while self.running:
            key = readchar.readkey()
            if key == "p":
                self.toggle_pause()
            elif key == "c":
                self.cancel_request = True
            if self.last_key_pressed == "c" and key =="c":
                 self.cancel = True
                 self.render()
            elif self.last_key_pressed == "c" and key =="x":
                self.cancel_request = False
            self.last_key_pressed = key
            self.render()
    def cancel_fetching(self):
        #stops keyboard thread
        self.running = False
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
        except Exception:
            pass
        self.save_progress()
        start_screen_loop()

    def cancel_point(self):
        if self.cancel:
            self.cancel_fetching()
            return True
        return False
    def toggle_pause(self):
        with self.pause_condition:
            if not self.pause_request:
                self.pause_request = True
            else:
                if self.pause_started_time is not None:
                    pause_duration = time.time() - self.pause_started_time
                    self.total_paused_time += pause_duration
                    for state in self.meta_states.values():
                        if state["pause_started"]:
                            state["paused_time_total"] += time.time() - state["pause_started"]
                            state["pause_started"] = None
                    self.pause_started_time = None
                self.pause_request = False
                self.pause_condition.notify_all()

    def wait_if_paused(self):
        with self.pause_condition:
            if self.pause_request:
                self.pause_started_time = time.time()
                for state in self.meta_states.values():
                    if state["start_time"] and not state["done"]:
                        state["pause_started"] = time.time()
            while self.pause_request:
                self.set_step("Location Finder is[bold #CC241D] PAUSED [/bold #CC241D]- press p to continue")
                self.paused = True
                self.pause_condition.wait()
                self.paused = False
                self.set_step("Location Finder is[bold green] Running [/bold green]")


    def start_meta(self, meta):
        self.meta_states[meta]["start_time"] = time.time()
        self.meta_states[meta]["paused_time_total"] = 0
        self.meta_states[meta]["pause_started"] = None

    def stop_time(self, meta):
        self.meta_states[meta]["done"] = True

    def render(self):
        # Adjust total time: subtract total paused duration
        if self.paused:
            total_elapsed = self.pause_started_time - self.start_time
        else:
            total_elapsed = time.time() - self.start_time - self.total_paused_time
        total_pct = (self.total_found / self.total_to_fetch * 100) if self.total_to_fetch else 0
        total_estimated_sec = 0
        for meta, count in self.metas_to_fetch.items():
            total_estimated_sec += (estimated_time_per_loc.get(meta, 120) * count)

        if self.pause_request and not self.paused:
            paused_text = f"  --[bold green] PAUSE [/bold green] after current fetch"
        else:
            paused_text = ""
        if self.cancel_request:
            paused_text = f"  --[bold green] CANCEL[/bold green] request (Progress gets saved) - [bold #CC241D]c[/bold #CC241D] to confirm - [bold #CC241D]x[/bold #CC241D] to cancel "
        if self.cancel:
            paused_text = f"  --[bold #CC241D]CANCEL[/bold #CC241D] confirmed"
        header_panel = Group(
            Align.center(Text("c to cancel fetching (current progress gets saved)", style="italic #AAAAAA")),
            Panel(
            f"[bold green]Current step:[/bold green] {self.current_step}  \n"
            f"[bold #CC241D]Total:[/bold #CC241D] {self.total_found}/{self.total_to_fetch} ({total_pct:.1f}%)  \n"
            f"[bold #CC241D]Elapsed Time/Estimated Time: [/bold #CC241D]{self.format_time(total_elapsed)}/{self.format_time(total_estimated_sec)}{paused_text}",
            expand=True
        ))

        table = Table(expand=True, show_edge=False, box=None, padding=(0, 1))
        table.add_column("Meta", style="bold #78BCC4")
        table.add_column("Progress")
        table.add_column("Checked Locations", style="bold #CC241D")
        table.add_column("Faulty", style="bold #CC241D")
        table.add_column("No Streetview", style="bold #CC241D")
        table.add_column("Elapsed Time", style="bold green")
        table.add_column("Estimated Time", style="bold #78BCC4")

        for meta, state in self.meta_states.items():
            elapsed = self.get_meta_elapsed(meta)
            eta = self.compute_eta(meta, self.metas_to_fetch)
            self.progress_bars[meta].update(0, completed=state["found"])
            bar_render = self.progress_bars[meta].get_renderable()

            table.add_row(
                f"[#78BCC4]{meta}[/#78BCC4]",
                bar_render,
                str(state["checked"]),
                str(state["faulty"]),
                str(state["no_street_view"]),
                f"[bold green]{self.format_time(elapsed)}[/bold green]",
                f"[bold #CC241D]{eta}[/bold #CC241D]"
            )

        return Panel(Group(header_panel, Rule(style="bold grey"), table),
                     border_style="#00739C", padding=(1, 2),
                     title="[bold #CC241D]Location Finder[/bold #CC241D]")

    def get_meta_elapsed(self, meta):
        state = self.meta_states[meta]
        if state["start_time"] is None:
            return 0
        if state["done"] is True:
            return state["time"]
        # Subtract per-meta paused time
        paused_time = state["paused_time_total"]
        if state["pause_started"] is not None:
            paused_time += time.time() - state["pause_started"]
        state["time"] = time.time() - state["start_time"] - paused_time
        return state["time"]

    def compute_eta(self, meta, metas_to_fetch):
        eta_seconds = estimated_time_per_loc.get(meta, 120) * metas_to_fetch.get(meta, 0)
        return self.format_time(eta_seconds)

    def format_time(self, seconds):
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02}:{m:02}:{s:02}"

    def set_step(self, step: str):
        self.current_step = step

    def update_found(self, meta):
        self.meta_states[meta]["checked"] += 1
        self.meta_states[meta]["found"] += 1
        self.total_found += 1

    def update_no_street_view(self, meta):
        self.meta_states[meta]["checked"] += 1
        self.meta_states[meta]["no_street_view"] += 1

    def update_faulty(self, meta):
        self.meta_states[meta]["checked"] += 1
        self.meta_states[meta]["faulty"] += 1


#global inits
#needed for IDE update
console = Console(force_terminal=True)
#init progress bar

valid_found_locations = 0

def setup_selenium():
    options = webdriver.ChromeOptions()
    # comment out if you want the live view of the fetching
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    # reduces verbose logs
    options.add_argument("--disable-gcm")
    options.add_argument("--disable-logging")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])  # hides extra warnings on Windows

    #no logs in the console
    service = Service()
    #redirect stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        driver = webdriver.Chrome(service=service, options=options)
    finally:
        # Restore stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # load google maps to accept cookies
    driver.get("https://www.google.com/maps")
    time.sleep(one_sleep_time_in_seconds)
    try:
        accept_button = driver.find_element(
            By.XPATH,
            '//button[contains(., "Accept") or contains(., "Alle akzeptieren")]'
        )
        accept_button.click()
        return driver
    except:
        return driver
import time
import requests

def query_overpass_api(query,meta_name,new_fetch=False,new_save=False):
    coords = []
    os.makedirs(OVERPASS_SAVE_DIR, exist_ok=True)
    if os.path.exists(f"{OVERPASS_SAVE_DIR}/{meta_name}.json") and not new_fetch:
        with open(f"{OVERPASS_SAVE_DIR}/{meta_name}.json", "r", encoding="utf-8") as f:
            coords = json.load(f)
        prog.set_step(f"Loaded {len(coords)} coords for meta {meta_name} from saved file")
        return coords
    if isinstance(query, dict):
        if os.path.exists(f"{OVERPASS_SAVE_DIR}/{meta_name}.json") and new_fetch and not new_save:
            coords = []
        elif os.path.exists(f"{OVERPASS_SAVE_DIR}/{meta_name}.json"):
            with open(f"{OVERPASS_SAVE_DIR}/{meta_name}.json", "r", encoding="utf-8") as f:
                coords = json.load(f)
        for dis_code, disc_query in query.items():
            prog.set_step(f"Fetching {dis_code} for {meta_name}")
            while True:
                try:
                    r = requests.post(OVERPASS_API_URL, data={'data': disc_query})
                    r.raise_for_status()
                    break
                except requests.exceptions.RequestException:
                    time.sleep(5)
            locations = r.json()
            for loc in locations['elements']:
                if loc['type'] == 'way' and 'geometry' in loc:
                    coords.extend([(g['lat'], g['lon']) for g in loc['geometry']])
            with open(f"{OVERPASS_SAVE_DIR}/{meta_name}.json", "w", encoding="utf-8") as f:
                json.dump(coords, f)
    else:
        if os.path.exists(f"{OVERPASS_SAVE_DIR}/{meta_name}.json") and new_fetch and not new_save:
            coords = []
        elif os.path.exists(f"{OVERPASS_SAVE_DIR}/{meta_name}.json"):
            with open(f"{OVERPASS_SAVE_DIR}/{meta_name}.json", "r", encoding="utf-8") as f:
                coords = json.load(f)
        while True:
            try:
                r = requests.post(OVERPASS_API_URL, data={'data': query})
                r.raise_for_status()
                break
            except requests.exceptions.RequestException:
                time.sleep(5)
        locations = r.json()
        for loc in locations['elements']:
            if loc['type'] == 'way' and 'geometry' in loc:
                coords.extend([(g['lat'], g['lon']) for g in loc['geometry']])
        with open(f"{OVERPASS_SAVE_DIR}/{meta_name}.json", "w", encoding="utf-8") as f:
            prog.set_step(f"Saved {len(coords)} coords for meta {meta_name} ")
            json.dump(coords, f)
    return coords



def streetview_available(driver, lat, lon, threshold = 0.8):
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    driver.get(url)
    time.sleep(two_sleep_time_in_seconds)
    #check if most of the pixels are black = no streetview available
    screenshot_data = driver.get_screenshot_as_png()
    #convert to an array to count the black pixels
    img_array = np.array(Image.open(io.BytesIO(screenshot_data)).convert("L"))
    #count all pixels with a value <30 (black - gray)
    black_pixels_count = np.sum(img_array < 30)
    ratio = black_pixels_count / img_array.size
    return ratio < threshold

def setup_folders(meta, lat, lon):
    unlabeled_folder = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled"
    zoom_folder = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder"
    labeled_folder = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled"
    output_folder_s1 = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled_zoomed/stage0"
    output_folder_s2 = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled_zoomed/stage1"
    os.makedirs(unlabeled_folder, exist_ok=True)
    os.makedirs(zoom_folder, exist_ok=True)
    os.makedirs(labeled_folder, exist_ok=True)
    os.makedirs(output_folder_s1, exist_ok=True)
    os.makedirs(output_folder_s2, exist_ok=True)

def init_location(driver, url, skip_loading):
    if not skip_loading:
        driver.get(url)
        time.sleep(two_sleep_time_in_seconds)
    #click in the center to focus windows
    actions = ActionChains(driver)
    window_size = driver.get_window_size()
    center_x = window_size['width'] // 2
    center_y = window_size['height'] // 2
    actions.move_by_offset(center_x, center_y).click().perform()
    actions.reset_actions()
    time.sleep(zero_five_sleep_time_in_seconds)
    return actions

def wait_for_streetview_fully_loaded(driver, check_interval=zero_five_sleep_time_in_seconds):
    end_time = time.time() + four_sleep_time_in_seconds
    last_img = None
    #compare last_img and new image
    while time.time() < end_time:
        screenshot_data = driver.get_screenshot_as_png()
        img_array = np.array(Image.open(io.BytesIO(screenshot_data)).convert("L"))
        #Only use every 5th pixel => speedup comparison
        img_array = img_array[::5, ::5]

        if last_img is not None:
            diff = np.mean(np.abs(img_array - last_img))
            if diff < 1:
                #no differences => assume fully loaded
                return True

        last_img = img_array
        time.sleep(check_interval)
    return False

def bounding_box(predictions, model, image_file_name, img, meta, lat, lon):
    # split by zoom step / pan
    split_by_ = image_file_name.split("_")
    zoom_lvl = split_by_[1][-1]
    pan_lvl = split_by_[2][1]

    if predictions is None or len(predictions) == 0:
        return None

    for box in predictions:
        # Determine label and confidence depending on input type
        if model is not None:
            # YOLO model
            pred_label = model.names[int(box.cls[0])]
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        else:
            # TF/TFLite detection
            pred_label = box[6]        # label stored in last element
            score = float(box[5])      # confidence
            x1, y1, x2, y2 = map(int, box[0:4])

        label_conf = f"{pred_label} {score:.2f}"

        # Save the coords of the found detection (with the pan/zoom level)
        found_detections.append((
            zoom_lvl,
            pan_lvl,
            ((x2 - x1) / 2) + x1,
            ((y2 - y1) / 2) + y1
        ))

        if SAVE_IMAGES:
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label background
            (w, h), _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
            # Put text
            cv2.putText(img, label_conf, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if SAVE_IMAGES:
        output_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled", image_file_name)
        cv2.imwrite(output_path, img)

def run_inference_stage_1(meta,lat,lon):
    global prog
    model = YOLO(MODEL_STAGE1)
    test_images = os.listdir(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled")
    for image_file_name in test_images:
        prog.set_step(f"Detecting {meta} in {image_file_name} ({lat},{lon})")
        full_image_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled", image_file_name)
        img = cv2.imread(full_image_path)
        predictions = model.predict(full_image_path,verbose=False, imgsz=IMAGE_SIZE_STAGE, conf=CONF_THRESHOLD_STAGE1, stream=False)[0].boxes
        bounding_box(predictions, model, image_file_name, img, meta, lat, lon)
    prog.cancel_point()
    prog.set_step(f"Done annotating images {meta} ({lat},{lon})")

def parse_streetview_link(link):

    coords = re.search(r'@([-0-9.]+),([-0-9.]+),', link)
    lat, lng = (float(coords.group(1)), float(coords.group(2))) if coords else (None, None)

    zoom_match = re.search(r'([0-9.]+)y', link)
    heading_match = re.search(r'([0-9.]+)h', link)
    pitch_match = re.search(r'([0-9.]+)t', link)

    #zoom = float(zoom_match.group(1)) if zoom_match else 1.5
    zoom = float(zoom_match.group(1)) if zoom_match else 1.5
    heading = float(heading_match.group(1)) if heading_match else 0.0

    pitch = float(pitch_match.group(1)) if pitch_match else 0.0
    pitch = pitch - 90

    pano_match = re.search(r'!1s([a-zA-Z0-9_-]+)', link)
    pano_id = pano_match.group(1) if pano_match else None
    prog.set_step(f"Added location {lat}, {lng} successfully")
    return {
        "lat": lat,
        "lng": lng,
        "heading": heading,
        "pitch": pitch,
        "zoom": zoom,
        "panoId": None,
        "countryCode": None,
        "stateCode": None,
        "extra": {
            "panoId": pano_id,
            "panoDate": None
        }
    }


def save_location(link,meta):
    global current_save_file
    if os.path.exists(current_save_file):
        with open(current_save_file, "r") as f:
            data = json.load(f)
    else:
        data = {"name": meta, "customCoordinates": []}

    entry = parse_streetview_link(link)
    data["customCoordinates"].append(entry)

    with open(current_save_file, "w") as f:
        json.dump(data, f, indent=2)


def validate_found_location(labels,meta):
    global valid_found_locations
    valid_search_labels = meta_validate_results.get(meta, ())
    #remove all false detected metas - (None) when no meta is detected
    filtered = [
        (meta if label in valid_search_labels else label, conf)
        for (label, conf) in labels
        if label is not None
    ]
    #no valid meta found
    if not filtered:
        prog.update_faulty(meta)
        return None
    #find the index with the highest confidence for the search label (we save only this location)
    max_index = None
    max_conf = -1
    for i, (label, conf) in enumerate(filtered):
        if label in valid_search_labels and conf > max_conf:
            max_conf = conf
            max_index = i

    counts = Counter([label for (label, _) in filtered])
    #valid location if the detected labels are at least 50% valid
    label_count = sum(counts.get(lbl, 0) for lbl in valid_search_labels)
    #eg if 2 austria bollards and 1 france bollard - assume the france is false
    if label_count / len(filtered) >= 0.5:
        valid_found_locations+=1
        prog.update_found(meta)
        if max_index is not None:
            save_location(zoomed_links[max_index],meta)


def run_inference_stage_2(meta, lat, lon):
    global prog

    output_folder_s1 = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled_zoomed/stage0"
    output_folder_s2 = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled_zoomed/stage1"

    model_s1 = YOLO(MODEL_STAGE1)
    model_s2 = YOLO(MODEL_STAGE2)
    test_images = os.listdir(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder")
    labels = []
    for image_file_name in test_images:
        prog.set_step(f"Validating found meta {meta} in zoomed in image {image_file_name} ({lat},{lon})")
        #"streetview"+str(image_file_name.split("zoomed")[-1])
        full_image_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder",image_file_name )
        img = cv2.imread(full_image_path)
        predictions_s1 = model_s1.predict(full_image_path,verbose=False, imgsz=IMAGE_SIZE_STAGE, conf=CONF_THRESHOLD_STAGE1, stream=False)[0].boxes

        if predictions_s1 is None or len(predictions_s1) == 0:
            continue
        img_stage0 = img.copy()
        img_stage1 = img.copy()
        best_detected_label = None
        best_conf = 0

        for box in predictions_s1:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(img_stage0, (x1, y1), (x2, y2), (255, 128, 0), 1)
            cv2.putText(img_stage0, f"bollard {box.conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            #crop for the 2nd stage
            crop_img = img[y1:y2, x1:x2]
            #skip small cropped metas
            if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                continue
            predictions_s2 = model_s2.predict(crop_img,verbose=False, imgsz=IMAGE_SIZE_STAGE, conf=CONF_THRESHOLD_STAGE2, stream=False)[0].boxes
            for box_s2 in predictions_s2:
                pred_label = model_s2.names[int(box_s2.cls[0])]
                label_conf = f"{pred_label} {float(box_s2.conf[0]):.2f}"
                sx1, sy1, sx2, sy2 = map(int, box_s2.xyxy[0].cpu().numpy())
                #position in the uncropped image
                abs_x1 = x1 + sx1
                abs_y1 = y1 + sy1
                abs_x2 = x1 + sx2
                abs_y2 = y1 + sy2
                if float(box_s2.conf[0]) > best_conf:
                    best_conf = float(box_s2.conf[0])
                    best_detected_label = pred_label
                cv2.rectangle(img_stage1, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                cv2.putText(img_stage1, label_conf, (abs_x1, abs_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if SAVE_IMAGES:
            cv2.imwrite(os.path.join(output_folder_s1, image_file_name), img_stage0)
            cv2.imwrite(os.path.join(output_folder_s2, image_file_name), img_stage1)
        labels.append((best_detected_label, best_conf))
    prog.cancel_point()
    validate_found_location(labels,meta)

#meassuered timing/pitch (by hand)
calibration_px_x = [100, 150,200, 300, 400, 500, 600, 700, 800, 900, 1000]
calibration_time = [0.32,0.37, 0.40, 0.49, 0.565, 0.64, 0.685, 0.73, 0.77,0.785,0.8]
calibration_px_y = [50, 100, 150, 200, 250, 300, 350, 400, 450]
calibration_offset_pitch = [4.52, 8.96, 13.09, 17.75, 22.55, 27.46, 30.46, 33.57, 37.1]
def hold_time_from_dx(dx):
    dx = abs(dx)
    if dx <= calibration_px_x[0]:
        #scale linear to 0px
        return calibration_time[0] * dx / calibration_px_x[0]
    elif dx >= calibration_px_x[-1]:
        #above my measured
        return calibration_time[-1] * dx / calibration_px_x[-1]
    else:
        return float(np.interp(dx, calibration_px_x, calibration_time))

def pitch_from_dy(dy):
    direction = -1 if dy > 0 else 1
    dy = abs(dy)
    if dy <= calibration_px_y[0]:
        offset = calibration_offset_pitch[0] * dy / calibration_px_y[0]
    elif dy >= calibration_px_y[-1]:
        offset = calibration_offset_pitch[-1]
    else:
        offset = float(np.interp(dy, calibration_px_y, calibration_offset_pitch))
    #90 is the base pitch
    return 90 + direction * offset


def update_url_pitch(url, dy):
    new_pitch = pitch_from_dy(dy)  # your function to compute new pitch

    # Replace the first occurrence of a t-value (number followed by 't')
    updated_url = re.sub(
        r'\d+(\.\d+)?t',
        f'{new_pitch:.2f}t',
        url,
        count=1  # only replace the first one
    )

    return updated_url

def look_at_object(driver, pan_index, obj_x, obj_y,meta, lat, lon,idx, zoom_steps=2):

    global prog
    window_size = driver.get_window_size()
    center_x = window_size['width'] // 2
    center_y = window_size['height'] // 2

    dx = obj_x - center_x  # positive = object right of center
    dy = obj_y - center_y  # positive = object below center

    if dx > 0:
        actions = ActionChains(driver)
        actions.key_down(Keys.ARROW_RIGHT).perform()
        time.sleep(hold_time_from_dx(dx))
        actions = ActionChains(driver)
        actions.key_up(Keys.ARROW_RIGHT).perform()
    if dx < 0:
        actions = ActionChains(driver)
        actions.key_down(Keys.ARROW_LEFT).perform()
        time.sleep(hold_time_from_dx(dx))
        actions = ActionChains(driver)
        actions.key_up(Keys.ARROW_LEFT).perform()

    time.sleep(one_five_sleep_time_in_seconds)
    object_centered_url = update_url_pitch(driver.current_url,dy)
    driver.get(object_centered_url)
    wait_for_streetview_fully_loaded(driver)



    scene_elem = driver.find_element(By.CSS_SELECTOR, '[aria-label="Street View"]')

    js_code = """
            const targetX = arguments[0];
            const targetY = arguments[1];
            const zoomIn = arguments[2];
            const steps = arguments[3];
            const scene = document.querySelector('[aria-label="Street View"]');
            if (!scene) {
                console.error("Street View element not found");
                return;
            }
            const rect = scene.getBoundingClientRect();
            console.log("[JS DEBUG] Street View size:", rect.width, rect.height);

            const safeX = Math.max(0, Math.min(targetX, rect.width - 1));
            const safeY = Math.max(0, Math.min(targetY, rect.height - 1));

            const clientX = rect.left + safeX;
            const clientY = rect.top + safeY;

            // Hover over detected object
            const moveEvt = new MouseEvent('mousemove', {
                bubbles: true,
                cancelable: true,
                clientX: clientX,
                clientY: clientY
            });
            scene.dispatchEvent(moveEvt);

            // Zoom
            const delta = zoomIn ? -100 : 100;
              function sendWheel(step) {
            scene.dispatchEvent(new WheelEvent('wheel', {
                bubbles: true,
                cancelable: true,
                clientX,
                clientY,
                deltaY: delta
            }));
        }
        //small sleep to give streetview time for loading
        for (let i = 0; i < steps; i++) {
            setTimeout(() => sendWheel(i), i * 200); 
        }
            //move courser to the top left (no move arrow visable on the screenshot)
            const topLeftEvt = new MouseEvent('mousemove',{
                bubbles: true,
                cancelable: true,
                clientX: rect.left,
                clientY: rect.top
            });
            scene.dispatchEvent(topLeftEvt);
        """
    driver.execute_script(js_code, center_x, center_y, True, zoom_steps)
    time.sleep(one_sleep_time_in_seconds)
    zoomed_links.append(driver.current_url)
    if SAVE_IMAGES:
        screenshot_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder", f"zoomed_p{pan_index}_{idx}.png")
        driver.save_screenshot(screenshot_path)

def zoom_in_found_objects(driver, url, pan_hold_time, meta,lat,lon,look_at_every_object = False):
    global prog
    if len(found_detections) == 0:
        prog.set_step(f"No meta detections found ({lat}, {lon})")
        return
    for idx, (zoom_lvl, pan_lvl, det_x, det_y) in enumerate(found_detections):
        prog.cancel_point()
        prog.set_step(f"Zoom in detected detection of {meta} ({lat}, {lon}):  {idx+1}/{len(found_detections)}")
        actions = init_location(driver, url, skip_loading=False)
        for i in range(int(pan_lvl)):
            wait_for_streetview_fully_loaded(driver)
            actions = ActionChains(driver)
            actions.key_down(Keys.ARROW_RIGHT).perform()
            time.sleep(pan_hold_time)
            actions = ActionChains(driver)
            actions.key_up(Keys.ARROW_RIGHT).perform()
            time.sleep(zero_five_sleep_time_in_seconds)
        for _ in range(zoom_scrolls[int(zoom_lvl)]):
            actions.send_keys(Keys.ADD).perform()
            time.sleep(zero_five_sleep_time_in_seconds)
        time.sleep(zero_five_sleep_time_in_seconds)
        #update_streetview_url(driver.current_url, det_x, det_y,driver)
        look_at_object(driver,pan_lvl,det_x,det_y, meta, lat, lon, idx)


def identify_language(meta,lat,lon,language):
    test_images = os.listdir(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled")
    for image_file_name in test_images:
        prog.set_step(f"Detecting {meta} in {image_file_name} ({lat},{lon})")
        full_image_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled", image_file_name)
        img = cv2.imread(full_image_path)
        #convert image to greyscale - OCR (tesseract) works much better
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #use a gausian filter to reduce noice
        gray = cv2.GaussianBlur(grayscale_image, (4, 4), 0)
        #text is a 1 => backgournd 0 (based on intensity)
        threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 20, 10)
        extreacted_text = pytesseract.image_to_string(threshold_image, lang=language)
        if not extreacted_text:
            return {"text": "", "language": "unknown"}
        try:
            lang = detect(extreacted_text)
        except:
            lang = "unknown"
        return {"text": extreacted_text, "language": lang}

#Non maxima surpression - Its needed for the TF model -> otherwise we have lots of overlapping, nearly idientical labels
def nms_tflite(detections, iou_threshold=0.8):
    if len(detections) == 0:
        return []
    boxes = np.array([[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections], dtype=np.float32)  # x, y, w, h
    scores = np.array([d[5] for d in detections], dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.01,
        nms_threshold=iou_threshold
    )
    if len(indices) == 0:
        return []
    filtered = [detections[i] for i in indices.flatten()]
    return filtered

def detect_num_plates_s1(meta, lat, lon):
    global prog
    #FLite model
    interpreter = tf.lite.Interpreter(model_path=CAR_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    test_images = os.listdir(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled")
    for image_file_name in test_images:
        prog.set_step(f"Detecting {meta} in {image_file_name} ({lat},{lon})")
        full_image_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled", image_file_name)
        img = cv2.imread(full_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resize input to 640,640
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        resized = cv2.resize(img_rgb, (w, h))
        input_tensor = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)
        #inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        h_img, w_img, _ = img.shape
        detections = []

        for pred in output_data:
            score = pred[4]
            if score < CONF_THRESHOLD_STAGE1:
                continue
            #class id
            cls_id = int(np.argmax(pred[5:]))
            label = CLASS_LABELS_CARS[cls_id]

            if label not in KEEP_CLASSES_CARS:
                continue
            #box coordinates (YOLO notation)
            x, y, bw, bh = pred[0:4]
            x1 = int((x - bw/2) * w_img)
            y1 = int((y - bh/2) * h_img)
            x2 = int((x + bw/2) * w_img)
            y2 = int((y + bh/2) * h_img)
            #skip small detections (zoom level will not be sufficient for a good labeling in stage 2)
            if abs(x2 - x1) < 60 or abs(y2 - y1) < 60:
                continue
            detections.append([x1, y1, x2, y2, cls_id, float(score), label])
        detections = nms_tflite(detections)
        bounding_box(detections, None, image_file_name, img, meta, lat, lon)

    prog.cancel_point()
    prog.set_step(f"Done annotating images {meta} ({lat},{lon})")

def detect_num_plates_s2(meta, lat, lon):
    global prog
    VEHICLE_CLASSES = {2, 3, 5, 7}
    output_folder_s1 = f"{BASE_FOLDER}/{meta}/{lat}_{lon}/labeled_zoomed/stage0"
    model = YOLO(CAR_MODEL)
    test_images = os.listdir(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder")
    labels = []
    for image_file_name in test_images:
        prog.set_step(f"Validating found meta {meta} in zoomed in image {image_file_name} ({lat},{lon})")
        # "streetview"+str(image_file_name.split("zoomed")[-1])
        full_image_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/zoom_folder", image_file_name)
        img = cv2.imread(full_image_path)
        predictions = model.predict(full_image_path, verbose=False, imgsz=IMAGE_SIZE_STAGE, conf=CONF_THRESHOLD_STAGE1, stream=False)[0]
        filtered_boxes = []
        for box in predictions.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                filtered_boxes.append(box)
        bounding_box(filtered_boxes, model, image_file_name, img, meta, lat, lon)


def capture_streetview_360_degree_images(driver, lat, lon, meta, pan_hold_time = one_sleep_time_in_seconds):
    global prog
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    actions = init_location(driver, url, skip_loading=True)
    prog.set_step(f"Taking 4 screenshots (360 degree) for  {meta} ({lat},{lon})")
    for pan_idx in range(pan_steps[0]):
        prog.cancel_point()
        wait_for_streetview_fully_loaded(driver)
        screenshot_path = os.path.join(f"{BASE_FOLDER}/{meta}/{lat}_{lon}/unlabeled", f"streetview_z0_p{pan_idx}.png")
        driver.save_screenshot(screenshot_path)
        actions.key_down(Keys.ARROW_RIGHT).perform()
        time.sleep(pan_hold_time)
        actions.key_up(Keys.ARROW_RIGHT).perform()
        time.sleep(zero_five_sleep_time_in_seconds)
    if "bollard" in meta:
        run_inference_stage_1(meta,lat,lon)
        zoom_in_found_objects(driver, url, pan_hold_time, meta, lat, lon)
        run_inference_stage_2(meta, lat, lon)
        return
    else:
        detect_num_plates_s1(meta,lat,lon)
        zoom_in_found_objects(driver, url, pan_hold_time, meta, lat, lon)
        detect_num_plates_s2(meta, lat, lon)
        return

















import re
from urllib.parse import urlparse, parse_qs, urlunparse
import re

"""
def update_streetview_url(url, obj_x, obj_y, driver):

    # --- Compute offsets relative to screen center ---
    window_size = driver.get_window_size()
    center_x = window_size['width'] // 2
    center_y = window_size['height'] // 2

    dx = obj_x - center_x  # positive = object right of center
    dy = obj_y - center_y  # positive = object below center

    # --- Conversion factors (calibrated empirically) ---
    heading_per_px = -0.0826  # degrees per horizontal pixel
    tilt_per_px = -0.0941  # degrees per vertical pixel

    # --- Extract current heading and tilt from URL ---
    match = re.search(r'(\d+(\.\d+)?)h,(\d+(\.\d+)?)t', url)
    if match:
        heading = float(match.group(1))
        tilt = float(match.group(3))
    else:
        heading = 0.0
        tilt = 0.0

    # --- Apply offsets ---
    # Sign convention: right/bottom = negative, left/top = positive
    new_heading = heading + dx * heading_per_px
    new_tilt = tilt + dy * tilt_per_px

    # --- Clamp / wrap ---
    new_tilt = max(0, min(90, new_tilt))
    new_heading = new_heading % 360

    # --- Replace or insert h,t in URL ---
    if match:
        new_url = re.sub(r'(\d+(\.\d+)?)h,(\d+(\.\d+)?)t',
                         f"{new_heading:.2f}h,{new_tilt:.2f}t", url)
    else:
        # Insert after zoom/pitch/yaw segment
        new_url = url.replace("/data=", f"/{new_heading:.2f}h,{new_tilt:.2f}t/data=")

    return new_url
"""






def live_thread(prog: ProgressManager, refresh_per_second: int = 1):
    with Live(prog.render(), console=console, refresh_per_second=refresh_per_second) as live:
        while prog.running:
            live.update(prog.render())
            time.sleep(1 / refresh_per_second)

def save_meta_times(prog, filename="meta_times.csv"):
    # Ensure directory exists
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Meta", "Found", "Time (seconds)"])

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for meta, state in prog.meta_states.items():
            start_time = state.get("start_time")
            total_time = state.get("time")
            count = state.get("found", 0)

            if total_time is None and start_time:
                # calculate elapsed if not stored yet
                total_time = time.time() - start_time

            writer.writerow([meta, count, round(total_time or 0, 2)])
def get_unique_filename(filename):
    name = filename.split(".json")[0]
    new_path = os.path.join(BASE_FOLDER, filename)
    counter = 1
    while os.path.exists(new_path):
        new_filename = f"{name}({counter}).json"
        new_path = os.path.join(BASE_FOLDER, new_filename)
        counter += 1
    return new_path
def find_locations(metas_to_fetch=None, resume_state=None):
    global prog, current_save_file
    driver = setup_selenium()
    if resume_state:
        metas_to_fetch = resume_state["metas_to_fetch"]

    prog = ProgressManager(metas_to_fetch,driver)
    if resume_state:
        current_save_file = resume_state["save_file"]
        prog.metas_to_fetch = resume_state["metas_to_fetch"]
        prog.meta_states = resume_state["meta_states"]
        prog.total_found = resume_state["total_found"]
        prog.total_to_fetch = resume_state["total_to_fetch"]
        prog.total_paused_time = resume_state.get("total_paused_time", 0)
        prog.paused = resume_state.get("paused", False)
        prog.current_step = resume_state.get("current_step", "Resumed...")

        #restore time and adjust for offline gap
        if resume_state.get("global_start_time"):
            saved_timestamp = resume_state.get("saved_timestamp")
            resume_timestamp = time.time()
            prog.start_time = resume_state["global_start_time"]
            if saved_timestamp:
                offline_gap = resume_timestamp - saved_timestamp
                #shift start time
                prog.start_time += offline_gap
                for meta_state in prog.meta_states.values():
                    if meta_state["start_time"] is not None:
                        meta_state["start_time"] += offline_gap
    thread = threading.Thread(target=live_thread, args=(prog, 1), daemon=True)
    thread.start()
    if current_save_file is None:
        if settings["append_mode"]:
            save_path = os.path.join(BASE_FOLDER, settings["save_filename"])
            if not os.path.exists(save_path):
                with open(save_path, "w") as f:
                    json.dump({"name": None, "customCoordinates": []}, f, indent=2)
        else:
            save_path = get_unique_filename(settings["save_filename"])
            with open(save_path, "w") as f:
                json.dump({"name": None, "customCoordinates": []}, f, indent=2)
        current_save_file = save_path
    # fetch metas
    for meta, count in metas_to_fetch.items():
        already_checked = prog.meta_states[meta]["found"]

        remaining = count - already_checked
        if remaining <= 0:
            continue
        #meta has not been started yet
        if prog.meta_states[meta]["start_time"] is None:
            prog.start_meta(meta)
        global valid_found_locations
        search_query = metas_query_mapping[meta]
        locations = query_overpass_api(search_query, meta)
        valid_found_locations = already_checked
        while valid_found_locations < count:
            prog.wait_if_paused()
            prog.cancel_point()
            sample = random.sample(locations, min(1, len(locations)))
            lat, lon = sample[0][0], sample[0][1]
            # 41.6902838_12.7548271 italy
            if streetview_available(driver, lat, lon):
                setup_folders(meta, lat, lon)
                capture_streetview_360_degree_images(driver, lat, lon, meta)
                found_detections.clear()
                zoomed_links.clear()
            else:
                prog.update_no_street_view(meta)
        prog.stop_time(meta)
    time.sleep(two_sleep_time_in_seconds)
    save_meta_times(prog)
    prog.running = False
    thread.join()
    if os.path.exists(SAVE_RESUME_FETCHING_FILE):
        os.remove(SAVE_RESUME_FETCHING_FILE)
    start_screen_loop()

"""
if __name__ == "__main__":
    #selenium setup
    driver = setup_selenium()
    #fetch metas
    for meta, count in metas_to_fetch.items():
        search_query =  metas_query_mapping[meta]
        locations = query_overpass_api(search_query,meta)
        valid_found_locations = 0
        while valid_found_locations < count:
            sample = random.sample(locations, min(1, len(locations)))
            lat, lon = sample[0][0], sample[0][1]

            #41.6902838_12.7548271 italy
            if streetview_available(driver, lat, lon):
                setup_folders(meta, lat, lon)
                capture_streetview_360_degree_images(driver, lat, lon, meta)
                found_detections.clear()
                zoomed_links.clear()
            else:
                prog.update_no_street_view(meta)
    time.sleep(one_sleep_time_in_seconds)
"""