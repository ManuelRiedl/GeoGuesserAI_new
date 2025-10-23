import json
import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule
from rich.text import Text
from rich.table import Table
import readchar  # pip install readchar
import script_find_locations as find_locations
from variables import zero_five_sleep_time_in_seconds, zero_one_sleep_time_in_seconds, SAVE_RESUME_FETCHING_FILE, settings, SAVE_STATE_FILE
import variables as v
console = Console()
import os
import keyboard
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
# -------------------------------
# Meta configuration
# -------------------------------
metas_selected_from_user = {
    "austria_bollard": 0,
    "germany_bollard": 0,
    "luxenburg_bollard": 0,
    "slovenia_bollard": 0,
    "france_bollard": 0,
    "spain_bollard": 0,
    "portugal_bollard": 0,
    "italy_bollard": 0,
    "polen_bollard": 0,
    "hungary_bollard": 0,
    "czechia_bollard": 0,
    "croatia_bollard": 0,
    "lithuania_bollard": 0,
    "latvia_bollard": 0,
    "estonia_bollard": 0,
    "sweden_bollard": 0,
    "finland_bollard": 0,
    "norway_bollard": 0,
    "iceland_bollard": 0,
    "great_britain_num_plate":0,
    "us_district_of_columbia_num_plate": 0,
    "us_new_york_num_plate": 0
}
estimated_time_per_loc = {
    "austria_bollard": 60,
    "germany_bollard": 45,
    "luxenburg_bollard": 57,
    "slovenia_bollard": 34,
    "france_bollard": 34,
    "spain_bollard": 33,
    "portugal_bollard": 23,
    "italy_bollard": 60,
    "polen_bollard": 60,
    "hungary_bollard": 60,
    "czechia_bollard": 60,
    "croatia_bollard": 60,
    "lithuania_bollard": 60,
    "latvia_bollard": 60,
    "estonia_bollard": 60,
    "sweden_bollard": 60,
    "finland_bollard": 60,
    "norway_bollard": 60,
    "iceland_bollard": 60,
    "great_britain_num_plate":30,
    "us_district_of_columbia_num_plate": 20,
    "us_new_york_num_plate": 20
}
BUTTONS_START = ["Configure Metas", "Start Fetching", "Settings"]
warning_message ="No metas selected!  "
resume_message = Text.from_markup("Press [bold green]r[/bold green] to resume last fetching")
def render_start_screen(selected_idx=0):
    selected_metas = sum(1 for c in metas_selected_from_user.values() if c > 0)
    total_locations = sum(metas_selected_from_user.values())

    buttons = []
    for i, btn_text in enumerate(BUTTONS_START):
        text_style = "bold white on #CC241D" if i == selected_idx else "bold #CC241D"
        border_style = "#00739C"  # fixed border color

        buttons.append(
            Panel(
                Text(btn_text, style=text_style, justify="center"),
                border_style=border_style,
                padding=(0, 4),
                expand=False
            )
        )

    buttons_row = Columns(buttons, align="center", expand=False)
    warning = None
    if selected_metas == 0:
        warning = Align.center(Text(warning_message, style="bold #78BCC4"))
    if selected_metas == 0 and os.path.exists(SAVE_RESUME_FETCHING_FILE):
        combined = Text(warning_message, style="bold #78BCC4")
        combined.append(resume_message)
        warning = Align.center(combined)
    elif os.path.exists(SAVE_RESUME_FETCHING_FILE):
        warning = Align.center(resume_message)
    content_items = [
        Align.center(Text("Welcome to the Geoguessr Meta Finder!", style="bold white")),
        Align.center(Text("Enter to select | q to exit ", style="italic #AAAAAA")),
        Align.center(Text(" ", style="bold white")),
        Align.center("Use this tool to fetch random locations based on the selected metas."),
        Align.center("Select metas, set counts and start fetching!"),
        Align.center(""),
        Align.center(f"Selected Metas: {selected_metas}      Locations to fetch: {total_locations}"),
    ]
    if warning:
        content_items.append(warning)
    content_items.append(Align.center(buttons_row))

    content = Group(*content_items)
    welcome_panel = Panel(
        content,
        title="[bold #CC241D]Geoguessr-Meta Finder AI[/]",
        border_style="#00739C",
        padding=(1, 2),
        expand=True,
    )

    console.print(welcome_panel)
def render_settings_screen(selected_idx=0):
    # Define settings with name, value, and description
    cls()
    settings_list = [
        {
            "name": "Filename",
            "value": settings['save_filename'],
            "description": "The JSON file where detected locations are saved (root folder)."
        },
        {
            "name": "Append Mode",
            "value": "Yes" if settings['append_mode'] else "No",
            "description": "Whether new locations are appended to the existing file."
        },
        {
            "name": "Debug Mode",
            "value": "Yes" if settings['debug_mode'] else "No",
            "description": "Save screenshots of detected metas (root folder)."
        }
    ]
    # Create table with 3 columns
    table = Table(expand=True, show_edge=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="bold #78BCC4", justify="right")
    table.add_column("Value", style="bold #CC241D", justify="center")
    table.add_column("Description", style="dim white", justify="left")

    # Add rows with highlight for selected_idx
    for i, s in enumerate(settings_list):
        style = "reverse" if i == selected_idx else ""
        table.add_row(s["name"], s["value"], s["description"], style=style)

    # Compose panel content
    content = Group(
        Align.center(Text("Enter to edit | q to return", style="italic #AAAAAA")),
        Align.center(Text("")),
        Align.center(table)
    )

    panel = Panel(
        content,
        title="[bold #CC241D]Settings[/]",
        border_style="#00739C",
        padding=(1, 2),
        expand=True
    )
    cls()
    console.print(panel)
def settings_loop():
    selected_idx = 0
    options = ["save_filename", "append_mode","debug_mode"]

    while True:
        render_settings_screen(selected_idx)
        key = readchar.readkey()
        if key == readchar.key.UP:
            selected_idx = (selected_idx - 1) % len(options)
        elif key == readchar.key.DOWN:
            selected_idx = (selected_idx + 1) % len(options)
        elif key == readchar.key.ENTER:
            if options[selected_idx] == "save_filename":
                cls()
                console.print("Enter new filename (30 chars) - current ([bold #CC241D]{}[/bold #CC241D]):".format(settings["save_filename"]))
                new_filename = input("> ").strip()
                new_filename = new_filename[:30]
                if not new_filename:
                    new_filename = settings["save_filename"]
                if not new_filename.lower().endswith(".json"):
                    new_filename += ".json"
                if new_filename:
                    settings["save_filename"] = new_filename
            elif options[selected_idx] == "append_mode":
                settings["append_mode"] = not settings["append_mode"]
            elif options[selected_idx] == "debug_mode":
                settings["debug_mode"] = not settings["debug_mode"]
        elif key.lower() == "q":
            cls()
            break

def format_seconds(seconds):
    if seconds <= 0:
        return "-"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    parts = []
    if h > 0:
        parts.append(f"{h} h")
    if m > 0 or h > 0:
        parts.append(f"{m} min")
    if s > 0 or (h == 0 and m == 0):
        parts.append(f"{s} sec")

    return " ".join(parts)
def render_select_metas_screen(selected_row=0):
    table = Table(expand=True, show_edge=False, box=None, padding=(0, 1))
    table.add_column("Meta", style="bold #78BCC4", justify="left")
    table.add_column("Amount", style="bold #F7444E", justify="center")
    table.add_column("Estimated Time/location", style="bold #00739C", justify="center")

    total_count = 0
    total_estimated_sec = 0

    for i, (meta, count) in enumerate(metas_selected_from_user.items()):
        style = "reverse" if i == selected_row else ""
        time_per_loc = estimated_time_per_loc.get(meta, 0)
        meta_total_sec = count * time_per_loc

        total_count += count
        total_estimated_sec += meta_total_sec

        estimated_time = format_seconds(meta_total_sec) if count > 0 else "-"
        table.add_row(meta, str(count), estimated_time, style=style)

    # --- Add empty row as visual separation ---
    table.add_row("", "", "")

    # --- Totals row ---
    table.add_row(
        "[bold #CC241D]Total[/bold #CC241D]",
        f"[bold #F7444E]{total_count}[/bold #F7444E]",
        f"[bold #00739C]{format_seconds(total_estimated_sec)}[/bold #00739C]"
    )

    content = Group(
        Align.center(Text("Configure Metas", style="bold white")),
        Align.center(
            "Use ↑/↓ to select a meta | (Strg) ←/→ to change count | c (C) set selected (all) row to 0 | Alt to change all rows | q to return"),
        Align.center(""),
        table
    )

    panel = Panel(
        content,
        title="[bold #CC241D]Geoguessr-Meta Finder AI[/bold #CC241D]",
        border_style="#00739C",
        padding=(1, 2),
        expand=True
    )

    cls()  # clear console
    console.print(panel)


def start_screen_loop():
    selected_idx = 0
    cls()
    while True:
        render_start_screen(selected_idx)
        key = readchar.readkey()
        if key == readchar.key.RIGHT:
            cls()
            selected_idx = (selected_idx + 1) % len(BUTTONS_START)
        elif key == readchar.key.LEFT:
            cls()
            selected_idx = (selected_idx - 1) % len(BUTTONS_START)
        elif key == "r":
            resume_state = load_resume_state()
            find_locations.find_locations(resume_state=resume_state)
            break
        elif key == "q":
            quit_application()
            cls()
            break
        elif key == readchar.key.ENTER:
            if selected_idx == 0:
                cls()
                select_metas_loop()
            elif selected_idx == 1:
                cls()
                metas_to_fetch = {meta: count for meta, count in metas_selected_from_user.items() if count > 0}

                if not metas_to_fetch:
                    global warning_message
                    warning_message = "Select a meta first!  "
                    continue
                find_locations.find_locations(metas_to_fetch=metas_to_fetch)
                warning_message = "Done fetching"
                break
            elif selected_idx == 2:
                cls()
                settings_loop()

def select_metas_loop():
    selected_row = 0
    metas_list = list(metas_selected_from_user.keys())
    while True:
        # Detect Ctrl + Left / Right without blocking
        key_pressed = False
        render_select_metas_screen(selected_row)
        ctrl_pressed = keyboard.is_pressed('ctrl')
        left_pressed = keyboard.is_pressed('left')
        right_pressed = keyboard.is_pressed('right')
        alt_pressed = keyboard.is_pressed('alt')
        if left_pressed:
            step = -1
        elif right_pressed:
            step = 1
        else:
            step = 0
        if step != 0:
            if ctrl_pressed and alt_pressed:
                #Alt + Ctrl → all metas ±5
                for key in metas_selected_from_user:
                    metas_selected_from_user[key] = max(0, metas_selected_from_user[key] + step * 5)
            elif ctrl_pressed:
                #Ctrl → selected meta ±5
                key = metas_list[selected_row]
                metas_selected_from_user[key] = max(0, metas_selected_from_user[key] + step * 5)
            elif alt_pressed:
                #Alt → all metas ±1
                for key in metas_selected_from_user:
                    metas_selected_from_user[key] = max(0, metas_selected_from_user[key] + step * 1)
            else:
                #selected meta ±1
                key = metas_list[selected_row]
                metas_selected_from_user[key] = max(0, metas_selected_from_user[key] + step * 1)
            key_pressed = True
            time.sleep(zero_one_sleep_time_in_seconds)

        if key_pressed:
            continue
        key = readchar.readkey()
        if key == readchar.key.UP:
            selected_row = (selected_row - 1) % len(metas_list)
        elif key == readchar.key.DOWN:
            selected_row = (selected_row + 1) % len(metas_list)
        elif key == "c":
            metas_selected_from_user[metas_list[selected_row]] = 0
        elif key == "C":
            for meta in metas_selected_from_user:
                metas_selected_from_user[meta] = 0
        elif key == "q":
            cls()
            break

def quit_application():
    save_sate = {
        "settings": v.settings,
        "metas_selected_from_user": metas_selected_from_user
    }
    with open(SAVE_STATE_FILE, "w") as f:
        json.dump(save_sate,f, indent=2)

def load_resume_state():
    try:
        with open(SAVE_RESUME_FETCHING_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
def load_last_state():
    global metas_selected_from_user
    if os.path.exists(SAVE_STATE_FILE):
        with open(SAVE_STATE_FILE, "r") as f:
            json_data = json.load(f)
            v.settings.clear()
            v.settings.update(json_data["settings"])
            metas_selected_from_user = json_data["metas_selected_from_user"]

if __name__ == "__main__":
    load_last_state()
    start_screen_loop()