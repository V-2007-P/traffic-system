import csv
import math
import os
import shutil
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "TrafficVideo.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
MODEL_ZIP_PATH = MODEL_PATH + ".zip"
VIOLATIONS_CSV = os.path.join(BASE_DIR, "violations.csv")

YOLO_CONFIG_DIR = os.path.join(BASE_DIR, "Ultralytics")
MPLCONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)

import cv2
import torch
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from PIL import Image, ImageTk
import pyttsx3

BG = "#06111f"
PANEL = "#0b1e33"
PANEL_2 = "#102a44"
BORDER = "#173a5d"
TEXT = "#e8f2ff"
MUTED = "#89a2bd"
CYAN = "#1dd6ff"
GREEN = "#35d66b"
BLUE = "#2f7dff"
YELLOW = "#ffc43d"
PURPLE = "#9a6cff"
RED = "#ff4d5e"

SPEED_LIMIT = 120
ZEBRA_Y = 170
ZEBRA_COOLDOWN = 3


def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return
    if os.path.exists(MODEL_ZIP_PATH):
        shutil.copyfile(MODEL_ZIP_PATH, MODEL_PATH)
        print("Created model file:", MODEL_PATH)
        return
    raise FileNotFoundError(f"YOLO model not found. Put yolov8n.pt in {BASE_DIR}")


def speak(text):
    if engine is None:
        return
    with speak_lock:
        engine.say(text)
        engine.runAndWait()


def get_track_id(cx, cy):
    global track_id
    for tid, (px, py, _) in tracks.items():
        if math.hypot(cx - px, cy - py) < 90:
            return tid
    tid = track_id
    track_id += 1
    return tid


def refine_class(name, conf, w, h):
    aspect = w / (h + 1e-6)
    area = w * h
    if name == "person" and conf >= 0.60:
        return "person"
    if name in ["motorcycle", "bicycle"]:
        return "bike"
    if name == "car":
        return "car" if area > 6000 and aspect > 0.9 else "bike"
    if name in ["bus", "truck"]:
        return name
    return None


def set_status(text, color):
    status_var.set(text)
    status_dot.configure(fg=color)
    status_value.configure(fg=color)


def start_video():
    global cap, running, tracks, track_id, speed_memory, violation_memory, traffic_history
    if running:
        return
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        set_status("Video not found", RED)
        print("Video not found:", VIDEO_PATH)
        return
    running = True
    tracks = {}
    track_id = 0
    speed_memory.clear()
    violation_memory.clear()
    traffic_history = []
    set_status("System Active", GREEN)
    process()


def stop_video():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None
    set_status("System Offline", RED)


def save_violation(tid, speed, frame, vehicle_type):
    now = time.time()
    if now - violation_memory.get(tid, 0) < 5:
        return
    violation_memory[tid] = now

    ts = time.strftime("%I:%M:%S %p")
    filename = f"violation_{tid}_{int(now)}.jpg"
    cv2.imwrite(os.path.join(BASE_DIR, filename), frame)

    with open(VIOLATIONS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([tid, vehicle_type, f"{speed:.1f}", ts, filename])

    text = f"Overspeeding   {vehicle_type.upper()} | ID:{tid}   {ts}"
    violations_list.insert(0, text)
    while violations_list.size() > 8:
        violations_list.delete(tk.END)


def draw_badge(frame, x1, y1, text, color):
    y = max(24, y1 - 46)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y - 20), (x1 + tw + 14, y + th + 12), color, -1)
    cv2.putText(frame, text, (x1 + 7, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def process():
    global tracks, zebra_alert_time
    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(20, process)
        return

    frame = cv2.resize(frame, (900, 500))
    results = model.predict(frame, conf=0.55, device=device, verbose=False)

    car = bike = person = 0
    speeds = []
    new_tracks = {}
    curr_time = time.time()
    zebra_detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = refine_class(model.names[cls].lower(), conf, x2 - x1, y2 - y1)
            if not name:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = get_track_id(cx, cy)
            speed = 0

            if tid in tracks:
                px, py, last_time = tracks[tid]
                dt = max(curr_time - last_time, 0.001)
                raw_speed = math.hypot(cx - px, cy - py) / dt
                speed_memory[tid] = 0.7 * speed_memory.get(tid, raw_speed) + 0.3 * raw_speed
                speed = speed_memory[tid]
                speeds.append(speed)

            new_tracks[tid] = (cx, cy, curr_time)

            if name == "person" and abs(cy - ZEBRA_Y) < 70:
                zebra_detected = True

            if name == "car":
                car += 1
                color = (49, 214, 96)
                badge = GREEN
            elif name == "bike":
                bike += 1
                color = (255, 196, 61)
                badge = BLUE
            elif name == "person":
                person += 1
                color = (154, 108, 255)
                badge = PURPLE
            else:
                color = (210, 210, 210)
                badge = MUTED

            if speed > SPEED_LIMIT:
                color = (77, 77, 255)
                badge = RED
                save_violation(tid, speed, frame, name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name.upper()} | ID:{tid} | {int(speed)} km/h"
            draw_badge(frame, x1, y1, label_text, hex_to_bgr(badge))

    tracks = new_tracks

    if zebra_detected and time.time() - zebra_alert_time > ZEBRA_COOLDOWN:
        zebra_alert_time = time.time()
        threading.Thread(target=speak, args=("Attention. Zebra crossing ahead.",), daemon=True).start()

    update_dashboard(car, bike, person, speeds)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.imgtk = image
    video_label.configure(image=image)
    root.after(20, process)


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return b, g, r


def update_dashboard(car, bike, person, speeds):
    total = car + bike + person
    avg_speed = int(sum(speeds) / len(speeds)) if speeds else 0

    total_var.set(str(total))
    car_var.set(str(car))
    bike_var.set(str(bike))
    people_var.set(str(person))
    avg_speed_var.set(f"{avg_speed} km/h")
    fps_var.set(f"{cap.get(cv2.CAP_PROP_FPS):.1f}")
    time_var.set(time.strftime("%d %b %Y  |  %I:%M:%S %p"))

    traffic_history.append(total)
    if len(traffic_history) > 24:
        traffic_history.pop(0)

    draw_vehicle_chart(total, car, bike, person)
    draw_traffic_chart()


def draw_vehicle_chart(total, car, bike, person):
    count_canvas.delete("all")
    cx, cy, radius = 82, 84, 58
    values = [(car, GREEN), (bike, YELLOW), (person, PURPLE)]
    start = 90
    if total == 0:
        count_canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, outline=BORDER, width=18)
    else:
        for value, color in values:
            if value <= 0:
                continue
            extent = -360 * value / total
            count_canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius,
                                    start=start, extent=extent, style="arc", outline=color, width=18)
            start += extent
    count_canvas.create_text(cx, cy - 8, text=str(total), fill=TEXT, font=("Segoe UI", 20, "bold"))
    count_canvas.create_text(cx, cy + 18, text="Total", fill=MUTED, font=("Segoe UI", 10))

    legend = [("Cars", car, GREEN), ("Bikes", bike, YELLOW), ("People", person, PURPLE)]
    for i, (name, value, color) in enumerate(legend):
        y = 42 + i * 38
        pct = int((value / total) * 100) if total else 0
        count_canvas.create_rectangle(175, y, 187, y + 12, fill=color, outline=color)
        count_canvas.create_text(198, y + 6, anchor="w", text=name, fill=TEXT, font=("Segoe UI", 10))
        count_canvas.create_text(300, y + 6, anchor="e", text=f"{value} ({pct}%)", fill=MUTED, font=("Segoe UI", 10))


def draw_traffic_chart():
    traffic_canvas.delete("all")
    w, h = 360, 170
    left, top, right, bottom = 38, 18, w - 18, h - 28
    traffic_canvas.create_rectangle(left, top, right, bottom, outline=BORDER)
    for i in range(1, 5):
        y = top + (bottom - top) * i / 5
        traffic_canvas.create_line(left, y, right, y, fill="#183650")
    for i in range(1, 6):
        x = left + (right - left) * i / 6
        traffic_canvas.create_line(x, top, x, bottom, fill="#183650")

    values = traffic_history or [0]
    max_value = max(max(values), 5)
    step = (right - left) / max(len(values) - 1, 1)
    points = []
    for i, value in enumerate(values):
        x = left + i * step
        y = bottom - (value / max_value) * (bottom - top)
        points.extend([x, y])

    if len(points) >= 4:
        fill_points = [left, bottom] + points + [left + (len(values)-1)*step, bottom]
        traffic_canvas.create_polygon(fill_points, fill="#123f83", outline="")
        traffic_canvas.create_line(points, fill=BLUE, width=3, smooth=True)
    for i in range(0, len(points), 2):
        traffic_canvas.create_oval(points[i]-3, points[i+1]-3, points[i]+3, points[i+1]+3, fill=BLUE, outline=BLUE)
    traffic_canvas.create_text(left, bottom + 14, text="00:00", fill=MUTED, font=("Segoe UI", 8))
    traffic_canvas.create_text((left+right)/2, bottom + 14, text="12:00", fill=MUTED, font=("Segoe UI", 8))
    traffic_canvas.create_text(right, bottom + 14, text="24:00", fill=MUTED, font=("Segoe UI", 8))


def make_panel(parent, **grid_options):
    frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    frame.grid(**grid_options)
    return frame


def make_stat(parent, row, title, variable, color):
    box = tk.Frame(parent, bg=PANEL)
    box.grid(row=row, column=0, sticky="ew", padx=14, pady=8)
    tk.Label(box, text=title, fg=color, bg=PANEL, font=("Segoe UI", 10, "bold")).pack(anchor="w")
    tk.Label(box, textvariable=variable, fg=TEXT, bg=PANEL, font=("Segoe UI", 18, "bold")).pack(anchor="w")
    tk.Frame(parent, bg=BORDER, height=1).grid(row=row+1, column=0, sticky="ew", padx=14)


def create_ui():
    global root, video_label, total_var, car_var, bike_var, people_var, avg_speed_var
    global fps_var, status_var, status_dot, status_value, time_var
    global count_canvas, traffic_canvas, violations_list

    root = tk.Tk()
    root.title("Smart Traffic Monitoring System")
    root.geometry("1400x900")
    root.minsize(1180, 760)
    root.configure(bg=BG)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Nav.TButton", background="#0b376d", foreground=TEXT,
                    borderwidth=0, focuscolor="#0b376d", font=("Segoe UI", 11, "bold"), padding=12)
    style.map("Nav.TButton", background=[("active", "#145aa6")])

    header = tk.Frame(root, bg="#08223d", height=78, highlightbackground=BORDER, highlightthickness=1)
    header.pack(side="top", fill="x", padx=8, pady=(8, 0))
    header.pack_propagate(False)
    tk.Label(header, text="Smart Traffic Monitoring System", fg=TEXT, bg="#08223d",
             font=("Segoe UI", 18, "bold")).pack(side="left", padx=(28, 8), pady=(14, 0), anchor="n")
    tk.Label(header, text="Real-time Traffic Analytics & Vehicle Detection", fg=MUTED, bg="#08223d",
             font=("Segoe UI", 10)).place(x=29, y=48)
    time_var = tk.StringVar(value=time.strftime("%d %b %Y  |  %I:%M:%S %p"))
    tk.Label(header, textvariable=time_var, fg=TEXT, bg="#08223d", font=("Segoe UI", 11)).pack(side="right", padx=28)

    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=8, pady=8)

    nav = tk.Frame(body, bg="#071b30", width=190, highlightbackground=BORDER, highlightthickness=1)
    nav.pack(side="left", fill="y")
    nav.pack_propagate(False)
    for i, item in enumerate(["Dashboard", "Live Feed", "Analytics", "Vehicle Count", "Violations", "Reports", "Settings", "Logout"]):
        ttk.Button(nav, text=item, style="Nav.TButton").pack(fill="x", padx=12, pady=(18 if i == 0 else 6, 0))

    main = tk.Frame(body, bg=BG)
    main.pack(side="left", fill="both", expand=True, padx=(12, 0))
    main.grid_columnconfigure(0, weight=3)
    main.grid_columnconfigure(1, weight=1)
    main.grid_rowconfigure(1, weight=1)

    feed = make_panel(main, row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
    feed_header = tk.Frame(feed, bg=PANEL)
    feed_header.pack(fill="x", padx=16, pady=(12, 6))
    tk.Label(feed_header, text="Live Feed", fg=TEXT, bg=PANEL, font=("Segoe UI", 13, "bold")).pack(side="left")
    tk.Button(feed_header, text="Start", command=start_video, bg=GREEN, fg="#03110a",
              relief="flat", font=("Segoe UI", 9, "bold"), width=9).pack(side="right", padx=(6, 0))
    tk.Button(feed_header, text="Stop", command=stop_video, bg=RED, fg="white",
              relief="flat", font=("Segoe UI", 9, "bold"), width=9).pack(side="right")
    video_label = tk.Label(feed, bg="#020812", fg=MUTED, text="Loading video...",
                           width=900, height=500, font=("Segoe UI", 14, "bold"))
    video_label.pack(fill="both", expand=True, padx=14, pady=(0, 14))
    stats = make_panel(main, row=0, column=1, sticky="nsew", pady=(0, 12))
    tk.Label(stats, text="Live Stats", fg=TEXT, bg=PANEL, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 10))
    total_var = tk.StringVar(value="0")
    car_var = tk.StringVar(value="0")
    bike_var = tk.StringVar(value="0")
    people_var = tk.StringVar(value="0")
    avg_speed_var = tk.StringVar(value="0 km/h")
    make_stat(stats, 1, "Total Vehicles", total_var, BLUE)
    make_stat(stats, 3, "Cars", car_var, GREEN)
    make_stat(stats, 5, "Bikes", bike_var, YELLOW)
    make_stat(stats, 7, "People", people_var, PURPLE)
    make_stat(stats, 9, "Avg. Speed", avg_speed_var, RED)

    lower = tk.Frame(main, bg=BG)
    lower.grid(row=1, column=0, columnspan=2, sticky="nsew")
    lower.grid_columnconfigure(0, weight=1)
    lower.grid_columnconfigure(1, weight=1)
    lower.grid_columnconfigure(2, weight=1)

    vehicle_panel = make_panel(lower, row=0, column=0, sticky="nsew", padx=(0, 12))
    tk.Label(vehicle_panel, text="Vehicle Count (Today)", fg=TEXT, bg=PANEL, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 0))
    count_canvas = tk.Canvas(vehicle_panel, bg=PANEL, highlightthickness=0, height=180)
    count_canvas.pack(fill="both", expand=True, padx=8, pady=8)

    traffic_panel = make_panel(lower, row=0, column=1, sticky="nsew", padx=(0, 12))
    tk.Label(traffic_panel, text="Traffic Overview", fg=TEXT, bg=PANEL, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 0))
    traffic_canvas = tk.Canvas(traffic_panel, bg=PANEL, highlightthickness=0, height=180)
    traffic_canvas.pack(fill="both", expand=True, padx=8, pady=8)

    violation_panel = make_panel(lower, row=0, column=2, sticky="nsew")
    tk.Label(violation_panel, text="Recent Violations", fg=TEXT, bg=PANEL, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 0))
    violations_list = tk.Listbox(violation_panel, bg=PANEL, fg=TEXT, selectbackground=PANEL_2,
                                 borderwidth=0, highlightthickness=0, font=("Segoe UI", 10), height=8)
    violations_list.pack(fill="both", expand=True, padx=14, pady=10)

    control = tk.Frame(root, bg=BG)
    control.pack(fill="x", padx=210, pady=(0, 8))
    tk.Button(control, text="START", command=start_video, bg=GREEN, fg="#03110a",
              relief="flat", font=("Segoe UI", 11, "bold"), width=12).pack(side="left", padx=8)
    tk.Button(control, text="STOP", command=stop_video, bg=RED, fg="white",
              relief="flat", font=("Segoe UI", 11, "bold"), width=12).pack(side="left", padx=8)
    fps_var = tk.StringVar(value="0.0")
    status_var = tk.StringVar(value="System Offline")
    tk.Label(control, text="FPS:", fg=MUTED, bg=BG, font=("Segoe UI", 10)).pack(side="left", padx=(24, 4))
    tk.Label(control, textvariable=fps_var, fg=CYAN, bg=BG, font=("Segoe UI", 10, "bold")).pack(side="left")
    status_dot = tk.Label(control, text="*", fg=RED, bg=BG, font=("Segoe UI", 16, "bold"))
    status_dot.pack(side="right", padx=(4, 0))
    status_value = tk.Label(control, textvariable=status_var, fg=RED, bg=BG, font=("Segoe UI", 10, "bold"))
    status_value.pack(side="right")

    footer = tk.Frame(root, bg=BG)
    footer.pack(fill="x", padx=8, pady=(0, 8))
    tk.Label(footer, text="Under the Guidance of Sushma B R (CDS Department)", fg=CYAN, bg=BG,
             font=("Segoe UI", 11, "bold")).pack()
    tk.Label(footer, text="PROJECT TEAM", fg=CYAN, bg=BG, font=("Segoe UI", 12, "bold")).pack(pady=(4, 2))
    tk.Label(footer, text="Shreedhara M M     Aditya A K", fg=MUTED, bg=BG,
             font=("Segoe UI", 10)).pack()

    draw_vehicle_chart(0, 0, 0, 0)
    draw_traffic_chart()


def on_close():
    stop_video()
    root.destroy()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
ensure_model_file()
model = YOLO(MODEL_PATH)
try:
    model.to(device)
except Exception as e:
    print("Model.to(device) not supported, running on default.", e)

cap = None
running = False
tracks = {}
track_id = 0
speed_memory = {}
violation_memory = {}
traffic_history = []
zebra_alert_time = 0

speak_lock = threading.Lock()
try:
    engine = pyttsx3.init()
except Exception as e:
    engine = None
    print("Text-to-speech disabled:", e)

create_ui()
root.protocol("WM_DELETE_WINDOW", on_close)
root.after(800, start_video)
root.mainloop()