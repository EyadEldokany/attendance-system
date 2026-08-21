# Attendance System — Setup & User Guide

Face-recognition attendance tracking for Windows.  
The system connects to your surveillance camera, detects everyone who walks through the door, and automatically logs entry and exit times. On first run it learns faces on its own — no employee photos needed in advance.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [First-Time Installation](#2-first-time-installation)
3. [Camera Setup](#3-camera-setup)
4. [Door Line Calibration](#4-door-line-calibration)
4b. [Archive Mode — Processing a Folder of Old Recordings](#4b-archive-mode--processing-a-folder-of-old-recordings)
5. [Running the Application](#5-running-the-application)
6. [Using the GUI](#6-using-the-gui)
7. [Enrolling Employees](#7-enrolling-employees)
8. [Reviewing Attendance](#8-reviewing-attendance)
9. [Excel Export](#9-excel-export)
10. [Auto-Start on Windows Boot](#10-auto-start-on-windows-boot)
11. [Settings Reference](#11-settings-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Confidence Thresholds Explained](#13-confidence-thresholds-explained)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Windows 10 64-bit | Windows 10/11 64-bit |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU mode) | NVIDIA GPU (CUDA) |
| Storage | 10 GB free | 20 GB free |
| Network | LAN access to camera | LAN access to camera |
| Camera | Any RTSP-capable IP camera | 1080p RTSP camera |

> **No internet required** after first installation. The system runs fully offline.

---

## 2. First-Time Installation

### Option A — Pre-built executable (recommended for clients)

1. Copy the `AttendanceSystem` folder to the client machine (e.g. `C:\AttendanceSystem\`).
2. Copy the `~/.insightface` folder from the developer machine to the same location on the client:  
   `C:\Users\<username>\.insightface\`  
   *(This contains the face recognition model — ~300 MB — so no internet download is needed.)*
3. **Install Tesseract-OCR** (needed for archive-mode timestamp reading — see note below), then double-click `AttendanceSystem.exe` to launch.

### Option B — Run from source (developer/IT setup)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install insightface==0.7.3

# 3. Launch
python app.py
```

### Installing Tesseract-OCR (required for Archive Mode)

`pytesseract` (in requirements.txt) is just a thin Python wrapper — it needs the actual **Tesseract-OCR engine** installed separately on the Windows machine:

1. Download the Windows installer: https://github.com/UB-Mannheim/tesseract/wiki (look for the `tesseract-ocr-w64-setup-*.exe` link).
2. Run the installer (default install path is fine: `C:\Program Files\Tesseract-OCR\`).
3. This is a one-time install — after that it works fully offline, no internet needed for actual use.
4. If `pytesseract` can't find it automatically, add this near the top of `main.py`/`app.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

---

## 3. Camera Setup

Open `config.yaml` in any text editor and set your camera's RTSP address:

```yaml
camera:
  source: "rtsp://admin:password@192.168.1.64:554/stream1"
```

Replace `admin`, `password`, `192.168.1.64`, and the path with your camera's actual values.  
You can find the RTSP address in your camera's manual or its web interface.

**Common RTSP formats by brand:**

| Brand | Typical RTSP URL |
|-------|-----------------|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Generic | `rtsp://user:pass@IP:554/stream1` |

> To test the URL, paste it into **VLC Media Player → Media → Open Network Stream**.

---

## 4. Door Line Calibration

This step tells the system exactly where the door is and which direction counts as "entry".  
**Run once during installation.**

```powershell
python setup_door_line.py
```

**Step-by-step:**

1. A live camera feed window opens.
2. **Click two points** along the door line (e.g. click the left edge of the doorway, then the right edge).
3. When prompted, **walk through the door from outside to inside** in front of the camera.
4. The system records the direction and saves it to `config.yaml` automatically.
5. Close the window.

After calibration `config.yaml` will contain the correct `door_line` values — do not edit them manually.

For **archive mode** (no live camera connected), run this instead with a sample video from your archive folder:
```powershell
python setup_door_line.py "D:\archive\sample_video.mp4"
```
Since you can't physically walk through the door in a recorded video, the tool plays through the footage and pauses at the first detected crossing — watch that clip and press `i` (entry) or `o` (exit) to calibrate the direction correctly.

---

## 4b. Archive Mode — Processing a Folder of Old Recordings

This is the current mode for this deployment: instead of watching a live camera, the system analyzes a folder of already-recorded footage (e.g. a year of archived video) and reconstructs the attendance log from it.

### Why this needs a quick manual step per video

Old recordings usually aren't named with their recording date, so the system reads the **date/time burned into the video image itself** (the on-screen timestamp overlay most CCTV systems stamp on every frame).

**Important, tested finding:** on this footage, that overlay uses a digital/seven-segment-style font over a semi-transparent box whose contrast changes depending on what's behind it (a dark doorway vs. a bright marble floor, for example). Automatic OCR (even after several image-processing attempts) reads it correctly *most* of the time but not reliably *every* time — occasionally misreading a digit.

Rather than silently trusting OCR and risking wrong dates in the final log, the system uses a smarter design:

- It only needs **one confirmed timestamp per video file**, not one per event.
- Since a camera records continuously at a steady frame rate, once you confirm (or correct) the *start* time of a video, the system calculates every subsequent event's exact timestamp mathematically from that one reference point (`start_time + frame_number ÷ frame_rate`) — no repeated OCR guessing required.
- This turns a full year of footage into roughly **one quick visual check per video file** (glance at the cropped date/time image, accept the OCR's guess or type the correct one), which is both faster and far more reliable than hoping automatic OCR gets every single event right.

### Step-by-step

1. **Calibrate the timestamp region once** (tells the system where on the screen the date/time text appears):
   ```powershell
   python setup_timestamp_region.py "D:\archive\sample_video.mp4"
   ```
   Draw a box around the burned-in date/time text, press `t` to test the OCR read, adjust the box if needed, then press `s` to save.

2. **Open the app and go to Archive → Process Archive Footage…** (or click the "Process Archive Footage" button on the main screen).

3. **Scan Folder tab:** browse to the folder containing the recordings and click "Start Scan". The tool finds every video file and attempts an automatic timestamp read for each one. This can take a while for a large archive but only needs to run once — re-scanning the same folder later skips files already registered.

4. **Confirm & Process tab:** any video the system wasn't fully confident about appears in the list. Select one, look at the cropped timestamp image and the OCR's guess, correct the date/time field if needed, and click "Confirm this video". (High-confidence reads are confirmed automatically and won't appear here.)

5. Once videos are confirmed, click **"Start Processing Confirmed Videos"**. The system runs detection, tracking, door-crossing, and face recognition across each video, computing accurate real-world timestamps for every entry/exit event, same as the live pipeline.

6. Review low-confidence face matches and label new employees exactly as described in section 7–8 below — this part works identically whether events came from a live camera or archived footage.

---

## 5. Running the Application

**From the executable:**
```
Double-click AttendanceSystem.exe
```

**From source:**
```powershell
python app.py
```

The main window opens:

```
┌──────────────────────────────────────────────────────────┐
│  Attendance System                        [_] [□] [×]   │
├────────────────────────────┬─────────────────────────────┤
│                            │  Today's Attendance         │
│   [ LIVE CAMERA FEED ]     │─────────────────────────────│
│                            │  08:02  Ali Hassan    IN ✓  │
│   Green boxes = people     │  08:15  Sara Ahmed    IN ✓  │
│   White line  = door       │  08:31  Unknown   review ⚠  │
│                            │─────────────────────────────│
│                            │  [Review Faces]             │
│                            │  [Manage Employees]         │
├────────────────────────────┴─────────────────────────────┤
│  Camera: Connected  |  Employees: 12  |  Last: 08:31     │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Using the GUI

### Menu Bar

| Menu | Item | What it does |
|------|------|--------------|
| File | Settings | Edit camera URL, thresholds, export path |
| File | Export Excel now | Immediately write all events to the Excel file |
| Help | About | Version information |

### Live Feed Panel (left)
- Shows the camera stream in real time.
- **Green boxes** appear around every detected person.
- **White line** marks the configured door line.
- Boxes flash blue (ENTRY) or red (EXIT) when a crossing is detected.

### Attendance Log Panel (right)
- Lists all events for today, newest first.
- Updates automatically every 15 seconds and instantly on each new event.
- Status colours: **green** = confirmed, **orange** = needs review, **red** = rejected.

### Status Bar (bottom)
- **Camera** — green = connected, red = disconnected (auto-reconnects).
- **Employees** — total registered employees.
- **Last event** — time of the most recent crossing.

---

## 7. Enrolling Employees

The system learns faces automatically — **no photos needed in advance**.

**How it works:**
1. A new person walks through the door. The system detects them and saves their face.
2. After they appear **3 or more times**, they show up in the **Review Faces** dialog.
3. The supervisor opens Review Faces, sees the person's photo, types their name, and clicks **Save as new employee**.
4. From that moment on, the system recognises that person automatically.

**To open Review Faces:** Click the **[Review Faces]** button in the main window.

### New Faces Tab

| Column | Meaning |
|--------|---------|
| Track # | Internal ID assigned to that person's movement |
| Seen X× | How many times they have crossed the door |
| First seen | Date and time of first detection |

**Actions:**
- **Save as new employee** — type a name in the box and click. Creates a new employee record.
- **Add to existing employee** — select an existing name from the dropdown. Adds this face sample to their profile (useful if they were already enrolled but not recognised in this instance).
- **Ignore** — marks as "not an employee" (e.g. a delivery person). They will not be prompted again.

---

## 8. Reviewing Attendance

Low-confidence events (face detected but not certain enough to confirm automatically) go into the **Review Attendance** tab.

**To open:** Click **[Review Faces]** → select the **Review Attendance** tab.

Each row shows:
- Time and direction (IN / OUT)
- Confidence score
- Face photo (if captured)

**Actions per row:**
- **Assign** — select which employee this event belongs to. The record is updated and marked confirmed.
- **Reject** — mark the event as rejected (e.g. a false detection). It is removed from the pending list but kept in the database.

Click **Refresh** to reload the list after making changes.

---

## 9. Excel Export

The attendance log is automatically exported to:
```
exports\attendance_log.xlsx
```

The file updates **automatically after every crossing event**.  
If the file is open in Excel when an update happens, the system waits and retries — no data is lost.

**To export manually:** File → Export Excel now.

**To change the export location:** File → Settings → Excel Export path.

The Excel file contains one sheet with columns:

| Column | Content |
|--------|---------|
| Event ID | Unique record number |
| Employee Name | Name or "Unknown" |
| Event Type | IN or OUT |
| Timestamp | Date and time |
| Confidence | Match score (0–1) |
| Status | confirmed / needs_review / rejected |

---

## 10. Auto-Start on Windows Boot

To make the application start automatically whenever the computer turns on:

1. Make sure the app has been built (`dist\AttendanceSystem\AttendanceSystem.exe` exists).
2. Run `install_startup.bat` **as Administrator** (right-click → Run as administrator).
3. Done. The app will launch automatically at every Windows login.

**To remove auto-start:**
```powershell
schtasks /delete /tn "AttendanceSystem" /f
```

---

## 11. Settings Reference

Open via **File → Settings**.

| Setting | Default | Description |
|---------|---------|-------------|
| RTSP URL | — | Camera stream address |
| Process every N frames | 3 | Skip frames for performance (higher = faster but less precise) |
| Confidence threshold | 0.5 | Minimum detection confidence for person detection |
| Match threshold | 0.62 | Above this = confirmed identity |
| Review threshold | 0.45 | Between 0.45–0.62 = needs manual review |
| Min face size (px) | 40 | Ignore faces smaller than this (avoids distant/blurry detections) |
| Excel export path | exports/attendance_log.xlsx | Where to write the Excel file |

Click **Save** to apply. Camera URL changes require an application restart.

---

## 12. Troubleshooting

### Camera shows "Connecting…" and never connects
- Confirm the RTSP URL is correct — test it in VLC first.
- Ensure the PC and camera are on the same network.
- Check that no firewall is blocking port 554.
- Verify the username and password in the URL are correct.

### "Unknown" appears even for known employees
- The lighting may have changed significantly — open **Review Faces** and re-label.
- Lower `match_threshold` slightly (e.g. to 0.58) in Settings if false unknowns are frequent.

### Crossings are logged in the wrong direction (entry logged as exit)
- Re-run door line calibration: `python setup_door_line.py`
- Walk through the door from **outside to inside** when prompted.

### The Excel file is not updating
- Check that `excel_sync.enabled` is `true` in `config.yaml`.
- The file updates only when someone crosses. Use **File → Export Excel now** to force an update.
- Make sure another process does not have the file open and locked indefinitely.

### The app crashes on startup
- Check `logs\attendance_system.log` for the error message.
- Make sure the `models\yolov8n.pt` file exists.
- Make sure the `.insightface\models\buffalo_l\` folder exists in the user's home directory.

### High CPU usage
- Increase `process_every_n_frames` in Settings (e.g. from 3 to 5).
- Set `detection.device` to `"cuda"` in `config.yaml` if an NVIDIA GPU is available.

---

## 13. Confidence Thresholds Explained

When someone crosses the door, the system compares their face against all enrolled employees and produces a **confidence score** between 0 and 1.

| Score range | Meaning | What happens |
|-------------|---------|--------------|
| ≥ 0.62 | Strong match | Event logged as **confirmed** with employee name |
| 0.45 – 0.62 | Possible match | Event logged as **needs review** — supervisor reviews manually |
| < 0.45 | No match | Logged as unknown, face saved for labeling |

**Tuning advice:**  
During the first two weeks of real use, monitor how many events fall into "needs review". If there are too many false unknowns for known employees, lower `match_threshold` to 0.58. If known employees are being confused with each other, raise it to 0.65.

---

## File Structure Reference

```
AttendanceSystem\
├── AttendanceSystem.exe      ← Launch this
├── config.yaml               ← All settings (edit with Notepad)
├── _internal\                ← Application dependencies (do not modify)
├── models\
│   └── yolov8n.pt            ← Person detection model
├── data\
│   └── attendance.db         ← Database (backed up automatically)
├── exports\
│   └── attendance_log.xlsx   ← Excel attendance log
├── enrollment_faces\         ← Face photos captured for labeling
├── logs\
│   └── attendance_system.log ← System log (check here if something goes wrong)
└── assets\
```

> **Backup recommendation:** Copy the `data\` and `exports\` folders to a network drive or USB weekly.
