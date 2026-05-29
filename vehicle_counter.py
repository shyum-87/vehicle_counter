"""
vehicle_counter.py
===================

This script implements a real‑time vehicle detection, tracking and counting
system using the Ultralytics YOLO object detector and the Roboflow
Supervision library.  The program is designed for top‑down CCTV
installations with multiple lanes and counts vehicles as they enter or
exit the scene by crossing a horizontal counting line.  Counts are
maintained separately for each lane and vehicle type (car, bus, truck).

Key features
------------

* **Object detection** – Uses a pretrained YOLO model to detect vehicles
  in each frame.  The default classes used for counting correspond to
  COCO class indices for cars (2), buses (5) and trucks (7).
* **Multi‑object tracking** – Integrates the ByteTrack tracker from
  the Supervision library to assign persistent IDs to detected
  objects and ensure each vehicle is counted only once.
* **Direction and lane assignment** – Computes the vertical position
  of each tracked object across frames to determine whether it is
  moving upwards (entering) or downwards (exiting) across a
  configurable counting line.  The horizontal centre of the bounding
  box is used to assign the object to one of two lanes.
* **Database/API integration** – Provides placeholder functions to
  persist counts to a relational database via SQLAlchemy and send
  real‑time count updates to an external API via HTTP POST.  These
  functions are designed to be customised for your specific
  infrastructure.

Usage example
-------------

Run the script from the command line with arguments for your video
source (file path or RTSP URL), the path to the YOLO weights and
configuration for the counting line and lane divider:

```
python vehicle_counter.py \
    --source rtsp://username:password@192.168.1.10:554/stream1 \
    --weights yolov8s.pt \
    --y-line-ratio 0.5 \
    --lane-divider-ratio 0.5 \
    --db-url postgresql://user:pass@localhost/traffic_db \
    --api-url https://example.com/traffic/update
```

The `--y-line-ratio` option specifies the vertical position of the
counting line as a fraction of the frame height (0.0 = top,
1.0 = bottom).  The `--lane-divider-ratio` option sets the
horizontal position of the lane divider as a fraction of the frame
width (0.0 = left edge, 1.0 = right edge).  Adjust these values to
match your camera’s geometry.

Notes
-----

This script depends on the `ultralytics` and `supervision` Python
packages, which may not be installed in some environments.  To run
the script on your own machine, install the required packages with:

```
pip install ultralytics supervision opencv-python sqlalchemy requests
```

Then launch the script with the appropriate arguments.  Modify the
`save_to_db` and `send_to_api` functions to integrate with your
database and API.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import importlib
import logging
import os
import platform
import sys
from typing import Any, Dict, Optional


cv2: Any = None
np: Any = None
requests: Any = None
mss: Any = None
create_engine: Any = None
SQLAlchemyError: Any = Exception
APP_VERSION = "2026-05-29-check-deps-source-fix"


class DependencyError(RuntimeError):
    """Raised when a required runtime dependency cannot be loaded."""


def _missing_dependency_message(error: ImportError) -> str:
    """Format import errors, including native-library load failures."""
    missing_pkg = str(error).split("'")[1] if "'" in str(error) else str(error)
    return (
        f"Missing required package or native library: {missing_pkg}. Please install "
        "the dependencies listed in the module documentation before running this script."
    )


def load_core_dependencies(logger: Optional[logging.Logger] = None) -> bool:
    """Load runtime dependencies that are not needed for --check-env."""
    global cv2, np, requests, mss, create_engine, SQLAlchemyError
    if all(value is not None for value in [cv2, np, requests, mss, create_engine]):
        return True

    try:
        cv2_module = importlib.import_module("cv2")
        np_module = importlib.import_module("numpy")
        requests_module = importlib.import_module("requests")
        mss_module = importlib.import_module("mss")
        sqlalchemy_module = importlib.import_module("sqlalchemy")
        sqlalchemy_exc_module = importlib.import_module("sqlalchemy.exc")
    except ImportError as e:  # pragma: no cover
        message = _missing_dependency_message(e)
        if logger:
            logger.error(message)
        else:
            print(message, file=sys.stderr)
        return False

    cv2 = cv2_module
    np = np_module
    requests = requests_module
    mss = mss_module
    create_engine = sqlalchemy_module.create_engine
    SQLAlchemyError = sqlalchemy_exc_module.SQLAlchemyError
    return True


def get_runtime_environment() -> Dict[str, str]:
    """Return a small runtime summary for troubleshooting user environments."""
    return {
        "app_version": APP_VERSION,
        "script_path": os.path.abspath(__file__),
        "os_name": platform.system() or "unknown",
        "is_windows": str(os.name == "nt" or sys.platform.startswith("win")),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_architecture": platform.architecture()[0],
        "python_executable": sys.executable,
    }


def format_runtime_environment() -> str:
    """Format runtime environment details as a single log-friendly line."""
    env = get_runtime_environment()
    return (
        f"version={env['app_version']}, script={env['script_path']}, "
        f"os={env['os_name']}, is_windows={env['is_windows']}, "
        f"platform={env['platform']}, python={env['python_version']} "
        f"({env['python_architecture']}), executable={env['python_executable']}"
    )


def print_runtime_environment() -> None:
    """Print runtime environment details without starting capture/model loading."""
    env = get_runtime_environment()
    print("Runtime environment")
    print("===================")
    print(f"App version: {env['app_version']}")
    print(f"Script: {env['script_path']}")
    print(f"OS name: {env['os_name']}")
    print(f"Is Windows: {env['is_windows']}")
    print(f"Platform: {env['platform']}")
    print(f"Python: {env['python_version']} ({env['python_architecture']})")
    print(f"Executable: {env['python_executable']}")


def format_torch_dll_help(error: OSError) -> str:
    """Return actionable guidance for common Windows PyTorch DLL failures."""
    details = str(error)
    guidance = [
        "Failed to initialize PyTorch/Ultralytics native DLLs.",
        f"Original error: {details}",
        f"Detected runtime: {format_runtime_environment()}",
    ]
    if "WinError 1114" in details or "c10.dll" in details:
        guidance.extend([
            "This is a PyTorch native DLL initialization failure, not a screen "
            "capture coordinate or ROI selection problem.",
            "Recommended Windows checks:",
            "  1. Reinstall torch/torchvision for your target compute platform "
            "(CPU-only is the safest first test).",
            "  2. If you intend to use NVIDIA GPU, update the NVIDIA driver and "
            "install a torch/torchvision build matching that CUDA runtime.",
            "  3. Install or repair Microsoft Visual C++ Redistributable 2015-2022 "
            "(x64), then recreate the virtual environment if needed.",
            "Next diagnostic command: python vehicle_counter.py --check-deps",
        ])
    return "\n".join(guidance)


def _module_version(module: Any) -> str:
    """Return a module version string when available."""
    return str(getattr(module, "__version__", "unknown"))


def print_ml_dependency_check() -> bool:
    """Import ML/runtime packages one by one and print a troubleshooting report."""
    print_runtime_environment()
    print()
    print("Dependency check")
    print("================")

    ok = True
    for module_name in ["cv2", "numpy", "mss", "sqlalchemy", "requests"]:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            ok = False
            print(f"[FAIL] {module_name}: {_missing_dependency_message(exc)}")
        else:
            print(f"[ OK ] {module_name}: {_module_version(module)}")

    torch = None
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        ok = False
        print(f"[FAIL] torch: {_missing_dependency_message(exc)}")
    except OSError as exc:
        ok = False
        print(f"[FAIL] torch:\n{format_torch_dll_help(exc)}")
    else:
        print(f"[ OK ] torch: {_module_version(torch)}")
        print(f"       torch path: {getattr(torch, '__file__', 'unknown')}")
        print(f"       torch CUDA build: {getattr(torch.version, 'cuda', None)}")
        print(f"       CUDA available: {torch.cuda.is_available()}")

    for module_name in ["torchvision", "ultralytics", "supervision"]:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            ok = False
            print(f"[FAIL] {module_name}: {_missing_dependency_message(exc)}")
        except OSError as exc:
            ok = False
            print(f"[FAIL] {module_name}:\n{format_torch_dll_help(exc)}")
        else:
            print(f"[ OK ] {module_name}: {_module_version(module)}")

    if ok:
        print("\nAll checked dependencies imported successfully.")
    else:
        print("\nOne or more dependencies failed to import. Fix these before running capture.")
    return ok


def load_zones(path: str) -> Dict:
    """Load zone configuration from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert point lists to numpy arrays for cv2.pointPolygonTest
    for key, zone in data["zones"].items():
        zone["polygon"] = np.array(zone["points"], dtype=np.int32)
    return data


def point_in_zone(px: float, py: float, polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon zone."""
    return cv2.pointPolygonTest(polygon, (px, py), False) >= 0


class VehicleCounter:
    """Real‑time vehicle detection, tracking and counting system."""

    # COCO class IDs for vehicles of interest
    VEHICLE_CLASSES = {
        2: "car",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        source: str,
        weights: str,
        zones_path: str = "zones.json",
        db_url: Optional[str] = None,
        api_url: Optional[str] = None,
        screen_region: Optional[Dict[str, int]] = None,
    ) -> None:
        self.source = source
        self.weights = weights
        self.db_url = db_url
        self.api_url = api_url
        self.screen_region = screen_region

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Runtime environment: %s", format_runtime_environment())
        if not load_core_dependencies(self.logger):
            raise DependencyError("Failed to load required runtime dependencies")

        # Load zone configuration
        if not os.path.isfile(zones_path):
            self.logger.error(
                "Zone config not found: %s. Run zone_config.py first.", zones_path
            )
            raise FileNotFoundError(f"Zone config not found: {zones_path}")
        self.zones_data = load_zones(zones_path)
        self.zones = self.zones_data["zones"]
        self.logger.info("Loaded %d zones from %s", len(self.zones), zones_path)

        # Validate weights file exists locally (for air-gapped environments)
        if not os.path.isfile(weights):
            self.logger.error(
                "YOLO weights file not found: %s. "
                "In air-gapped environments, download the weights file in advance "
                "and provide the local path via --weights.",
                weights,
            )
            raise FileNotFoundError(f"Weights file not found: {weights}")

        # YOLO, PyTorch and Supervision are loaded lazily after screen-region
        # selection so users can choose a monitor region before any heavy model
        # dependency is initialized.
        self.model = None
        self.device: Any = "cpu"
        self.tracker = None
        self.sv = None

        # Database engine (initialised lazily)
        self.db_engine = None
        if self.db_url:
            try:
                self.db_engine = create_engine(self.db_url)
                with self.db_engine.connect() as connection:
                    pass
                self.logger.info("Connected to database: %s", self.db_url)
            except SQLAlchemyError as e:
                self.logger.error("Failed to connect to DB: %s", e)
                self.db_engine = None

        # Zone-based counting state
        # Counts vehicles entering road_3 by direction and type
        self.counts = {
            "total": 0,
            "by_type": {"car": 0, "bus": 0, "truck": 0},
            "left_turn": {"total": 0, "car": 0, "bus": 0, "truck": 0},   # from road_11
            "right_turn": {"total": 0, "car": 0, "bus": 0, "truck": 0},  # from road_7
            "straight": {"total": 0, "car": 0, "bus": 0, "truck": 0},    # other origin
        }
        # Per-object tracking: origin zone, whether counted, vehicle type
        self.object_info: Dict[int, Dict[str, object]] = {}

        # Hourly counts for bar chart: {hour_str: {"car": n, "bus": n, "truck": n}}
        self.hourly_counts: Dict[str, Dict[str, int]] = {}

        # CSV logging
        self.csv_path = os.path.splitext(zones_path)[0] + "_counts.csv"
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "direction", "vehicle_type",
            "car_count", "bus_count", "truck_count", "total_count",
        ])
        self._closed = False
        self.logger.info("CSV log: %s", self.csv_path)

    def _cleanup(self, capture=None, is_screen: bool = False) -> None:
        """Close opened resources before the program exits."""
        if capture is not None:
            if is_screen:
                sct, _ = capture
                sct.close()
            else:
                capture.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        if not self._closed:
            self._csv_file.close()
            self._closed = True
            self.logger.info(
                "CSV log saved: %s (%d records)",
                self.csv_path, self.counts["total"],
            )

    def save_to_db(self, timestamp: datetime.datetime, lane: str, direction: str, vehicle_type: str) -> None:
        """Persist a count event to the configured database.

        This method uses SQLAlchemy to execute a simple INSERT into a
        table called ``vehicle_counts``.  Adjust the table name and
        schema according to your database design.
        """
        if not self.db_engine:
            return
        try:
            with self.db_engine.begin() as connection:
                connection.execute(
                    """
                    INSERT INTO vehicle_counts (timestamp, lane, direction, vehicle_type)
                    VALUES (:timestamp, :lane, :direction, :vehicle_type)
                    """,
                    {
                        "timestamp": timestamp,
                        "lane": lane,
                        "direction": direction,
                        "vehicle_type": vehicle_type,
                    },
                )
        except SQLAlchemyError as e:
            self.logger.error("DB insert failed: %s", e)

    def send_to_api(self, data: Dict[str, object]) -> None:
        """Send a count event to an external API as JSON.

        The API endpoint must be configured via the ``--api-url``
        argument.  Errors are logged but do not terminate the program.
        """
        if not self.api_url:
            return
        try:
            response = requests.post(self.api_url, json=data, timeout=2)
            if response.status_code != 200:
                self.logger.warning(
                    "API call returned status %s: %s", response.status_code, response.text
                )
        except requests.RequestException as e:
            self.logger.error("Failed to send API request: %s", e)

    def _draw_bar_chart(self, frame: np.ndarray) -> None:
        """Draw hourly stacked bar chart on the bottom-right of the frame."""
        if not self.hourly_counts:
            return

        # Chart dimensions
        chart_w = 380
        bar_max_h = 140
        bar_w = 28
        gap = 6
        padding = 10
        hours = sorted(self.hourly_counts.keys())
        # Show last 8 hours max
        if len(hours) > 8:
            hours = hours[-8:]

        num_bars = len(hours)
        chart_actual_w = num_bars * (bar_w + gap) + padding * 2
        chart_h = bar_max_h + 50  # space for labels
        chart_actual_w = max(chart_actual_w, 120)

        # Position: bottom-right
        x0 = frame.shape[1] - chart_actual_w - 10
        y0 = frame.shape[0] - chart_h - 10

        # Background
        cv2.rectangle(frame, (x0, y0), (x0 + chart_actual_w, y0 + chart_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + chart_actual_w, y0 + chart_h), (80, 80, 80), 1)

        # Title
        cv2.putText(frame, "Hourly Count", (x0 + 5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Find max count for scaling
        max_count = max(
            sum(self.hourly_counts[h].values()) for h in hours
        )
        if max_count == 0:
            max_count = 1

        type_colors = {"car": (0, 200, 0), "bus": (0, 180, 255), "truck": (255, 100, 100)}
        bar_base_y = y0 + chart_h - 25  # bottom of bars, leave room for labels

        for i, hour in enumerate(hours):
            bx = x0 + padding + i * (bar_w + gap)
            counts = self.hourly_counts[hour]
            total = sum(counts.values())

            # Draw stacked bars bottom-up
            cur_y = bar_base_y
            for vtype in ["truck", "bus", "car"]:
                cnt = counts.get(vtype, 0)
                if cnt == 0:
                    continue
                seg_h = max(int((cnt / max_count) * bar_max_h), 3)
                color = type_colors[vtype]
                cv2.rectangle(frame, (bx, cur_y - seg_h), (bx + bar_w, cur_y), color, -1)
                # Count label on segment
                if seg_h >= 12:
                    cv2.putText(frame, str(cnt), (bx + 2, cur_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cur_y -= seg_h

            # Total on top of bar
            total_y = bar_base_y - max(int((total / max_count) * bar_max_h), 3) - 5
            cv2.putText(frame, str(total), (bx, max(total_y, y0 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Hour label below bar
            cv2.putText(frame, hour, (bx - 2, bar_base_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        # Legend
        legend_y = y0 + chart_h - 8
        lx = x0 + padding
        for vtype, color in type_colors.items():
            cv2.rectangle(frame, (lx, legend_y - 8), (lx + 10, legend_y), color, -1)
            cv2.putText(frame, vtype, (lx + 13, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            lx += 55

    @staticmethod
    def _torch_dll_help(error: OSError) -> str:
        """Return actionable guidance for common Windows PyTorch DLL failures."""
        return format_torch_dll_help(error)

    def _load_model_and_tracker(self) -> bool:
        """Load YOLO/PyTorch/Supervision after the capture source is ready."""
        if self.model is not None and self.tracker is not None:
            return True

        try:
            ultralytics = importlib.import_module("ultralytics")
            sv = importlib.import_module("supervision")
            torch = importlib.import_module("torch")
        except ImportError as e:  # pragma: no cover
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
            self.logger.error(
                "Missing required package: %s. Please install dependencies from requirements.txt.",
                missing_pkg,
            )
            return False
        except OSError as e:  # pragma: no cover
            self.logger.error("%s", self._torch_dll_help(e))
            return False

        self.logger.info("Loading YOLO model from %s", self.weights)
        try:
            self.model = ultralytics.YOLO(self.weights)
        except OSError as e:  # pragma: no cover
            self.logger.error("%s", self._torch_dll_help(e))
            return False

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.logger.info("Using device: %s", self.device)
        self.sv = sv
        self.tracker = sv.ByteTrack()
        return True

    @staticmethod
    def _select_screen_region(sct) -> Optional[Dict[str, int]]:
        """Show full screen capture and let user drag-select a region."""
        full_monitor = sct.monitors[1]
        screenshot = sct.grab(full_monitor)
        full_frame = np.array(screenshot)[:, :, :3].copy()

        # Resize for selection UI (full res may be too large)
        max_h = 900
        scale = max_h / full_frame.shape[0]
        sel_w = int(full_frame.shape[1] * scale)
        sel_frame = cv2.resize(full_frame, (sel_w, max_h))

        cv2.putText(sel_frame, "Drag to select capture region, then press ENTER or SPACE",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        roi = cv2.selectROI("Select Region", sel_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Region")

        x, y, w, h = roi
        if w == 0 or h == 0:
            return None

        # Scale back to actual screen coordinates
        monitor = {
            "top": full_monitor["top"] + int(y / scale),
            "left": full_monitor["left"] + int(x / scale),
            "width": int(w / scale),
            "height": int(h / scale),
        }
        return monitor

    def _open_source(self):
        """Open video source or screen capture. Returns (capture, is_screen) tuple."""
        if self.source == "screen":
            sct = mss.MSS()
            if self.screen_region:
                monitor = self.screen_region
            else:
                # Let user drag-select a region
                self.logger.info("Select capture region by dragging on screen...")
                monitor = self._select_screen_region(sct)
                if monitor is None:
                    self.logger.error("No region selected.")
                    sct.close()
                    return None, False
            self.logger.info(
                "Screen capture: top=%d, left=%d, width=%d, height=%d",
                monitor["top"], monitor["left"], monitor["width"], monitor["height"],
            )
            return (sct, monitor), True

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error("Failed to open video source: %s", self.source)
            return None, False
        self.logger.info("Processing video stream: %s", self.source)
        return cap, False

    def _read_frame(self, capture, is_screen):
        """Read a single frame from video or screen capture."""
        if is_screen:
            sct, monitor = capture
            screenshot = sct.grab(monitor)
            # mss returns BGRA, convert to BGR for OpenCV
            frame = np.array(screenshot)[:, :, :3].copy()
            return True, frame
        else:
            return capture.read()

    def run(self) -> None:
        """Start the counting loop."""
        capture, is_screen = self._open_source()
        if capture is None:
            return

        if not self._load_model_and_tracker():
            self.logger.error(
                "Model/tracker initialization failed. Run `python vehicle_counter.py --check-deps` "
                "to identify the failing dependency before starting screen capture."
            )
            self._cleanup(capture, is_screen)
            return

        if is_screen:
            _, monitor = capture
            frame_width = monitor["width"]
            frame_height = monitor["height"]
        else:
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width == 0 or frame_height == 0:
                frame_width, frame_height = None, None

        # Scale zones if frame size differs from zone config image size
        zone_img_w, zone_img_h = self.zones_data["image_size"]
        self._scaled_zones = {}
        for key, zone in self.zones.items():
            if frame_width and frame_height and (zone_img_w != frame_width or zone_img_h != frame_height):
                scale_x = frame_width / zone_img_w
                scale_y = frame_height / zone_img_h
                scaled_pts = (zone["polygon"].astype(np.float64) * [scale_x, scale_y]).astype(np.int32)
            else:
                scaled_pts = zone["polygon"]
            self._scaled_zones[key] = {"type": zone["type"], "polygon": scaled_pts}

        # Main processing loop
        while True:
            ret, frame = self._read_frame(capture, is_screen)
            if not ret:
                break

            if frame_width is None or frame_height is None:
                frame_height, frame_width = frame.shape[:2]

            # Run YOLO detection
            results = self.model.predict(frame, device=self.device, verbose=False)
            result = results[0]

            # Convert to Supervision Detections and filter vehicle classes
            detections = self.sv.Detections.from_ultralytics(result)
            mask = np.isin(detections.class_id, list(self.VEHICLE_CLASSES.keys()))
            detections = detections[mask]

            # Update tracker
            tracked = self.tracker.update_with_detections(detections)

            # Process each tracked detection
            for bbox, track_id, class_id in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id
            ):
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                vehicle_type = self.VEHICLE_CLASSES.get(int(class_id), "unknown")

                # Initialise object info if new
                info = self.object_info.setdefault(
                    int(track_id),
                    {
                        "origin_zone": None,
                        "counted": False,
                        "vehicle_type": vehicle_type,
                    },
                )
                info["vehicle_type"] = vehicle_type

                # Determine which zone the vehicle is currently in
                current_zone = None
                for zone_key, zone_data in self._scaled_zones.items():
                    if point_in_zone(cx, cy, zone_data["polygon"]):
                        current_zone = zone_key
                        break

                # Record origin zone (first zone the vehicle appears in)
                if current_zone and info["origin_zone"] is None and current_zone != "road_3":
                    info["origin_zone"] = current_zone

                # Count when vehicle enters road_3 entry zone
                if current_zone == "road_3" and not info["counted"]:
                    info["counted"] = True
                    origin = info["origin_zone"]
                    vtype = info["vehicle_type"]

                    # Total count
                    self.counts["total"] += 1
                    if vtype in self.counts["by_type"]:
                        self.counts["by_type"][vtype] += 1

                    # Direction-based count
                    if origin == "road_11":
                        direction = "left_turn"
                    elif origin == "road_7":
                        direction = "right_turn"
                    else:
                        direction = "straight"

                    self.counts[direction]["total"] += 1
                    if vtype in self.counts[direction]:
                        self.counts[direction][vtype] += 1

                    timestamp = datetime.datetime.now()

                    # Update hourly counts
                    hour_key = timestamp.strftime("%H:00")
                    if hour_key not in self.hourly_counts:
                        self.hourly_counts[hour_key] = {"car": 0, "bus": 0, "truck": 0}
                    if vtype in self.hourly_counts[hour_key]:
                        self.hourly_counts[hour_key][vtype] += 1

                    self.logger.info(
                        "Count: direction=%s type=%s origin=%s id=%s",
                        direction, vtype, origin, track_id,
                    )

                    # Write CSV row
                    self._csv_writer.writerow([
                        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        direction,
                        vtype,
                        self.counts["by_type"]["car"],
                        self.counts["by_type"]["bus"],
                        self.counts["by_type"]["truck"],
                        self.counts["total"],
                    ])
                    self._csv_file.flush()

                    self.save_to_db(timestamp, direction, direction, vtype)
                    self.send_to_api({
                        "timestamp": timestamp.isoformat(),
                        "direction": direction,
                        "vehicle_type": vtype,
                        "origin_zone": origin,
                        "track_id": int(track_id),
                    })

            # --- Draw overlay ---
            overlay = frame.copy()

            # Draw zone polygons
            zone_colors = {"entry": (0, 200, 0), "origin": (200, 120, 0)}
            for zone_key, zone_data in self._scaled_zones.items():
                color = zone_colors.get(zone_data["type"], (128, 128, 128))
                cv2.fillPoly(overlay, [zone_data["polygon"]], color)
                # Zone label at centroid
                cx_z = int(np.mean(zone_data["polygon"][:, 0]))
                cy_z = int(np.mean(zone_data["polygon"][:, 1]))
                cv2.putText(frame, zone_key, (cx_z - 30, cy_z),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # Draw bounding boxes and IDs
            for bbox, track_id, class_id in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id
            ):
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_type = self.VEHICLE_CLASSES.get(int(class_id), "unknown")
                info = self.object_info.get(int(track_id), {})
                # Color: green=counted, yellow=tracking
                color = (0, 255, 0) if info.get("counted") else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{vehicle_type} #{int(track_id)}"
                if info.get("origin_zone"):
                    label += f" [{info['origin_zone']}]"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw count panel (black background at top-left)
            panel_h = 160
            panel_w = 400
            cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
            y_text = 22
            line_h = 22
            cv2.putText(frame, f"3si Entry Total: {self.counts['total']}  "
                        f"(Car:{self.counts['by_type']['car']} Bus:{self.counts['by_type']['bus']} "
                        f"Truck:{self.counts['by_type']['truck']})",
                        (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_text += line_h + 5
            lt = self.counts["left_turn"]
            cv2.putText(frame, f"Left Turn (11si): {lt['total']}  "
                        f"(Car:{lt['car']} Bus:{lt['bus']} Truck:{lt['truck']})",
                        (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_text += line_h
            rt = self.counts["right_turn"]
            cv2.putText(frame, f"Right Turn (7si): {rt['total']}  "
                        f"(Car:{rt['car']} Bus:{rt['bus']} Truck:{rt['truck']})",
                        (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            y_text += line_h
            st = self.counts["straight"]
            cv2.putText(frame, f"Straight/Other:   {st['total']}  "
                        f"(Car:{st['car']} Bus:{st['bus']} Truck:{st['truck']})",
                        (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_text += line_h + 5
            cv2.putText(frame, "Press Q to quit",
                        (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            # Draw hourly bar chart
            self._draw_bar_chart(frame)

            # Display resized frame
            display_h = 720
            scale = display_h / frame.shape[0]
            display_w = int(frame.shape[1] * scale)
            display_frame = cv2.resize(frame, (display_w, display_h))
            cv2.imshow("Vehicle Counter", display_frame)

            if not hasattr(self, "_window_moved"):
                if is_screen:
                    cv2.moveWindow("Vehicle Counter", 0, 0)
                self._window_moved = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        self._cleanup(capture, is_screen)


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vehicle counting with YOLO and ByteTrack")
    parser.add_argument(
        "--version", action="version", version=f"vehicle_counter.py {APP_VERSION}"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source: file path, RTSP URL, or 'screen' for screen capture",
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8s.pt",
        help="Path to YOLO weights file (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="Print OS/Python runtime details and exit without opening capture/model",
    )
    parser.add_argument(
        "--check-deps", action="store_true",
        help="Import runtime/ML dependencies one by one and print diagnostics",
    )
    parser.add_argument(
        "--zones", type=str, default="zones.json",
        help="Path to zone configuration JSON (generated by zone_config.py)",
    )
    parser.add_argument("--screen-top", type=int, default=None)
    parser.add_argument("--screen-left", type=int, default=None)
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument(
        "--select-screen-region", action="store_true",
        help="For --source screen, choose the capture area by mouse drag at startup",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="SQLAlchemy database URL (optional)",
    )
    parser.add_argument(
        "--api-url", type=str, default=None,
        help="Endpoint URL to send count events as JSON (optional)",
    )
    parsed = parser.parse_args(args)
    diagnostic_mode = parsed.check_env or parsed.check_deps
    if not diagnostic_mode and not parsed.source:
        parser.error("--source is required unless --check-env or --check-deps is used")

    screen_values = [
        parsed.screen_top, parsed.screen_left, parsed.screen_width, parsed.screen_height
    ]
    if parsed.source == "screen" and any(v is not None for v in screen_values) and not all(
        v is not None for v in screen_values
    ):
        parser.error(
            "--screen-top, --screen-left, --screen-width and --screen-height "
            "must be provided together. Omit all four or use --select-screen-region "
            "to choose the area with the mouse."
        )
    return parsed


def main() -> None:
    """Entry point for command line execution."""
    args = parse_args()
    if args.check_env:
        print_runtime_environment()
        return
    if args.check_deps:
        ok = print_ml_dependency_check()
        sys.exit(0 if ok else 1)

    screen_region = None
    if (
        args.source == "screen"
        and not args.select_screen_region
        and all(
            v is not None
            for v in [
                args.screen_top, args.screen_left, args.screen_width, args.screen_height
            ]
        )
    ):
        screen_region = {
            "top": args.screen_top,
            "left": args.screen_left,
            "width": args.screen_width,
            "height": args.screen_height,
        }

    try:
        counter = VehicleCounter(
            source=args.source,
            weights=args.weights,
            zones_path=args.zones,
            db_url=args.db_url,
            api_url=args.api_url,
            screen_region=screen_region,
        )
    except DependencyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    counter.run()


if __name__ == "__main__":
    main()