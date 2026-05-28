"""
zone_config.py
==============

Interactive GUI tool for defining polygon zones on a road image or
screen capture. Zones are used by vehicle_counter.py to determine
vehicle entry direction (straight, left turn, right turn).

Usage:
    python zone_config.py --source road_image.jpg
    python zone_config.py --source screen --screen-top 200 --screen-left 100 --screen-width 800 --screen-height 600
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None


ZONE_DEFS = [
    {"key": "road_3", "type": "entry", "label": "3si Road (Entry Detection)"},
    {"key": "road_11", "type": "origin", "label": "11si Road (Left Turn Origin)"},
    {"key": "road_7", "type": "origin", "label": "7si Road (Right Turn Origin)"},
]

COLORS = {
    "entry": (0, 200, 0),       # green
    "origin": (200, 120, 0),    # blue
    "drawing": (0, 255, 255),   # yellow
}


class ZoneDrawer:
    """Interactive polygon zone drawing on an OpenCV window."""

    WINDOW_NAME = "Zone Config"

    def __init__(self, frame: np.ndarray, output_path: str):
        self.original_frame = frame.copy()
        self.output_path = output_path
        self.image_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # State
        self.current_zone_idx = 0
        self.current_points: List[List[int]] = []
        self.confirmed_zones: Dict[str, Dict] = {}
        self.done = False

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to add polygon vertices."""
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])

    def _draw_overlay(self):
        """Draw all confirmed zones and the current zone being drawn."""
        display = self.original_frame.copy()
        overlay = display.copy()

        # Draw confirmed zones as filled semi-transparent polygons
        for key, zone_data in self.confirmed_zones.items():
            pts = np.array(zone_data["points"], dtype=np.int32)
            color = COLORS[zone_data["type"]]
            cv2.fillPoly(overlay, [pts], color)
            # Label at centroid
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(display, key, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # Draw current zone points and lines
        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(display, tuple(pts[i]), tuple(pts[i + 1]), COLORS["drawing"], 2)
            if len(pts) > 2:
                cv2.line(display, tuple(pts[-1]), tuple(pts[0]), COLORS["drawing"], 1)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, COLORS["drawing"], -1)

        # Draw instruction bar at top
        if not self.done:
            zone_def = ZONE_DEFS[self.current_zone_idx]
            text = f"[{self.current_zone_idx + 1}/3] Draw: {zone_def['label']} | Click: vertex, R: reset, Enter: confirm, Q: cancel"
        else:
            text = "All zones drawn | S: save, Q: cancel"

        cv2.rectangle(display, (0, 0), (display.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(display, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return display

    def _confirm_current_zone(self):
        """Confirm the current zone and advance to the next."""
        if len(self.current_points) < 3:
            return  # Need at least 3 points for a polygon

        zone_def = ZONE_DEFS[self.current_zone_idx]
        self.confirmed_zones[zone_def["key"]] = {
            "type": zone_def["type"],
            "points": self.current_points[:],
        }
        self.current_points = []
        self.current_zone_idx += 1

        if self.current_zone_idx >= len(ZONE_DEFS):
            self.done = True

    def _save(self):
        """Save confirmed zones to JSON."""
        data = {
            "image_size": list(self.image_size),
            "zones": self.confirmed_zones,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Zones saved to {self.output_path}")

    def run(self):
        """Main drawing loop."""
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        while True:
            display = self._draw_overlay()
            cv2.imshow(self.WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("Cancelled.")
                break
            elif key == 13:  # Enter
                if not self.done:
                    self._confirm_current_zone()
            elif key == ord("r"):
                if not self.done:
                    self.current_points = []
            elif key == ord("s"):
                if self.done:
                    self._save()
                    break

        cv2.destroyAllWindows()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Zone configuration tool for vehicle counter")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image file path or 'screen' for screen capture",
    )
    parser.add_argument("--screen-top", type=int, default=None)
    parser.add_argument("--screen-left", type=int, default=None)
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument(
        "--output", type=str, default="zones.json",
        help="Output JSON file path (default: zones.json)",
    )
    return parser.parse_args(args)


def load_frame(args) -> Optional[np.ndarray]:
    """Load a single frame from image file or screen capture."""
    if args.source == "screen":
        if mss is None:
            print("mss package is required for screen capture.", file=sys.stderr)
            return None
        with mss.MSS() as sct:
            if all(v is not None for v in [args.screen_top, args.screen_left,
                                           args.screen_width, args.screen_height]):
                monitor = {"top": args.screen_top, "left": args.screen_left,
                           "width": args.screen_width, "height": args.screen_height}
            else:
                # Let user drag-select a region
                full_monitor = sct.monitors[1]
                screenshot = sct.grab(full_monitor)
                full_frame = np.array(screenshot)[:, :, :3].copy()
                max_h = 900
                scale = max_h / full_frame.shape[0]
                sel_w = int(full_frame.shape[1] * scale)
                sel_frame = cv2.resize(full_frame, (sel_w, max_h))
                cv2.putText(sel_frame, "Drag to select region, then press ENTER or SPACE",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                roi = cv2.selectROI("Select Region", sel_frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select Region")
                x, y, w, h = roi
                if w == 0 or h == 0:
                    print("No region selected.", file=sys.stderr)
                    return None
                monitor = {
                    "top": full_monitor["top"] + int(y / scale),
                    "left": full_monitor["left"] + int(x / scale),
                    "width": int(w / scale),
                    "height": int(h / scale),
                }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)[:, :, :3].copy()
            return frame
    else:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Failed to load image: {args.source}", file=sys.stderr)
            return None
        return frame


def main():
    args = parse_args()
    frame = load_frame(args)
    if frame is None:
        sys.exit(1)

    print(f"Loaded frame: {frame.shape[1]}x{frame.shape[0]}")
    drawer = ZoneDrawer(frame, args.output)
    drawer.run()


if __name__ == "__main__":
    main()
