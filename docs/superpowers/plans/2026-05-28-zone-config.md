# Zone Configuration Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `zone_config.py` — an interactive GUI tool that lets the user draw 3 polygon zones on a road image or screen capture, then saves them to `zones.json`.

**Architecture:** Single-file OpenCV app. Mouse callback adds polygon vertices, keyboard controls flow (Enter=confirm, R=reset, S=save, Q=cancel). Zones are drawn sequentially in fixed order. Output is a JSON file consumed by `vehicle_counter.py`.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, mss (screen capture), json

---

### File Structure

| File | Responsibility |
|------|---------------|
| `zone_config.py` (create) | Interactive zone drawing GUI + CLI entry point |
| `zones.json` (output) | Generated zone configuration, consumed by vehicle_counter.py |

---

### Task 1: CLI argument parsing and image loading

**Files:**
- Create: `zone_config.py`

- [ ] **Step 1: Create `zone_config.py` with argument parsing**

```python
import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

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
            if all(v is not None for v in [args.screen_top, args.screen_left, args.screen_width, args.screen_height]):
                monitor = {"top": args.screen_top, "left": args.screen_left,
                           "width": args.screen_width, "height": args.screen_height}
            else:
                monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)[:, :, :3].copy()
            return frame
    else:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Failed to load image: {args.source}", file=sys.stderr)
            return None
        return frame
```

- [ ] **Step 2: Add `main()` that loads frame and displays it**

Append to `zone_config.py`:

```python
def main():
    args = parse_args()
    frame = load_frame(args)
    if frame is None:
        sys.exit(1)

    print(f"Loaded frame: {frame.shape[1]}x{frame.shape[0]}")
    cv2.imshow("Zone Config", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test manually**

Run: `python zone_config.py --source road_image.jpg`
Expected: Window shows the road image. Press any key to close.

- [ ] **Step 4: Commit**

```bash
git add zone_config.py
git commit -m "feat: zone_config.py scaffold with arg parsing and image loading"
```

---

### Task 2: Interactive polygon drawing with mouse callback

**Files:**
- Modify: `zone_config.py`

- [ ] **Step 1: Add the ZoneDrawer class**

Insert after the `COLORS` dict and before `parse_args`:

```python
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
            # Draw lines between points
            for i in range(len(pts) - 1):
                cv2.line(display, tuple(pts[i]), tuple(pts[i + 1]), COLORS["drawing"], 2)
            # Close polygon preview (dashed effect: just draw lighter)
            if len(pts) > 2:
                cv2.line(display, tuple(pts[-1]), tuple(pts[0]), COLORS["drawing"], 1)
            # Draw circles at vertices
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
```

- [ ] **Step 2: Update `main()` to use ZoneDrawer**

Replace the existing `main()`:

```python
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
```

- [ ] **Step 3: Test manually with image**

Run: `python zone_config.py --source road_image.jpg`
Expected:
1. Window shows road image with instruction bar at top
2. Click 3+ points to draw a polygon for "3si Road"
3. Press Enter to confirm — polygon fills green, advances to next zone
4. Repeat for "11si Road" (blue) and "7si Road" (blue)
5. After all 3, press S to save → `zones.json` is created
6. Press Q at any time to cancel

- [ ] **Step 4: Test with screen capture**

Run: `python zone_config.py --source screen --screen-top 200 --screen-left 100 --screen-width 800 --screen-height 600`
Expected: Same behavior but using a screen capture as the background image.

- [ ] **Step 5: Commit**

```bash
git add zone_config.py
git commit -m "feat: interactive polygon zone drawing with mouse + keyboard controls"
```

---

### Task 3: Final polish and push

**Files:**
- Modify: `zone_config.py`
- Modify: `README.md`

- [ ] **Step 1: Verify `zones.json` output format**

Run: `python zone_config.py --source road_image.jpg`
Draw all 3 zones and save. Then verify:

Run: `python -c "import json; d=json.load(open('zones.json')); print(json.dumps(d, indent=2))"`

Expected output structure:
```json
{
  "image_size": [width, height],
  "zones": {
    "road_3": {"type": "entry", "points": [[x, y], ...]},
    "road_11": {"type": "origin", "points": [[x, y], ...]},
    "road_7": {"type": "origin", "points": [[x, y], ...]}
  }
}
```

- [ ] **Step 2: Update README.md zone config section**

Replace the "1. Zone Setup" section in README.md with actual usage instructions:

```markdown
### 1. Zone Setup

Run the zone configuration tool to define detection zones:

```bash
# Using a saved image
python zone_config.py --source road_image.jpg

# Using screen capture
python zone_config.py --source screen --screen-top 200 --screen-left 100 --screen-width 800 --screen-height 600
```

The tool guides you through drawing 3 zones in order:
1. **3si Road (Entry)** — where vehicles are counted entering
2. **11si Road (Origin)** — left turn origin
3. **7si Road (Origin)** — right turn origin

Controls:
- **Left click**: add polygon vertex
- **R**: reset current zone
- **Enter**: confirm current zone
- **S**: save all zones (after all 3 are drawn)
- **Q**: cancel

Zones are saved to `zones.json`.
```

- [ ] **Step 3: Commit and push**

```bash
git add zone_config.py README.md
git commit -m "feat: zone_config.py complete with README update"
git push
```
