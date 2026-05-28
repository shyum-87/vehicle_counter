# Zone Configuration Tool Design

## Overview

A GUI tool (`zone_config.py`) that lets the user draw polygon zones on a road image or live screen capture. These zones define where vehicles are detected, tracked, and counted in the main `vehicle_counter.py`.

## Zones (3 total)

| Zone key | Type | Purpose |
|---|---|---|
| `road_3` | `entry` | 3 o'clock road — vehicles arriving here are counted |
| `road_11` | `origin` | 11 o'clock road — vehicles starting here and reaching `road_3` = left turn |
| `road_7` | `origin` | 7 o'clock road — vehicles starting here and reaching `road_3` = right turn |

No exclusion zones needed. The `road_3` entry zone is placed past the 6 o'clock parking lot entrance, so parking-bound vehicles never reach it.

## Input

- `--source`: image file path (e.g. `road_image.jpg`) or `screen`
- Screen capture uses the same `--screen-top/left/width/height` options as `vehicle_counter.py`

## User Flow

1. Image or screen capture frame displayed in an OpenCV window
2. Zones are drawn one at a time in fixed order: `road_3` → `road_11` → `road_7`
3. For each zone:
   - Top of window shows: "Draw [zone name] — Click: add vertex, R: reset, Enter: confirm"
   - Left-click adds polygon vertices (connected by lines in real-time)
   - `R` key resets current zone's points
   - `Enter` key confirms current zone and advances to next
4. After all 3 zones are confirmed:
   - All zones displayed with overlay
   - `S` to save, `Q` to cancel

## Output

`zones.json` in the project root:

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

Coordinates are in pixel space of the source image/capture frame.

## Visualization

- Confirmed zones: semi-transparent overlay
  - Entry zone (`road_3`): green
  - Origin zones (`road_11`, `road_7`): blue
- Current zone being drawn: bright yellow lines + circle on each vertex
- Zone label text at centroid of each confirmed polygon

## Vehicle Counting Logic (for context)

The main counter will use these zones as follows:
- Track each vehicle's trajectory across frames
- Record which zone a vehicle first appears in (origin)
- When a vehicle enters `road_3`:
  - If origin was `road_11` → left turn count
  - If origin was `road_7` → right turn count
  - Otherwise → straight/other count
- Count by vehicle type: car (COCO 2), bus (COCO 5), truck (COCO 7)
- No additional model training required — YOLOv8s COCO weights cover all 3 classes

## Dependencies

No new packages needed. Uses `cv2`, `numpy`, `mss`, `json` — all already installed.
