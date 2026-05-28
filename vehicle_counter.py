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

import argparse
import datetime
import logging
import sys
from typing import Dict, Optional


try:
    # Import optional dependencies.  These imports will fail in
    # environments where the packages are not installed.  Catch the
    # exceptions early so the user gets a clear error message.
    from ultralytics import YOLO  # type: ignore
    import supervision as sv  # type: ignore
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import requests  # type: ignore
    from sqlalchemy import create_engine  # type: ignore
    from sqlalchemy.exc import SQLAlchemyError  # type: ignore
except ImportError as e:  # pragma: no cover
    missing_pkg = str(e).split("'")[1]
    print(
        f"Missing required package: {missing_pkg}. Please install the "
        "dependencies listed in the module documentation before running this script.",
        file=sys.stderr,
    )
    # Re‑raise to inform the user.  In production, you might choose to
    # install packages programmatically or handle missing imports more
    # gracefully.
    raise


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
        y_line_ratio: float = 0.5,
        lane_divider_ratio: float = 0.5,
        db_url: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        """
        Initialise the vehicle counter.

        Parameters
        ----------
        source : str
            Path or RTSP URL of the video source.
        weights : str
            Path to the YOLO model weights.  You can download pretrained
            weights from Ultralytics (e.g. yolov8n.pt, yolov8s.pt,
            yolov11n.pt, etc.).
        y_line_ratio : float, optional
            Vertical position of the counting line expressed as a fraction
            of frame height (0 = top, 1 = bottom).  Defaults to 0.5.
        lane_divider_ratio : float, optional
            Horizontal position of the lane divider expressed as a
            fraction of frame width (0 = left, 1 = right).  Defaults
            to 0.5.
        db_url : str, optional
            SQLAlchemy database URL for persisting count records.  If
            provided, the script will attempt to insert a row for each
            counted vehicle.
        api_url : str, optional
            Endpoint URL for sending count events as JSON via HTTP
            POST.  If provided, the script will send each count event
            to this endpoint.
        """
        self.source = source
        self.weights = weights
        self.y_line_ratio = y_line_ratio
        self.lane_divider_ratio = lane_divider_ratio
        self.db_url = db_url
        self.api_url = api_url

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load YOLO model
        self.logger.info("Loading YOLO model from %s", weights)
        self.model = YOLO(weights)

        # Initialise tracker
        self.tracker = sv.ByteTrack()

        # Database engine (initialised lazily)
        self.db_engine = None
        if self.db_url:
            try:
                self.db_engine = create_engine(self.db_url)
                # Test connection by connecting and immediately closing
                with self.db_engine.connect() as connection:
                    pass
                self.logger.info("Connected to database: %s", self.db_url)
            except SQLAlchemyError as e:
                self.logger.error("Failed to connect to DB: %s", e)
                self.db_engine = None

        # Counting state
        self.counts = {
            "lane1": {"up": 0, "down": 0},
            "lane2": {"up": 0, "down": 0},
        }
        # Maintain per object history to avoid double counting
        self.object_info: Dict[int, Dict[str, object]] = {}

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

    def run(self) -> None:
        """Start the counting loop.

        Opens the video source, processes each frame to detect and track
        vehicles, updates counts when vehicles cross the counting line,
        and optionally displays annotated frames.
        """
        # Open video capture
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error("Failed to open video source: %s", self.source)
            return
        self.logger.info("Processing video stream: %s", self.source)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width == 0 or frame_height == 0:
            # Some streams return 0 for width/height before reading a frame.
            # We'll update these values after the first successful read.
            frame_width, frame_height = None, None

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_width is None or frame_height is None:
                frame_height, frame_width = frame.shape[:2]

            # Compute positions of counting line and lane divider
            y_line = int(frame_height * self.y_line_ratio)
            lane_divider_x = int(frame_width * self.lane_divider_ratio)

            # Run YOLO detection.  The Ultralytics API returns a list
            # of results, one per image; we take the first result.
            results = self.model.predict(frame, device=0)
            result = results[0]

            # Convert to Supervision Detections
            detections = sv.Detections.from_ultralytics(result)

            # Filter detections by vehicle classes of interest
            mask = np.isin(detections.class_id, list(self.VEHICLE_CLASSES.keys()))
            detections.xyxy = detections.xyxy[mask]
            detections.class_id = detections.class_id[mask]
            detections.confidence = detections.confidence[mask]

            # Update tracker with filtered detections
            tracked = self.tracker.update_with_detections(detections)

            # Process each tracked detection
            for bbox, track_id, class_id in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id
            ):
                # Compute centre of bounding box
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                lane = "lane1" if cx < lane_divider_x else "lane2"
                vehicle_type = self.VEHICLE_CLASSES.get(int(class_id), "unknown")

                # Initialise object history if new
                info = self.object_info.setdefault(
                    int(track_id),
                    {
                        "positions": [],
                        "counted_up": False,
                        "counted_down": False,
                        "lane": lane,
                        "vehicle_type": vehicle_type,
                    },
                )
                # Update lane and vehicle_type in case class changes
                info["lane"] = lane
                info["vehicle_type"] = vehicle_type

                # Append current y position
                info["positions"].append(cy)
                # Keep only last few positions to limit memory usage
                if len(info["positions"]) > 5:
                    info["positions"] = info["positions"][-5:]

                # Determine if the object has crossed the counting line
                # Check if we have at least two positions to compare
                if len(info["positions"]) >= 2:
                    prev_y = info["positions"][-2]
                    curr_y = info["positions"][-1]
                    # Crossing from above to below: count as down (exit)
                    if (
                        prev_y < y_line <= curr_y
                        and not info["counted_down"]
                    ):
                        self.counts[lane]["down"] += 1
                        info["counted_down"] = True
                        timestamp = datetime.datetime.now()
                        self.logger.info(
                            "Exit: lane=%s type=%s id=%s", lane, vehicle_type, track_id
                        )
                        self.save_to_db(timestamp, lane, "down", vehicle_type)
                        self.send_to_api(
                            {
                                "timestamp": timestamp.isoformat(),
                                "lane": lane,
                                "direction": "down",
                                "vehicle_type": vehicle_type,
                                "track_id": int(track_id),
                            }
                        )
                    # Crossing from below to above: count as up (enter)
                    elif (
                        prev_y > y_line >= curr_y
                        and not info["counted_up"]
                    ):
                        self.counts[lane]["up"] += 1
                        info["counted_up"] = True
                        timestamp = datetime.datetime.now()
                        self.logger.info(
                            "Enter: lane=%s type=%s id=%s", lane, vehicle_type, track_id
                        )
                        self.save_to_db(timestamp, lane, "up", vehicle_type)
                        self.send_to_api(
                            {
                                "timestamp": timestamp.isoformat(),
                                "lane": lane,
                                "direction": "up",
                                "vehicle_type": vehicle_type,
                                "track_id": int(track_id),
                            }
                        )

            # Optional: annotate frame for display
            # Draw counting line
            cv2.line(
                frame,
                (0, y_line),
                (frame_width, y_line),
                (0, 255, 255),
                2,
            )
            # Draw lane divider
            cv2.line(
                frame,
                (lane_divider_x, 0),
                (lane_divider_x, frame_height),
                (255, 0, 255),
                2,
            )
            # Draw bounding boxes and IDs
            for bbox, track_id, class_id in zip(
                tracked.xyxy, tracked.tracker_id, tracked.class_id
            ):
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_type = self.VEHICLE_CLASSES.get(int(class_id), "unknown")
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{vehicle_type} #{int(track_id)}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            # Draw counts on frame
            cv2.putText(
                frame,
                f"Lane1 Up: {self.counts['lane1']['up']} Down: {self.counts['lane1']['down']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Lane2 Up: {self.counts['lane2']['up']} Down: {self.counts['lane2']['down']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display the frame in a window.  Press 'q' to quit.
            cv2.imshow("Vehicle Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vehicle counting with YOLO and ByteTrack")
    parser.add_argument("--source", type=str, required=True, help="Video source (file path or RTSP URL)")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Path to YOLO weights file (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--y-line-ratio",
        type=float,
        default=0.5,
        help="Vertical position of counting line as fraction of frame height (0-1)",
    )
    parser.add_argument(
        "--lane-divider-ratio",
        type=float,
        default=0.5,
        help="Horizontal position of lane divider as fraction of frame width (0-1)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="SQLAlchemy database URL (optional)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Endpoint URL to send count events as JSON (optional)",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point for command line execution."""
    args = parse_args()
    counter = VehicleCounter(
        source=args.source,
        weights=args.weights,
        y_line_ratio=args.y_line_ratio,
        lane_divider_ratio=args.lane_divider_ratio,
        db_url=args.db_url,
        api_url=args.api_url,
    )
    counter.run()


if __name__ == "__main__":
    main()