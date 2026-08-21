"""
Two-step initial setup tool for the door line, supporting two modes:

LIVE MODE (default): connects to the live camera configured in config.yaml.
  You walk physically through the door once to calibrate the true entry
  direction.

ARCHIVE MODE (pass a video file path as argument): uses a frame from a
  recorded video instead of a live camera - useful when the system runs in
  archive/batch mode with no live camera connected. Since you can't
  physically "walk through the door" in a pre-recorded video, the tool plays
  through the footage, pauses at the first detected door crossing, and asks
  you to label it as an entry or exit by watching that clip.

Usage:
    python setup_door_line.py                          (live camera mode)
    python setup_door_line.py "D:/archive/sample.mp4"   (archive mode)

Step 1: click and drag two points to mark the door line, press 's' to confirm.
Step 2 (live mode): physically walk from the door to the inside once.
Step 2 (archive mode): watch the video play, and when it pauses on a
  detected crossing, press 'i' if that person was entering, 'o' if exiting.

'r' to redo the line, 'q' to quit without saving.
"""
import cv2
import sys
import time
from src.config import load_config
from src.camera_stream import CameraStream
from src.detector_tracker import DetectorTracker
from src.device_utils import resolve_device
from src.door_crossing import DoorCrossingTracker

points = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))


def _select_line(frame, w, h, config):
    window_name = "Click two points for the door line - s to confirm / r to redo / q to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)
    print("Step 1: click two points on the frame to mark the door line, then press 's'.")

    while True:
        display = frame.copy()
        for pt in points:
            cv2.circle(display, pt, 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord("s"):
            if len(points) != 2:
                print("You need to select two points first.")
                continue
            p1_norm = [points[0][0] / w, points[0][1] / h]
            p2_norm = [points[1][0] / w, points[1][1] / h]
            config.set("door_line.point1", p1_norm, persist=True)
            config.set("door_line.point2", p2_norm, persist=True)
            cv2.destroyAllWindows()
            return p1_norm, p2_norm


def _calibrate_direction_live(camera, config, p1_norm, p2_norm, w, h):
    print("\nStep 2: now physically walk from the door to the inside (the real "
          "entry direction) and cross the door line in front of the camera...")
    print("(press 'q' at any time to cancel)")

    detector = DetectorTracker(
        model_path=config.get("detection.model_path"),
        confidence_threshold=config.get("detection.confidence_threshold", 0.5),
        device=resolve_device(config.get("detection.device", "auto")),
    )
    door = DoorCrossingTracker(
        point1=p1_norm, point2=p2_norm,
        entry_direction="positive_to_negative",  # placeholder, resolved below
        min_crossing_confidence_frames=3,
        frame_width=w, frame_height=h,
    )

    window_name = "Walk in from the door now - press q to cancel"
    cv2.namedWindow(window_name)
    timeout_seconds = 30
    start = time.time()

    while time.time() - start < timeout_seconds:
        ok, frame = camera.read()
        if not ok:
            continue
        detections = detector.track(frame)
        for det in detections:
            result_direction = door.update(det["track_id"], det["bbox"])
            if result_direction is not None:
                raw = "positive_to_negative" if result_direction == "in" else "negative_to_positive"
                config.set("door_line.entry_direction", raw, persist=True)
                cv2.destroyAllWindows()
                print(f"\nCrossing detected - saved the correct entry direction: {raw}")
                return True

        cv2.imshow(window_name, frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\nNo crossing was detected within the time limit. Try running the "
          "tool again and walk clearly in front of the camera.")
    return False


def _calibrate_direction_archive(video_path, config, p1_norm, p2_norm, w, h):
    print("\nStep 2: playing through the video. It will pause at the first "
          "detected door crossing - watch the clip and press 'i' if that person "
          "was entering, or 'o' if they were exiting. Press 'q' to cancel.")

    detector = DetectorTracker(
        model_path=config.get("detection.model_path"),
        confidence_threshold=config.get("detection.confidence_threshold", 0.5),
        device=resolve_device(config.get("detection.device", "auto")),
    )
    door = DoorCrossingTracker(
        point1=p1_norm, point2=p2_norm,
        entry_direction="positive_to_negative",  # placeholder, resolved below
        min_crossing_confidence_frames=3,
        frame_width=w, frame_height=h,
    )

    cap = cv2.VideoCapture(video_path)
    window_name = "Archive calibration - press q to cancel"
    cv2.namedWindow(window_name)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h))
        detections = detector.track(frame)

        crossing_result = None
        crossing_track = None
        for det in detections:
            r = door.update(det["track_id"], det["bbox"])
            if r is not None:
                crossing_result = r
                crossing_track = det
                break

        display = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.line(display, (int(p1_norm[0]*w), int(p1_norm[1]*h)),
                  (int(p2_norm[0]*w), int(p2_norm[1]*h)), (0, 255, 255), 2)
        cv2.imshow(window_name, display)

        if crossing_result is not None:
            print(f"\nCrossing detected for track {crossing_track['track_id']}.")
            print("Was this person ENTERING or EXITING? Press 'i' for entry, 'o' for exit.")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("i") or key == ord("o"):
                    human_says_entry = (key == ord("i"))
                    # the raw transition that JUST happened produced crossing_result
                    # ("in" or "out" under a placeholder direction) - map it correctly
                    # based on what the human actually observed.
                    raw_transition_was_positive_to_negative = (crossing_result == "in")
                    if human_says_entry:
                        correct_raw = ("positive_to_negative" if raw_transition_was_positive_to_negative
                                       else "negative_to_positive")
                    else:
                        correct_raw = ("negative_to_positive" if raw_transition_was_positive_to_negative
                                       else "positive_to_negative")
                    config.set("door_line.entry_direction", correct_raw, persist=True)
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"\nSaved the correct entry direction: {correct_raw}")
                    return True
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nNo crossing was detected in this video. Try a different sample "
          "video that clearly shows someone using the door.")
    return False


def main():
    config = load_config("config.yaml")
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if video_arg:
        print(f"Archive mode: using sample video '{video_arg}'")
        cap = cv2.VideoCapture(video_arg)
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("Could not read a frame from the video file.")
            sys.exit(1)
    else:
        print("Live camera mode (no video file given).")
        camera = CameraStream(
            source=config.get("camera.source"),
            frame_width=config.get("camera.frame_width", 1280),
            frame_height=config.get("camera.frame_height", 720),
        )
        ok, frame = camera.read()
        if not ok:
            print("Failed to read a frame from the camera. Check camera.source in config.yaml")
            sys.exit(1)

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (config.get("camera.frame_width", 1280), config.get("camera.frame_height", 720)))
    h, w = frame.shape[:2]

    p1_norm, p2_norm = _select_line(frame, w, h, config)
    print(f"Door line saved: {p1_norm} -> {p2_norm}")

    if video_arg:
        _calibrate_direction_archive(video_arg, config, p1_norm, p2_norm, w, h)
    else:
        _calibrate_direction_live(camera, config, p1_norm, p2_norm, w, h)
        camera.release()


if __name__ == "__main__":
    main()
