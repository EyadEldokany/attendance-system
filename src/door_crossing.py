"""
Logic for detecting door crossings (entry/exit) based on the tracking of each person's center
across consecutive frames. The idea is to calculate the person's center position relative
to the door line in each frame, and when it changes from one "side" to another
in a stable manner (not due to simple jitter or transient noise), it is considered an actual
crossing event and its direction is determined.
"""
from collections import defaultdict, deque


def _side_of_line(point, p1, p2):
    """
    Returns a positive or negative value depending on which side the point is on relative to the line p1-p2.
    (Cross product principle)
    """
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


class DoorCrossingTracker:
    """
    Important design note: Instead of ambiguous names like "left_to_right" which can be prone to error
    with sloped lines or different camera angles, the system uses an explicit and unambiguous naming convention:
    "positive_to_negative" or "negative_to_positive" - which is automatically determined by the
    setup_door_line.py tool during installation. When the installer actually passes through
    the door once in the real "entry" direction, the system records the correct direction guaranteed 100% regardless of
    the line angle or camera direction.
    """
    def __init__(self, point1, point2, entry_direction="positive_to_negative",
                 min_crossing_confidence_frames=3, frame_width=1280, frame_height=720):
        # point1/point2 come as ratios (0-1) from config.yaml, convert them to actual pixel coordinates
        self.p1 = (point1[0] * frame_width, point1[1] * frame_height)
        self.p2 = (point2[0] * frame_width, point2[1] * frame_height)
        self.entry_direction = entry_direction  
        self.min_confidence_frames = min_crossing_confidence_frames

        # For each track_id, we keep a history of the side (positive/negative) for the last several frames
        self._side_history = defaultdict(lambda: deque(maxlen=10))
        self._already_logged = set()   

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, track_id, bbox):
        """
        Receives the current position of the person and returns None if there is no crossing,
        or "in" / "out" if an actual and confirmed crossing has occurred.
        """
        centroid = self._centroid(bbox)
        side = _side_of_line(centroid, self.p1, self.p2)
        side_sign = 1 if side > 0 else -1
        history = self._side_history[track_id]
        history.append(side_sign)

        if len(history) < self.min_confidence_frames * 2:
            return None 
        # ensure the side has stabilized "before" and "after" the crossing instead of relying on a single jittery frame
        before = list(history)[: self.min_confidence_frames]
        after = list(history)[-self.min_confidence_frames:]

        stable_before = all(v == before[0] for v in before)
        stable_after = all(v == after[0] for v in after)

        if stable_before and stable_after and before[0] != after[0]:
            if track_id in self._already_logged:
                return None 

            raw_transition = "positive_to_negative" if before[0] > after[0] else "negative_to_positive"
            direction = "in" if raw_transition == self.entry_direction else "out"

            self._already_logged.add(track_id)
            history.clear()
            return direction

        return None

    def forget_track(self, track_id):
        """Called when a track disappears from the frame for a long time, to clear memory."""
        self._side_history.pop(track_id, None)
        self._already_logged.discard(track_id)
