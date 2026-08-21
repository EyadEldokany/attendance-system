"""
Manages the connection to the camera stream (RTSP) with automatic reconnection when disconnected.
The system is supposed to run 24/7, so it must be resistant to network interruptions or camera restarts.
"""
import cv2
import time
import logging

logger = logging.getLogger("attendance.camera")


class CameraStream:
    def __init__(self, source, reconnect_attempts=10, reconnect_delay_seconds=5,
                 frame_width=1280, frame_height=720):
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._cap = None
        self._connect()

    def _connect(self):
        # source can be a number (local USB camera for testing) or an RTSP link
        src = int(self.source) if str(self.source).isdigit() else self.source
        self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # reduce the buffer to reduce latency in the real-time stream
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            logger.error(f"Failed to connect to camera: {self.source}")
            return False
        logger.info(f"Successfully connected to camera: {self.source}")
        return True

    def _reconnect(self):
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning(
                f"Attempting to reconnect to camera ({attempt}/{self.reconnect_attempts})..."
            )
            if self._cap is not None:
                self._cap.release()
            time.sleep(self.reconnect_delay_seconds)
            if self._connect():
                return True
        logger.critical(
            "Failed to reconnect to camera. "
            "Ensure local network stability and correct RTSP link."
        )
        return False

    def read(self):
        """
        Returns (success, frame). If the connection is lost, it tries to reconnect automatically
        before returning a final failure to the caller.
        """
        if self._cap is None or not self._cap.isOpened():
            if not self._reconnect():
                return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Camera stream disconnected, attempting to reconnect...")
            if not self._reconnect():
                return False, None
            ret, frame = self._cap.read()
            if not ret:
                return False, None

        if self.frame_width and self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
