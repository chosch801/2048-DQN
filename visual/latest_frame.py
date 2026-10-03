"""Bounded latest-frame acquisition; never queue stale screenshots."""
from dataclasses import dataclass
import threading
import time

from visual.roi_capture import RegionCapture


@dataclass(frozen=True)
class Frame:
    sequence: int
    started_at: float
    capture_seconds: float
    image: object


class LatestFrames:
    def __init__(self, region, stop_event, capture_factory=RegionCapture):
        self.region, self.stop_event = region, stop_event
        self.capture_factory = capture_factory
        self.closed = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            with self.capture_factory(self.region) as capture:
                sequence = 0
                while not self.closed.is_set() and not self.stop_event.is_set():
                    started = time.monotonic()
                    image = capture.grab()
                    sequence += 1
                    frame = Frame(sequence, started, time.monotonic()-started, image)
                    with self.condition:
                        self.latest = frame
                        self.condition.notify_all()
                    self.closed.wait(max(0, .04-frame.capture_seconds))
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def get(self, after_sequence, not_before, timeout=2):
        deadline = time.monotonic()+timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError(f"棋盘截图失败：{self.error}") from self.error
                if self.stop_event.is_set() or self.closed.is_set():
                    raise RuntimeError("已暂停。")
                frame = self.latest
                if frame is not None and frame.sequence > after_sequence and frame.started_at >= not_before:
                    return frame
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("未收到新的棋盘画面，已暂停。")
                self.condition.wait(min(.1, remaining))

    def close(self):
        self.closed.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2)
