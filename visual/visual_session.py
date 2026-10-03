"""One observed board -> one action -> a validated spawn transition."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from visual import desktop_io
from afterstate import slide_board, valid_actions, spawn_outcomes
from evaluate import load_model
from visual.latest_frame import LatestFrames
from search import choose_expectimax_actions_batch
from visual.visual_board import BoardReader, RecognitionError


def valid_successor(previous, action, current):
    afterstate, _, changed = slide_board(previous, action)
    return changed and any(np.array_equal(current, state) for _, state in spawn_outcomes(afterstate))


class Session:
    def __init__(self, region, hwnd, checkpoint, key_mode, on_status, on_error, on_finished, stop_event):
        self.region, self.hwnd, self.checkpoint, self.key_mode = region, hwnd, checkpoint, key_mode
        self.on_status, self.on_error, self.on_finished = on_status, on_error, on_finished
        self.stop = stop_event

    def run(self):
        image = None
        frames = None
        try:
            reader = BoardReader()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_num_threads(1)
            model, version = load_model(self.checkpoint, device)
            if version != "v3.1-afterstate-quantile-cnn-dqn":
                raise RuntimeError("此视觉助手使用V3.1 transition奖励模型，请选择对应检查点。")
            for seconds in (3, 2, 1):
                self.on_status(f"{seconds}秒后开始，Esc可暂停。")
                if self.stop.wait(1):
                    self.on_error("已取消开始。", image)
                    return
            if not desktop_io.focus_window(self.hwnd):
                raise RuntimeError("无法激活选定的游戏窗口，请先点击游戏后重试。")
            previous, action, candidate = None, None, None
            frames = LatestFrames(self.region, self.stop)
            frames.start()
            sequence, not_before = 0, 0.0
            intervals = []
            stable_since = time.monotonic()
            waiting_since = stable_since
            banner, banner_since = None, None
            last_banner_check = 0.0
            unreadable_since = None
            cycle_ocr_seconds = cycle_banner_seconds = 0.0
            cycle_frames = cycle_errors = 0
            last_action_time = None
            moves = 0
            while not self.stop.is_set():
                if desktop_io.escape_pressed():
                    raise RuntimeError("已按Esc暂停。")
                if not desktop_io.is_foreground(self.hwnd):
                    raise RuntimeError("游戏窗口失去焦点，已暂停按键。")
                frame_start = time.perf_counter()
                frame = frames.get(sequence, not_before)
                frame_wait_seconds = time.perf_counter() - frame_start
                sequence, image = frame.sequence, frame.image
                capture_seconds = frame.capture_seconds
                read_start = time.perf_counter()
                cycle_frames += 1
                try:
                    board = reader.read(image)
                except RecognitionError as error:
                    cycle_ocr_seconds += time.perf_counter() - read_start
                    cycle_errors += 1
                    candidate = None
                    failed_at = time.monotonic()
                    if unreadable_since is None:
                        unreadable_since = failed_at
                    # A moving/spawning tile is not evidence of a game-over banner.
                    if previous is not None and failed_at - unreadable_since >= .75 and failed_at - last_banner_check >= 1:
                        last_banner_check = failed_at
                        banner_start = time.perf_counter()
                        text = reader.terminal_banner(image)
                        cycle_banner_seconds += time.perf_counter() - banner_start
                        if text:
                            if text != banner:
                                banner, banner_since = text, time.monotonic()
                            elif time.monotonic() - banner_since >= .7:
                                self.on_status("连续确认游戏结束提示，助手即将退出。", image)
                                self.on_finished()
                                return
                        else:
                            banner, banner_since = None, None
                    if time.monotonic() - waiting_since > 8:
                        raise error
                    self.stop.wait(.04)
                    continue
                unreadable_since = None
                banner, banner_since = None, None
                read_seconds = time.perf_counter() - read_start
                cycle_ocr_seconds += read_seconds
                now = time.monotonic()
                if candidate is None or not np.array_equal(candidate, board):
                    candidate, stable_since = board.copy(), now
                elif now - stable_since >= .16:
                    masks = valid_actions(board)
                    if not masks.any() and now - stable_since < .35:
                        self.stop.wait(.06)
                        continue
                    if previous is not None and not valid_successor(previous, action, board):
                        if now - waiting_since > 8:
                            raise RuntimeError("新棋盘与上次动作不符或按键未生效；为避免重复输入，已暂停。")
                    else:
                        self.on_status(f"已执行{moves}步；棋盘已确认。", image, board)
                        if not masks.any():
                            self.on_status("稳定棋盘无合法动作，助手即将退出。", image, board)
                            self.on_finished()
                            return
                        search_start = time.perf_counter()
                        action = int(choose_expectimax_actions_batch(model, board[None], masks[None], device, .99)[0])
                        search_seconds = time.perf_counter() - search_start
                        if self.stop.is_set() or desktop_io.escape_pressed():
                            raise RuntimeError("已暂停。")
                        desktop_io.tap_action(self.hwnd, action, self.key_mode)
                        previous, candidate = board.copy(), None
                        waiting_since = time.monotonic()
                        # Skip the first sliding/spawning animation frames before OCR.
                        not_before = waiting_since + .12
                        moves += 1
                        action_time = time.perf_counter()
                        if last_action_time is not None:
                            intervals.append(action_time - last_action_time)
                            intervals = intervals[-30:]
                        timing = {"moves": moves, "device": str(device), "capture_seconds": capture_seconds,
                                  "capture_backend": "GDI ROI / latest-frame worker",
                                  "frame_wait_seconds": frame_wait_seconds,
                                  "recent_mean_interval_seconds": sum(intervals)/len(intervals) if intervals else None,
                                  "ocr_seconds": read_seconds, "search_seconds": search_seconds,
                                  "cycle_ocr_seconds": cycle_ocr_seconds,
                                  "cycle_banner_seconds": cycle_banner_seconds,
                                  "cycle_frames": cycle_frames, "cycle_recognition_errors": cycle_errors,
                                  "action_interval_seconds": None if last_action_time is None else action_time-last_action_time,
                                  "scope": "capture/OCR are the final stable frame; interval includes all retries and waiting"}
                        last_action_time = action_time
                        directory = Path(__file__).parent / "visual_diagnostics"
                        directory.mkdir(exist_ok=True)
                        (directory / "latest_timing.json").write_text(json.dumps(timing), encoding="utf-8")
                        cycle_ocr_seconds = cycle_banner_seconds = 0.0
                        cycle_frames = cycle_errors = 0
                if now - waiting_since > 8:
                    raise RecognitionError("棋盘持续变化，无法确认稳定状态；请检查动画、选区或遮挡。")
                self.stop.wait(.06)
            self.on_error("已暂停，可检查预览后重新开始。", image)
        except Exception as error:
            if image is not None:
                directory = Path(__file__).parent / "visual_diagnostics"
                directory.mkdir(exist_ok=True)
                stem = str(time.time_ns())
                image.save(directory / f"{stem}.png")
                (directory / f"{stem}.json").write_text(json.dumps({"error": str(error), "region": self.region}, ensure_ascii=False), encoding="utf-8")
            self.on_error(str(error), image)
        finally:
            if frames is not None:
                frames.close()
