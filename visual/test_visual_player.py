"""Offline visual and control-flow checks; never send desktop input."""
import itertools
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from afterstate import slide_board, spawn_outcomes
from visual.visual_board import RecognitionError
from visual.visual_session import Session, valid_successor
from visual.latest_frame import Frame


class VisualPlayerTests(unittest.TestCase):


    def test_successor_requires_exactly_one_legal_spawn(self):
        before = np.array([[2,2,0,0],[4,8,0,0],[16,32,0,0],[64,128,0,0]])
        after, _, _ = slide_board(before, 2)
        for _, current in spawn_outcomes(after):
            self.assertTrue(valid_successor(before, 2, current))
        self.assertFalse(valid_successor(before, 2, before))
        self.assertFalse(valid_successor(before, 2, after))
        bad = after.copy(); bad[0,1] = 8
        self.assertFalse(valid_successor(before, 2, bad))

    def run_mock_session(self, reader, expected_keys=0, clock_step=1, configure=None):
        clock = itertools.count(start=100, step=clock_step)
        finish, error = MagicMock(), MagicMock()
        stop = threading.Event()
        image = Image.new("RGB", (320,320), "white")
        with tempfile.TemporaryDirectory() as directory, patch("visual.visual_session.BoardReader", return_value=reader), \
             patch("visual.visual_session.load_model", return_value=(MagicMock(), "v3.1-afterstate-quantile-cnn-dqn")), \
             patch("visual.visual_session.desktop_io") as io, \
             patch("visual.visual_session.LatestFrames") as frames, \
             patch("visual.visual_session.choose_expectimax_actions_batch", return_value=np.array([2])), \
             patch("visual.visual_session.time.monotonic", side_effect=lambda: next(clock)), \
             patch("visual.visual_session.Path", side_effect=lambda *args: Path(directory)/"session.py"), \
             patch.object(stop, "wait", return_value=False):
            io.focus_window.return_value = True
            io.is_foreground.return_value = True
            io.escape_pressed.return_value = False
            io.capture_region.return_value = image
            frames.return_value.get.side_effect = lambda sequence, barrier: Frame(sequence+1, barrier, .01, image)
            if configure is not None:
                configure(io)
            Session((0,0,320,320), 1, "unused", "wasd", MagicMock(), error, finish, stop).run()
            self.assertEqual(io.tap_action.call_count, expected_keys)
            frames.return_value.close.assert_called_once()
        return finish, error

    def test_brief_animation_error_does_not_run_full_banner_ocr(self):
        before = np.array([[2,2,8,16],[32,64,128,256],[512,1024,2048,4096],[8192,16384,32768,65536]])
        after, _, _ = slide_board(before, 2)
        after[0,3] = 2
        reader = MagicMock()
        reader.terminal_banner.return_value = None
        def configure(io):
            failed = False
            def read(image):
                nonlocal failed
                if not io.tap_action.called:
                    return before
                if not failed:
                    failed = True
                    raise RecognitionError("animation")
                return after
            reader.read.side_effect = read
        finish, error = self.run_mock_session(reader, 1, .05, configure)
        finish.assert_called_once()
        error.assert_not_called()
        reader.terminal_banner.assert_not_called()

    def test_stable_terminal_board_exits_without_a_key(self):
        reader = MagicMock()
        reader.read.return_value = np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])
        finish, error = self.run_mock_session(reader)
        finish.assert_called_once()
        error.assert_not_called()

    def test_persistent_game_over_overlay_still_exits(self):
        reader = MagicMock()
        reader.terminal_banner.return_value = "gameover"
        before = np.array([[2,2,0,0],[4,8,0,0],[16,32,0,0],[64,128,0,0]])
        def configure(io):
            def read(image):
                if io.tap_action.called:
                    raise RecognitionError("overlay")
                return before
            reader.read.side_effect = read
        finish, error = self.run_mock_session(reader, 1, .1, configure)
        finish.assert_called_once()
        error.assert_not_called()
        self.assertGreaterEqual(reader.terminal_banner.call_count, 2)

    def test_unreadable_screen_pauses_instead_of_exiting(self):
        reader = MagicMock()
        reader.read.side_effect = RecognitionError("uncertain")
        reader.terminal_banner.return_value = None
        finish, error = self.run_mock_session(reader)
        finish.assert_not_called()
        error.assert_called_once()

    def test_one_move_then_confirmed_terminal(self):
        before = np.array([[2,2,8,16],[32,64,128,256],[512,1024,2048,4096],[8192,16384,32768,65536]])
        after, _, _ = slide_board(before, 2)
        after[0,3] = 2
        reader = MagicMock()
        reader.read.side_effect = [before, before, after, after]
        finish, error = self.run_mock_session(reader, expected_keys=1)
        finish.assert_called_once()
        error.assert_not_called()



if __name__ == "__main__":
    unittest.main()
