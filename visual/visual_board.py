"""Read a calibrated 4x4 board; uncertain cells are never treated as empty."""
import hashlib

import cv2
import numpy as np


class RecognitionError(RuntimeError):
    pass


class BoardReader:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        self.glyph_cache = {}
        self.last_image_key = None
        self.last_board = None

    def read(self, image):
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        if min(h, w) < 160 or not 0.85 <= w / h <= 1.18:
            raise RecognitionError("请框选完整、近似正方形的4×4棋盘，宽高至少160像素。")
        if max(h, w) > 640:
            rgb = cv2.resize(rgb, (round(w * 640 / max(h,w)), round(h * 640 / max(h,w))), interpolation=cv2.INTER_AREA)
            h, w = rgb.shape[:2]
        image_key = (rgb.shape, hashlib.blake2b(rgb.tobytes(), digest_size=16).digest())
        if image_key == self.last_image_key:
            return self.last_board.copy()
        board = np.zeros((4, 4), dtype=np.int64)
        crops, locations = [], []
        for row in range(4):
            for col in range(4):
                x0, x1 = int((col + .08) * w / 4), int((col + .92) * w / 4)
                y0, y1 = int((row + .08) * h / 4), int((row + .92) * h / 4)
                cell = rgb[y0:y1, x0:x1]
                background = np.median(cell.reshape(-1, 3), axis=0)
                distance = np.max(np.abs(cell.astype(float) - background), axis=2)
                if distance.max() < 18:
                    continue
                # Glow changes slowly across a cell; printed strokes have sharp edges.
                detail = np.max(np.abs(cell.astype(np.float32) - cv2.GaussianBlur(cell.astype(np.float32), (0,0), 2)), axis=2)
                edge_y, edge_x = max(1, cell.shape[0] // 8), max(1, cell.shape[1] // 8)
                if detail[edge_y:-edge_y, edge_x:-edge_x].max() < 6:
                    continue
                mask = (distance > max(20, np.percentile(distance, 98) * .35)).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                clean = np.zeros_like(mask)
                for label in range(1, count):
                    x, y, cw, ch, area = stats[label]
                    if area >= 3 and x > 0 and y > 0 and x + cw < cell.shape[1] and y + ch < cell.shape[0]:
                        clean[labels == label] = 1
                ys, xs = np.where(clean)
                if not len(xs):
                    raise RecognitionError(f"第{row+1}行第{col+1}格存在纹理或裁切，无法确认数字/空格。")
                left, right, top, bottom = xs.min(), xs.max()+1, ys.min(), ys.max()+1
                binary = (255 - clean[top:bottom, left:right] * 255).astype(np.uint8)
                binary = cv2.copyMakeBorder(binary, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)
                key = (binary.shape, binary.tobytes())
                if key in self.glyph_cache:
                    board[row,col] = self.glyph_cache[key]
                    continue
                crops.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
                locations.append((row, col, key))
        if crops:
            results, _ = self.ocr.text_rec(crops)
            for (row, col, key), (text, confidence) in zip(locations, results):
                text = text.strip()
                if not text.isascii() or not text.isdigit() or confidence < .92:
                    raise RecognitionError(f"第{row+1}行第{col+1}格识别不确定：{text!r}，置信度{confidence:.2f}。")
                value = int(text)
                if value < 2 or value > 65536 or value & (value - 1):
                    raise RecognitionError(f"第{row+1}行第{col+1}格不是支持的2次幂：{text}。")
                board[row, col] = value
                if len(self.glyph_cache) >= 512:
                    self.glyph_cache.pop(next(iter(self.glyph_cache)))
                self.glyph_cache[key] = value
        if np.count_nonzero(board) < 2:
            raise RecognitionError("识别到的方块过少，请核对框选区域和游戏是否已经开始。")
        self.last_image_key, self.last_board = image_key, board.copy()
        return board

    def terminal_banner(self, image):
        results, _ = self.ocr(np.asarray(image.convert("RGB"))[:, :, ::-1].copy(), use_cls=False)
        for _, text, confidence in results or []:
            normalized = "".join(text.lower().split()).strip("!.。！")
            if confidence >= .95 and normalized in ("gameover", "游戏结束"):
                return normalized
        return None
