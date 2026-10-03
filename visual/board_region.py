"""Locate a regular 4x4 tile grid inside a loosely selected screenshot."""
import cv2
import numpy as np


def _clusters(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or value - np.mean(groups[-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(group)) for group in groups]


def locate_board(image):
    """Return an image-relative bbox; reject missing or ambiguous tile grids.

    Geometry uses repeated square tile edges, not a site's background colors.
    The outer margin is inferred from the inter-tile gap. A slightly clipped
    outer border is accepted, but missing tile centers are not fabricated.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(gray, 20, 60),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = set()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) >= 20 and .85 <= w / h <= 1.18 and cv2.contourArea(contour) >= .65*w*h:
            boxes.add((x, y, w, h))
    candidates = []
    checked = []
    for _, _, tile_width, tile_height in sorted(boxes, key=lambda b: b[2], reverse=True):
        size = (tile_width + tile_height) / 2
        if any(abs(size - old) < .08*old for old in checked):
            continue
        checked.append(size)
        tiles = [b for b in boxes if abs(b[2]-size) < .1*size and abs(b[3]-size) < .1*size]
        xs = _clusters([b[0]+b[2]/2 for b in tiles], .15*size)
        ys = _clusters([b[1]+b[3]/2 for b in tiles], .15*size)
        for xi in range(len(xs)-3):
            for yi in range(len(ys)-3):
                xx, yy = np.array(xs[xi:xi+4]), np.array(ys[yi:yi+4])
                dx, dy = np.diff(xx), np.diff(yy)
                pitch = float(np.mean(np.r_[dx, dy]))
                if not 1.01*size <= pitch <= 1.5*size or np.max(np.abs(np.r_[dx, dy]-pitch)) > .06*pitch:
                    continue
                occupied = set()
                for x, y, w, h in tiles:
                    col, row = int(np.argmin(abs(xx-(x+w/2)))), int(np.argmin(abs(yy-(y+h/2))))
                    if abs(xx[col]-(x+w/2)) < .12*size and abs(yy[row]-(y+h/2)) < .12*size:
                        occupied.add((row, col))
                if len(occupied) < 12 or any(sum(r == row for r, _ in occupied) < 2 for row in range(4)) or any(sum(c == col for _, c in occupied) < 2 for col in range(4)):
                    continue
                margin = pitch - size/2
                bbox = (max(0, round(xx[0]-margin)), max(0, round(yy[0]-margin)),
                        min(width, round(xx[-1]+margin)), min(height, round(yy[-1]+margin)))
                candidates.append((pitch, bbox))
    candidates.sort(reverse=True)
    if not candidates:
        raise ValueError("未找到可靠的4×4棋盘网格，请包含完整棋盘重新框选。")
    best_size, best = candidates[0]
    for size, box in candidates[1:]:
        if size > best_size*.7 and max(abs(a-b) for a, b in zip(box, best)) > best_size*.2:
            raise ValueError("选区包含多个相近大小的棋盘，请缩小到目标棋盘附近。")
    return best
