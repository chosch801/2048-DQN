"""Region-selecting Windows launcher for the visual 2048 player."""

import ctypes
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import ImageTk

from visual.desktop_io import capture_region, set_dpi_awareness, window_at_point


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "v3_1" / "dqn_ep20000.pth"


class DesktopPlayer:
    def __init__(self, root):
        self.root = root
        self.region = None
        self.hwnd = None
        self.worker = None
        self.stop_event = threading.Event()
        self.preview_valid = False
        self.closing = False
        self.finished = False
        self.photo = None
        root.title("2048 视觉玩家")
        root.geometry("640x740")
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="大致框住完整的4×4棋盘，可留一圈空白，程序自动校准网格。\n先检查识别结果，再点击开始；运行期间请保持游戏可见。",
                  wraplength=600).pack(anchor="w")
        self.select_button = ttk.Button(frame, text="① 框选游戏区域", command=self.select_region)
        self.select_button.pack(fill="x", pady=10)
        self.region_text = tk.StringVar(value="尚未选择区域")
        ttk.Label(frame, textvariable=self.region_text).pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=8)
        self.checkpoint = tk.StringVar(value=str(DEFAULT_CHECKPOINT))
        self.checkpoint_entry = ttk.Entry(row, textvariable=self.checkpoint)
        self.checkpoint_entry.pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(row, text="选择模型", command=self.browse)
        self.browse_button.pack(side="right")
        row = ttk.Frame(frame)
        row.pack(fill="x")
        self.key_mode = tk.StringVar(value="wasd")
        self.key_buttons = []
        for label, value in (("WASD", "wasd"), ("方向键", "arrows")):
            button = ttk.Radiobutton(row, text=label, variable=self.key_mode, value=value)
            button.pack(side="left", padx=8)
            self.key_buttons.append(button)
        self.start_button = ttk.Button(frame, text="② 识别正确，开始自动操作", command=self.start, state="disabled")
        self.start_button.pack(fill="x", pady=8)
        self.pause_button = ttk.Button(frame, text="暂停（运行时也可按 Esc）", command=self.pause, state="disabled")
        self.pause_button.pack(fill="x")
        self.status = tk.StringVar(value="开始后倒计时 3 秒；识别异常会暂停并保留截图，确认终局后关闭助手。")
        ttk.Label(frame, textvariable=self.status, wraplength=600).pack(anchor="w", pady=10)
        self.image_label = ttk.Label(frame)
        self.image_label.pack()
        self.board_text = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.board_text, font=("Consolas", 13)).pack(pady=8)

    def busy(self, active):
        state = "disabled" if active else "normal"
        for widget in (self.select_button, self.browse_button, self.checkpoint_entry, *self.key_buttons):
            widget.configure(state=state)
        self.start_button.configure(state="normal" if not active and self.preview_valid else "disabled")

    def show_image(self, image):
        if image is not None:
            displayed = image.copy()
            displayed.thumbnail((420, 340))
            self.photo = ImageTk.PhotoImage(displayed)
            self.image_label.configure(image=self.photo)

    def show_status(self, text, image=None, board=None):
        self.status.set(text)
        self.show_image(image)
        if board is not None:
            self.board_text.set("\n".join(" ".join(f"{int(value):5d}" for value in row) for row in board))

    def browse(self):
        path = filedialog.askopenfilename(title="选择检查点", filetypes=[("PyTorch checkpoint", "*.pth"), ("All", "*.*")])
        if path:
            self.checkpoint.set(path)

    def select_region(self):
        self.preview_valid = False
        self.busy(True)
        self.root.withdraw()
        self.root.after(250, self.open_selector)

    def open_selector(self):
        try:
            api = ctypes.windll.user32
            left, top, width, height = (api.GetSystemMetrics(index) for index in (76, 77, 78, 79))
            screenshot = capture_region((left, top, left + width, top + height))
            overlay = tk.Toplevel(self.root)
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.geometry(f"{width}x{height}+0+0")
            canvas = tk.Canvas(overlay, highlightthickness=0, cursor="crosshair")
            canvas.pack(fill="both", expand=True)
            photo = ImageTk.PhotoImage(screenshot)
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo
            canvas.create_text(width // 2, 30, text="拖动框住完整棋盘，可留空白；Esc 取消", fill="red", font=("Microsoft YaHei", 18, "bold"))
            overlay.update_idletasks()
            # Tk's negative geometry offsets mean distance from the right/bottom;
            # SetWindowPos instead uses physical virtual-desktop coordinates.
            api.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            api.SetWindowPos(ctypes.c_void_p(overlay.winfo_id()), ctypes.c_void_p(-1), left, top, width, height, 0x0040)
            origin = []
            rectangle = [None]

            def cancel(event=None):
                overlay.destroy()
                self.root.deiconify()
                self.busy(False)
                self.status.set("框选已取消。")

            def press(event):
                origin[:] = [event.x, event.y]
                if rectangle[0] is not None:
                    canvas.delete(rectangle[0])
                rectangle[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff88", width=3)

            def drag(event):
                if origin:
                    canvas.coords(rectangle[0], *origin, event.x, event.y)

            def release(event):
                if not origin:
                    return
                x0, x1 = sorted((max(0, min(width, origin[0])), max(0, min(width, event.x))))
                y0, y1 = sorted((max(0, min(height, origin[1])), max(0, min(height, event.y))))
                if x1 - x0 < 64 or y1 - y0 < 64:
                    return
                self.region = (left + x0, top + y0, left + x1, top + y1)
                selected = screenshot.crop((x0, y0, x1, y1))
                overlay.destroy()
                self.root.after(100, lambda: self.region_selected(selected))

            canvas.bind("<ButtonPress-1>", press)
            canvas.bind("<B1-Motion>", drag)
            canvas.bind("<ButtonRelease-1>", release)
            overlay.bind("<Escape>", cancel)
            overlay.focus_force()
        except Exception as exc:
            self.root.deiconify()
            self.busy(False)
            self.status.set(f"无法框选：{exc}")

    def region_selected(self, image):
        from visual.board_region import locate_board
        try:
            box = locate_board(image)
            left, top, _, _ = self.region
            self.region = (left + box[0], top + box[1], left + box[2], top + box[3])
            image = image.crop(box)
        except ValueError:
            # Exact manual crops remain supported for themes without clear edges.
            pass
        left, top, right, bottom = self.region
        self.hwnd = window_at_point((left + right) // 2, (top + bottom) // 2)
        self.root.deiconify()
        self.region_text.set(f"区域：{self.region}；目标窗口：{self.hwnd}")
        self.show_status("正在识别预览，请稍候……", image)

        def read():
            try:
                from visual.visual_board import BoardReader
                board = BoardReader().read(image)
                self.root.after(0, lambda: self.preview_done(image, board))
            except Exception as exc:
                text = str(exc)
                self.root.after(0, lambda: self.preview_failed(text, image))

        self.worker = threading.Thread(target=read, daemon=True)
        self.worker.start()

    def preview_done(self, image, board):
        self.preview_valid = bool(self.hwnd)
        self.busy(False)
        self.show_status("请逐格核对下面的识别结果；正确后点击开始。" if self.hwnd else "未找到目标窗口，请重新框选。", image, board)

    def preview_failed(self, text, image):
        self.preview_valid = False
        self.busy(False)
        self.board_text.set("")
        self.show_status(f"识别暂停：{text}。请检查截图并重新框选。", image)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        checkpoint = Path(self.checkpoint.get())
        if not checkpoint.is_file():
            self.status.set("模型文件不存在，请重新选择。")
            return
        if not self.preview_valid:
            return
        self.stop_event = threading.Event()
        self.busy(True)
        self.pause_button.configure(state="normal")
        region, hwnd, key_mode = self.region, self.hwnd, self.key_mode.get()

        def status(text, image=None, board=None):
            self.root.after(0, lambda: self.show_status(text, image, board))

        def error(text, image=None):
            self.root.after(0, lambda: self.session_error(text, image))

        def run():
            try:
                from visual.visual_session import Session
                session = Session(region=region, hwnd=hwnd, checkpoint=str(checkpoint), key_mode=key_mode,
                                  on_status=status, on_error=error,
                                  on_finished=lambda: self.root.after(0, self.session_finished), stop_event=self.stop_event)
                session.run()
            except Exception as exc:
                error(str(exc))
            finally:
                self.root.after(0, self.session_stopped)

        self.root.iconify()
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def session_error(self, text, image=None):
        self.preview_valid = False
        self.root.deiconify()
        self.root.lift()
        self.show_status(f"已暂停：{text}。修正后请重新框选并确认。", image)

    def session_finished(self):
        self.finished = True
        self.closing = True
        self.stop_event.set()
        self.root.after(100, self.wait_close)

    def session_stopped(self):
        if self.finished or self.closing:
            return
        self.busy(False)
        self.pause_button.configure(state="disabled")
        self.root.deiconify()

    def pause(self):
        self.stop_event.set()
        self.status.set("正在暂停，等待当前计算或按键释放完成……")

    def close(self):
        self.closing = True
        self.stop_event.set()
        self.root.withdraw()
        self.wait_close()

    def wait_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(100, self.wait_close)
        else:
            self.root.destroy()


def main():
    set_dpi_awareness()
    root = tk.Tk()
    DesktopPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
