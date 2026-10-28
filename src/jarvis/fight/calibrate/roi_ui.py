"""Region pickers for calibration.

``TkRoiPicker`` shows a captured Roblox frame in a window and lets the owner drag a box (Enter =
accept, S = skip, Esc = cancel). Tk renders PNG natively, so no Pillow dependency is needed.
``ScriptedRoiPicker`` replays predetermined boxes (tests, scripted re-calibration).
"""

from __future__ import annotations

import base64
from typing import Protocol

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h in frame pixels


class RoiPicker(Protocol):
    def select(self, image_bgr: np.ndarray, title: str, prompt: str, *, optional: bool = False) -> Box | None: ...


class CancelledByOwner(RuntimeError):
    pass


class ScriptedRoiPicker:
    def __init__(self, boxes: dict[str, Box | None]) -> None:
        self.boxes = boxes
        self.asked: list[str] = []

    def select(self, image_bgr: np.ndarray, title: str, prompt: str, *, optional: bool = False) -> Box | None:
        self.asked.append(title)
        return self.boxes.get(title)


class TkRoiPicker:
    def __init__(self, max_fraction: float = 0.85) -> None:
        self.max_fraction = max_fraction

    def select(self, image_bgr: np.ndarray, title: str, prompt: str, *, optional: bool = False) -> Box | None:
        import tkinter as tk

        root = tk.Tk()
        root.title(f"JARVIS calibration - {title}")
        root.attributes("-topmost", True)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        H, W = image_bgr.shape[:2]
        scale = min(1.0, (sw * self.max_fraction) / W, (sh * self.max_fraction - 80) / H)
        disp = cv2.resize(image_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", disp)
        if not ok:
            raise RuntimeError("could not encode the frame for display")
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        hint = "Drag a box, then press Enter."
        if optional:
            hint += "  Press S to skip this one."
        tk.Label(root, text=f"{prompt}\n{hint}  Esc cancels calibration.", justify="left", font=("Segoe UI", 11)).pack(anchor="w")
        canvas = tk.Canvas(root, width=disp.shape[1], height=disp.shape[0], cursor="crosshair")
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=photo)
        state: dict[str, object] = {"start": None, "rect": None, "box": None, "result": "pending"}

        def down(e):
            state["start"] = (e.x, e.y)
            if state["rect"]:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00e5ff", width=2)

        def drag(e):
            if state["start"] and state["rect"]:
                x0, y0 = state["start"]  # type: ignore[misc]
                canvas.coords(state["rect"], x0, y0, e.x, e.y)

        def up(e):
            if state["start"]:
                x0, y0 = state["start"]  # type: ignore[misc]
                x1, y1 = e.x, e.y
                state["box"] = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

        def accept(_e=None):
            if state["box"] and state["box"][2] >= 3 and state["box"][3] >= 2:  # type: ignore[index]
                state["result"] = "ok"
                root.destroy()

        def skip(_e=None):
            if optional:
                state["result"] = "skip"
                root.destroy()

        def cancel(_e=None):
            state["result"] = "cancel"
            root.destroy()

        canvas.bind("<ButtonPress-1>", down)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", up)
        root.bind("<Return>", accept)
        root.bind("<s>", skip)
        root.bind("<S>", skip)
        root.bind("<Escape>", cancel)
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.mainloop()
        if state["result"] == "cancel":
            raise CancelledByOwner(title)
        if state["result"] != "ok":
            return None
        x, y, w, h = state["box"]  # type: ignore[misc]
        return (int(x / scale), int(y / scale), max(1, int(w / scale)), max(1, int(h / scale)))
