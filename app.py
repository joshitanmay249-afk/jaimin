"""
app.py  –  Simple Handwritten Digit Recognition
Draw on the black canvas → prediction appears instantly on mouse release.
Right-click (or press C) to clear the canvas.
"""

import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageDraw
import joblib
import cv2
import mysql.connector

from config import DB_CONFIG, MODEL_PATH
from db_setup import setup as db_setup


# ── Database ──────────────────────────────────────────────────────────────────
class Database:
    def __init__(self):
        self._conn   = mysql.connector.connect(**DB_CONFIG)
        self._cursor = self._conn.cursor()

    def save(self, digit, confidence):
        self._cursor.execute(
            "INSERT INTO predictions (predicted, confidence) VALUES (%s, %s)",
            (digit, round(confidence * 100, 2))
        )
        self._conn.commit()

    def close(self):
        self._cursor.close()
        if self._conn.is_connected():
            self._conn.close()


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    CANVAS_PX   = 280          # on-screen canvas size
    GRID        = 28           # MNIST 28×28
    CELL        = CANVAS_PX // GRID   # 10 px per cell
    PEN_R       = 13           # brush radius

    def __init__(self, root):
        self.root  = root
        root.title("Digit Recognition")
        root.configure(bg="#111111")
        root.resizable(False, False)

        self.model = joblib.load(MODEL_PATH)
        self.db    = Database()

        # PIL internal image (28×28)
        self._img  = Image.new("L", (self.GRID, self.GRID), 0)
        self._draw = ImageDraw.Draw(self._img)
        self._last = None

        self._build()
        root.protocol("WM_DELETE_WINDOW", self._quit)
        root.bind("<c>", lambda e: self._clear())
        root.bind("<C>", lambda e: self._clear())

    def _build(self):
        frame = tk.Frame(self.root, bg="#111111")
        frame.pack(padx=24, pady=24)

        # ── Canvas ────────────────────────────────────────────────────────
        canvas_wrap = tk.Frame(frame, bg="#444444", padx=2, pady=2)
        canvas_wrap.pack(side="left")

        self.canvas = tk.Canvas(
            canvas_wrap,
            width=self.CANVAS_PX, height=self.CANVAS_PX,
            bg="#000000", cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._press)
        self.canvas.bind("<B1-Motion>",        self._drag)
        self.canvas.bind("<ButtonRelease-1>",  self._release)
        self.canvas.bind("<ButtonPress-3>",    lambda e: self._clear())   # right-click clear

        # ── Divider ───────────────────────────────────────────────────────
        tk.Frame(frame, bg="#333333", width=2, height=self.CANVAS_PX).pack(
            side="left", padx=20, pady=0, fill="y"
        )

        # ── Result display ────────────────────────────────────────────────
        right = tk.Frame(frame, bg="#111111", width=160)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Predicted", font=("Segoe UI", 13),
                 bg="#111111", fg="#666666").pack(pady=(30, 0))

        self.digit_lbl = tk.Label(
            right, text="?",
            font=("Consolas", 100, "bold"),
            bg="#111111", fg="#89b4fa",
            width=2
        )
        self.digit_lbl.pack()

        self.conf_lbl = tk.Label(
            right, text="",
            font=("Segoe UI", 11),
            bg="#111111", fg="#888888"
        )
        self.conf_lbl.pack()

        tk.Label(right, text="Right-click or\npress C to clear",
                 font=("Segoe UI", 8),
                 bg="#111111", fg="#444444").pack(pady=(20, 0))

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _press(self, e):
        self._last = (e.x, e.y)
        self._paint(e.x, e.y)

    def _drag(self, e):
        if self._last:
            x0, y0 = self._last
            x1, y1 = e.x, e.y
            steps = max(int(max(abs(x1-x0), abs(y1-y0)) // 4), 1)
            for i in range(steps + 1):
                t = i / steps
                self._paint(int(x0+(x1-x0)*t), int(y0+(y1-y0)*t))
        self._last = (e.x, e.y)

    def _release(self, e):
        self._last = None
        # Auto-predict
        if np.array(self._img).max() > 0:
            threading.Thread(target=self._infer, daemon=True).start()

    def _paint(self, cx, cy):
        r = self.PEN_R
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                fill="#ffffff", outline="")
        px, py, pr = cx/self.CELL, cy/self.CELL, r/self.CELL
        self._draw.ellipse([px-pr, py-pr, px+pr, py+pr], fill=255)

    # ── Clear ─────────────────────────────────────────────────────────────────
    def _clear(self):
        self.canvas.delete("all")
        self._img  = Image.new("L", (self.GRID, self.GRID), 0)
        self._draw = ImageDraw.Draw(self._img)
        self.digit_lbl.config(text="?", fg="#89b4fa")
        self.conf_lbl.config(text="")

    # ── Inference ─────────────────────────────────────────────────────────────
    def _infer(self):
        img = np.array(self._img, dtype=np.float32)

        # Centre & crop
        rows, cols = np.any(img>0, axis=1), np.any(img>0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0,-1]]
            c0, c1 = np.where(cols)[0][[0,-1]]
            crop = img[r0:r1+1, c0:c1+1]
            h, w = crop.shape
            size = max(h, w)
            pad  = int(size * 0.2)
            sq   = np.zeros((size+2*pad, size+2*pad), np.float32)
            top  = pad + (size-h)//2
            left = pad + (size-w)//2
            sq[top:top+h, left:left+w] = crop
            img  = sq

        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) / 255.0
        X   = img.flatten().reshape(1, -1)

        proba = self.model.predict_proba(X)[0]
        digit = int(np.argmax(proba))
        conf  = float(proba[digit])

        try:
            self.db.save(digit, conf)
        except Exception:
            pass

        self.root.after(0, self._show, digit, conf)

    def _show(self, digit, conf):
        self.digit_lbl.config(text=str(digit))
        self.conf_lbl.config(text=f"{conf*100:.1f}%")

    # ── Quit ──────────────────────────────────────────────────────────────────
    def _quit(self):
        self.db.close()
        self.root.destroy()


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db_setup()
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App(root)
    root.mainloop()
