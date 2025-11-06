#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adobe Scan – Python (Single File) — Fixed + Camera Capture
GUI: Tkinter
Image Ops: OpenCV + NumPy + Pillow
Export: PDF (Pillow)

What’s new in this version (Option C):
- FIX: Enhancement switching no longer stacks previews. "Original" truly shows original/cropped.
- FIX: _last_preview only used for "Add Page", not as the base for new previews.
- IMPROVED: "Magic Color" via LAB-CLAHE + mild unsharp masking.
- IMPROVED: Cleaner "Clean B&W" using adaptive threshold.
- NEW: Camera Scan (OpenCV window) — press SPACE to capture, ESC to cancel.
- Still fully single-file and deployable with PyInstaller.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps, ImageFilter
import numpy as np

# --- Try to import cv2 with a helpful message if missing ---
try:
    import cv2
except Exception as e:
    raise SystemExit(
        "OpenCV (cv2) is required.\nInstall with: pip install opencv-python\n\nDetails: " + str(e)
    )

APP_TITLE = "Adobe Scan – Python GUI (Single File)"
CANVAS_W, CANVAS_H = 920, 620
HANDLE_R = 7  # draggable handle radius


# ------------------- Helpers -------------------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def resize_to_fit(pil_img, max_w, max_h):
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h)
    if scale > 1:
        scale = min(scale, 1.0)  # never upscale
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil_img.resize(new_size, Image.LANCZOS), scale


def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_document(image_bgr):
    """
    Return 4-point contour of largest doc-like quad or None.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    # Preprocess small for speed
    ratio = 600.0 / max(h, w)
    small = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            pts /= ratio  # scale back to original size
            return order_points(pts)
    return None


def four_point_warp(image_bgr, pts):
    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    widthA = dist(br, bl)
    widthB = dist(tr, tl)
    maxW = int(max(widthA, widthB))

    heightA = dist(tr, br)
    heightB = dist(tl, bl)
    maxH = int(max(heightA, heightB))

    maxW = max(1, maxW)
    maxH = max(1, maxH)

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxW, maxH))
    return warped


# ------------------- Enhancements -------------------
def enhance_image_pil(pil_img, mode):
    """
    Enhancement in PIL/OpenCV as needed.
    Always returns RGB PIL image.
    """
    if mode == "Original":
        return pil_img.copy()

    if mode == "Grayscale":
        return ImageOps.grayscale(pil_img).convert("RGB")

    if mode == "Clean B&W":
        # Adaptive threshold for crisp documents
        img_cv = pil_to_cv(pil_img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 15
        )
        return Image.fromarray(bw).convert("RGB")

    if mode == "Magic Color":
        # LAB CLAHE on L channel, then mild unsharp mask
        img_cv = pil_to_cv(pil_img)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # unsharp: img*1.4 + (-gauss)*-0.4
        blur = cv2.GaussianBlur(bgr2, (0, 0), 1.0)
        sharp = cv2.addWeighted(bgr2, 1.4, blur, -0.4, 0)
        # Auto-contrast via PIL
        out = ImageOps.autocontrast(cv_to_pil(sharp))
        return out.convert("RGB")

    # Fallback
    return pil_img.copy()


# ------------------- Camera -------------------
def capture_from_camera(device_index=0, width=1280, height=720):
    """
    Opens a simple OpenCV window to capture a frame.
    Controls:
        SPACE - capture and return frame
        ESC   - cancel (return None)
    Returns: BGR np.array or None
    """
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        messagebox.showerror("Camera", "Could not open camera.")
        return None

    win = "Camera - SPACE: capture | ESC: cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    frame_captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        # Light guide rectangle
        h, w = overlay.shape[:2]
        margin = int(min(w, h) * 0.08)
        cv2.rectangle(overlay, (margin, margin), (w - margin, h - margin), (0, 255, 0), 2)
        cv2.putText(overlay, "Align document. Press SPACE to capture, ESC to cancel.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            frame_captured = None
            break
        if key == 32:  # SPACE
            frame_captured = frame.copy()
            break

    cap.release()
    cv2.destroyWindow(win)
    return frame_captured


# ------------------- App -------------------
class ScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{CANVAS_W+360}x{CANVAS_H+40}")
        self.minsize(CANVAS_W+360, CANVAS_H+40)

        # State
        self.orig_pil = None                    # original PIL (RGB)
        self.orig_cv = None                     # original OpenCV (BGR)
        self.display_pil = None                 # scaled image for canvas
        self.display_scale = 1.0
        self.corners_img = None                 # 4 points in original image coords
        self.corners_disp = None
        self.handle_ids = []
        self.poly_id = None
        self.dragging_idx = None

        self.page_stack = []                    # list of PIL pages (processed)
        self.enhance_mode = tk.StringVar(value="Magic Color")
        self._last_preview = None               # only for "Add Page", not as a filter base

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        # Left: canvas
        left = tk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="#111", width=CANVAS_W, height=CANVAS_H, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Right: controls
        right = tk.Frame(self, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # File controls
        ttk.Label(right, text="File").pack(anchor="w")
        ttk.Button(right, text="Import Image(s)", command=self.on_import).pack(fill=tk.X, pady=(2,6))
        ttk.Button(right, text="Camera Scan", command=self.on_camera_scan).pack(fill=tk.X, pady=(0,10))
        ttk.Button(right, text="Save Current Crop To Pages", command=self.on_add_page).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Export Pages as PDF", command=self.on_export_pdf).pack(fill=tk.X, pady=(2,12))

        # Detection
        ttk.Label(right, text="Detection & Crop").pack(anchor="w")
        ttk.Button(right, text="Auto-Detect Document", command=self.on_detect).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Auto-Crop Now", command=self.on_crop_now).pack(fill=tk.X, pady=(2,12))

        # Enhancement
        ttk.Label(right, text="Enhancement").pack(anchor="w")
        ttk.OptionMenu(right, self.enhance_mode, self.enhance_mode.get(),
                       "Magic Color", "Original", "Grayscale", "Clean B&W").pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Preview Enhancement", command=self.on_preview_enhancement).pack(fill=tk.X, pady=(2,12))

        # Pages
        ttk.Label(right, text="Pages").pack(anchor="w")
        self.pages_list = tk.Listbox(right, height=10)
        self.pages_list.pack(fill=tk.BOTH, expand=False)
        btns = tk.Frame(right)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Up", width=6, command=self.on_page_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Down", width=6, command=self.on_page_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Del", width=6, command=self.on_page_delete).pack(side=tk.LEFT, padx=2)
        ttk.Button(right, text="Clear All Pages", command=self.on_clear_pages).pack(fill=tk.X, pady=(2,12))

        # Footer
        self.status = tk.StringVar(value="Ready. Import an image or use Camera Scan to begin.")
        ttk.Label(right, textvariable=self.status, wraplength=320, foreground="#333").pack(fill=tk.X, pady=(20,0))

    # ---------- File ops ----------
    def on_import(self):
        paths = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff;*.tif")]
        )
        if not paths:
            return
        # load first image to work area; queue others as pages optionally
        self.load_pil(Image.open(paths[0]).convert("RGB"), label=os.path.basename(paths[0]))
        if len(paths) > 1 and messagebox.askyesno("Add Remaining As Pages?",
                                                  f"Also add the other {len(paths)-1} image(s) directly to Pages?"):
            for p in paths[1:]:
                try:
                    im = Image.open(p).convert("RGB")
                    self.page_stack.append(im)
                    self.pages_list.insert(tk.END, os.path.basename(p))
                except Exception as e:
                    messagebox.showwarning("Skip", f"Could not add {os.path.basename(p)}: {e}")

    def on_camera_scan(self):
        frame = capture_from_camera()
        if frame is None:
            self.status.set("Camera cancelled.")
            return
        # Convert BGR frame to PIL and load
        pil = cv_to_pil(frame)
        self.load_pil(pil, label="Captured Frame")
        # Auto-detect right away for convenience
        self.on_detect()

    def load_pil(self, pil_img, label="Image"):
        try:
            self.orig_pil = pil_img.convert("RGB")
            self.orig_cv = cv2.cvtColor(np.array(self.orig_pil), cv2.COLOR_RGB2BGR)
            self.corners_img = None
            self._last_preview = None
            self._render_image()
            self.status.set(f"Loaded: {label}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    # ---------- Rendering ----------
    def _render_image(self, show_overlay=True, preview_img=None):
        """
        Render the current image (orig_pil or preview_img) with optional overlay.
        """
        self.canvas.delete("all")
        img = preview_img if preview_img is not None else self.orig_pil
        if img is None:
            return

        disp, scale = resize_to_fit(img, CANVAS_W, CANVAS_H)
        self.display_pil = disp
        self.display_scale = scale

        self.tk_img = ImageTk.PhotoImage(disp)
        cx = (CANVAS_W - disp.size[0]) // 2
        cy = (CANVAS_H - disp.size[1]) // 2
        self.img_offset = (cx, cy)
        self.canvas.create_image(cx, cy, anchor=tk.NW, image=self.tk_img)

        if show_overlay and self.corners_img is not None:
            self._draw_overlay()

    def _draw_overlay(self):
        self.corners_disp = []
        cx, cy = self.img_offset
        for (x, y) in self.corners_img:
            dx = cx + x * self.display_scale
            dy = cy + y * self.display_scale
            self.corners_disp.append([dx, dy])

        # polygon
        if self.poly_id:
            self.canvas.delete(self.poly_id)
        flat = [coord for pt in self.corners_disp for coord in pt]
        self.poly_id = self.canvas.create_polygon(
            *flat, outline="#00FF88", width=2, fill="", dash=(4, 2)
        )

        # handles
        for hid in self.handle_ids:
            self.canvas.delete(hid)
        self.handle_ids = []
        for (dx, dy) in self.corners_disp:
            hid = self.canvas.create_oval(
                dx - HANDLE_R, dy - HANDLE_R, dx + HANDLE_R, dy + HANDLE_R,
                fill="#00FF88", outline="#002215", width=2
            )
            self.handle_ids.append(hid)

    # ---------- Detection / Crop ----------
    def on_detect(self):
        if self.orig_cv is None:
            return
        pts = detect_document(self.orig_cv)
        if pts is None:
            # fallback to near-borders
            h, w = self.orig_cv.shape[:2]
            margin = int(min(w, h) * 0.03)
            pts = np.array([[margin, margin],
                            [w - margin, margin],
                            [w - margin, h - margin],
                            [margin, h - margin]], dtype=np.float32)
            messagebox.showinfo("Info", "Auto-detect failed. Drag the four handles to fit your document.")
        self.corners_img = pts.tolist()
        self._render_image()
        self.status.set("Document corners ready. Drag handles to refine.")

    def on_crop_now(self):
        if self.orig_cv is None:
            return
        if not self.corners_img:
            self.on_detect()
            if not self.corners_img:
                return
        warped = four_point_warp(self.orig_cv, np.array(self.corners_img, dtype=np.float32))
        pil = cv_to_pil(warped)
        self._render_image(preview_img=pil)
        self._last_preview = pil  # preview for quick add
        self.status.set("Preview: cropped. Click 'Save Current Crop To Pages' to add.")

    # ---------- Enhancements ----------
    def _get_fresh_base(self):
        """
        Always regenerate the base image fresh (NO reuse of _last_preview).
        If corners exist -> use cropped warp; else original.
        """
        if self.orig_cv is not None and self.corners_img:
            warped = four_point_warp(self.orig_cv, np.array(self.corners_img, dtype=np.float32))
            return cv_to_pil(warped)
        return self.orig_pil if self.orig_pil is not None else None

    def on_preview_enhancement(self):
        base = self._get_fresh_base()
        if base is None:
            return
        mode = self.enhance_mode.get()
        out = enhance_image_pil(base, mode)
        self._render_image(preview_img=out)
        self._last_preview = out  # only to expedite "Add Page"
        self.status.set(f"Preview: {mode}")

    # ---------- Draggable Handles ----------
    def _find_handle_under(self, x, y):
        for i, (dx, dy) in enumerate(self.corners_disp or []):
            if (dx - x) ** 2 + (dy - y) ** 2 <= (HANDLE_R + 2) ** 2:
                return i
        return None

    def on_press(self, event):
        if not self.corners_img or not self.corners_disp:
            return
        idx = self._find_handle_under(event.x, event.y)
        if idx is not None:
            self.dragging_idx = idx

    def on_drag(self, event):
        if self.dragging_idx is None or not self.corners_img:
            return
        cx, cy = self.img_offset
        # clamp to image box
        x = max(cx, min(event.x, cx + self.display_pil.size[0]))
        y = max(cy, min(event.y, cy + self.display_pil.size[1]))
        # map back to image coords
        ix = (x - cx) / self.display_scale
        iy = (y - cy) / self.display_scale
        self.corners_img[self.dragging_idx] = [ix, iy]
        self._render_image()

    def on_release(self, _event):
        self.dragging_idx = None

    # ---------- Pages ----------
    def on_add_page(self):
        """
        Add current view as a page with chosen enhancement.
        Uses latest preview if available; else recomputes fresh base + enhancement.
        """
        page = self._last_preview
        if page is None:
            base = self._get_fresh_base()
            if base is None:
                return
            page = enhance_image_pil(base, self.enhance_mode.get())

        self.page_stack.append(page.convert("RGB"))
        self.pages_list.insert(tk.END, f"Page {len(self.page_stack)}")
        self.status.set(f"Added page #{len(self.page_stack)}")

    def on_page_up(self):
        i = self.pages_list.curselection()
        if not i:
            return
        i = i[0]
        if i == 0:
            return
        self.page_stack[i - 1], self.page_stack[i] = self.page_stack[i], self.page_stack[i - 1]
        name = self.pages_list.get(i)
        self.pages_list.delete(i)
        self.pages_list.insert(i - 1, name)
        self.pages_list.selection_set(i - 1)

    def on_page_down(self):
        i = self.pages_list.curselection()
        if not i:
            return
        i = i[0]
        if i >= len(self.page_stack) - 1:
            return
        self.page_stack[i + 1], self.page_stack[i] = self.page_stack[i], self.page_stack[i + 1]
        name = self.pages_list.get(i)
        self.pages_list.delete(i)
        self.pages_list.insert(i + 1, name)
        self.pages_list.selection_set(i + 1)

    def on_page_delete(self):
        i = self.pages_list.curselection()
        if not i:
            return
        i = i[0]
        self.pages_list.delete(i)
        del self.page_stack[i]
        # rename listbox labels
        self.pages_list.delete(0, tk.END)
        for idx in range(len(self.page_stack)):
            self.pages_list.insert(tk.END, f"Page {idx + 1}")
        self.status.set("Deleted selected page.")

    def on_clear_pages(self):
        if not self.page_stack:
            return
        if messagebox.askyesno("Confirm", "Clear all pages?"):
            self.page_stack.clear()
            self.pages_list.delete(0, tk.END)
            self.status.set("Cleared all pages.")

    def on_export_pdf(self):
        if not self.page_stack:
            messagebox.showinfo("Nothing to export", "Add at least one page.")
            return
        path = filedialog.asksaveasfilename(
            title="Export PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not path:
            return
        try:
            pages = [p.convert("RGB") for p in self.page_stack]
            if len(pages) == 1:
                pages[0].save(path, "PDF", resolution=300.0)
            else:
                pages[0].save(path, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])
            self.status.set(f"Exported {len(pages)} page(s) to: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Saved PDF: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


def main():
    # High-DPI scaling on Windows (optional best effort)
    try:
        import ctypes
        if sys.platform.startswith("win"):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = ScanApp()
    app.mainloop()


if __name__ == "__main__":
    main()
