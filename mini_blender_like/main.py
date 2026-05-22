from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

from .scene import Scene, Object3D, Material
from .obj_io import load_obj
from .renderer_sdf import RenderConfig, render
from .viewports import project_points_ortho


# -----------------------------
# Theme palette (soft "Blender-like")
# -----------------------------
BG = "#2b2b2b"          # main background
PANEL = "#303030"       # panel background (ttk uses style; for tk widgets use directly)
FIELD = "#3a3a3a"       # entry/list background
FG = "#e6e6e6"          # text
ACCENT = "#3a79ff"      # selection accent
MPL_BG = "#2f2f2f"      # matplotlib figure bg (close to panels)


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp_pos(x: float, lim: float = 999.0) -> float:
    return float(max(-lim, min(lim, x)))


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [int(max(0.0, min(1.0, c)) * 255) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) != 6:
        return (0.8, 0.8, 0.8)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (float(r), float(g), float(b))


def _mode_axes(mode: str) -> Tuple[int, int]:
    if mode == "XY":
        return (0, 1)
    if mode == "XZ":
        return (0, 2)
    if mode == "YZ":
        return (1, 2)
    raise ValueError("Unknown ortho mode")


def _rotation_matrix_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def _scene_bounds(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []

    for o in scene.objects:
        if o.kind == "plane":
            continue

        if o.kind == "sphere":
            c = np.array(o.sdf_params.get("center", (0, 0, 0)), dtype=np.float64)
            r = float(o.sdf_params.get("radius", 1.0))
            mins.append(c - r)
            maxs.append(c + r)

        elif o.kind == "box":
            c = np.array(o.sdf_params.get("center", (0, 0, 0)), dtype=np.float64)
            hs = np.array(o.sdf_params.get("half_size", (1, 1, 1)), dtype=np.float64)
            mins.append(c - hs)
            maxs.append(c + hs)

        elif o.kind == "cylinder":
            c = np.array(o.sdf_params.get("center", (0, 0, 0)), dtype=np.float64)
            r = float(o.sdf_params.get("radius", 1.0))
            hh = float(o.sdf_params.get("half_height", 1.0))
            mins.append(c - np.array([r, hh, r]))
            maxs.append(c + np.array([r, hh, r]))

        elif o.kind == "torus":
            c = np.array(o.sdf_params.get("center", (0, 0, 0)), dtype=np.float64)
            R = float(o.sdf_params.get("major_radius", 1.0))
            r = float(o.sdf_params.get("minor_radius", 0.25))
            ext = R + r
            mins.append(c - np.array([ext, r, ext]))
            maxs.append(c + np.array([ext, r, ext]))

        elif o.kind == "mesh_obj" and o.vertices is not None and o.vertices.size:
            mins.append(o.vertices.min(axis=0))
            maxs.append(o.vertices.max(axis=0))

    if not mins:
        return np.array([-1, -1, -1], dtype=np.float64), np.array([1, 1, 1], dtype=np.float64)

    mn = np.min(np.vstack(mins), axis=0)
    mx = np.max(np.vstack(maxs), axis=0)
    return mn, mx


def _fit_camera_for_scene(scene: Scene, yaw_deg: float, pitch_deg: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    mn, mx = _scene_bounds(scene)
    target = (mn + mx) / 2.0
    radius = float(np.linalg.norm(mx - mn) * 0.5 + 1e-6)
    radius = max(radius, 1.0)

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    dir_vec = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

    dist = radius * 2.6
    cam = target + dir_vec * dist
    max_dist = dist + radius * 3.0
    return (float(cam[0]), float(cam[1]), float(cam[2])), (float(target[0]), float(target[1]), float(target[2])), float(max_dist)


# ----------------------------
# Scrollable frame helper
# ----------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=480, bg=BG, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # IMPORTANT: use explicit bg to avoid ttk background issues
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width, bg=bg)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# ----------------------------
# App
# ----------------------------
class MiniBlenderLike(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Мини 3D-редактор")
        self.geometry("1650x950")
        self.minsize(1400, 820)

        self._apply_theme()

        self.scene = Scene(objects=[], background_path=None)
        self._seed_scene()

        # state
        self.status = tk.StringVar(value="Готово.")
        self.out_dir = tk.StringVar(value=os.path.abspath("outputs"))
        self.viewport_mode = tk.StringVar(value="3D")

        # Camera (viewport)
        self.cam_yaw = tk.DoubleVar(value=-60.0)
        self.cam_pitch = tk.DoubleVar(value=20.0)
        self.cam_dist = tk.DoubleVar(value=10.0)
        self.show_axes = tk.BooleanVar(value=False)

        # Render config UI
        self.r_w = tk.IntVar(value=900)
        self.r_h = tk.IntVar(value=520)
        self.r_steps = tk.IntVar(value=200)
        self.r_fov = tk.DoubleVar(value=55.0)
        self.r_yaw = tk.DoubleVar(value=-45.0)
        self.r_pitch = tk.DoubleVar(value=15.0)

        # Add object
        self.new_kind = tk.StringVar(value="sphere")
        self.add_x = tk.DoubleVar(value=0.0)
        self.add_y = tk.DoubleVar(value=0.0)
        self.add_z = tk.DoubleVar(value=0.0)
        self.add_a = tk.DoubleVar(value=1.0)
        self.add_b = tk.DoubleVar(value=0.8)
        self.add_c = tk.DoubleVar(value=0.8)
        self.add_d = tk.DoubleVar(value=0.8)
        self.add_plane_nx = tk.DoubleVar(value=0.0)
        self.add_plane_ny = tk.DoubleVar(value=1.0)
        self.add_plane_nz = tk.DoubleVar(value=0.0)
        self.add_plane_h = tk.DoubleVar(value=1.2)
        self.add_color_hex = tk.StringVar(value="#3aa0ff")
        self.add_tex_path = tk.StringVar(value="")

        # Edit selected
        self.edit_name = tk.StringVar(value="")
        self.edit_kind = tk.StringVar(value="Нет выбора")
        self.edit_x = tk.DoubleVar(value=0.0)
        self.edit_y = tk.DoubleVar(value=0.0)
        self.edit_z = tk.DoubleVar(value=0.0)
        self.edit_a = tk.DoubleVar(value=1.0)
        self.edit_b = tk.DoubleVar(value=0.8)
        self.edit_c = tk.DoubleVar(value=0.8)
        self.edit_d = tk.DoubleVar(value=0.8)
        self.edit_plane_nx = tk.DoubleVar(value=0.0)
        self.edit_plane_ny = tk.DoubleVar(value=1.0)
        self.edit_plane_nz = tk.DoubleVar(value=0.0)
        self.edit_plane_h = tk.DoubleVar(value=1.2)
        self.edit_color_hex = tk.StringVar(value="#3aa0ff")
        self.edit_tex_path = tk.StringVar(value="")

        # Matplotlib
        self.fig = plt.Figure(figsize=(10.6, 7.8), dpi=100)
        self.ax = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self._last_render_img: Optional[np.ndarray] = None

        # Drag state 2D
        self._drag_active = False
        self._drag_idx: Optional[int] = None
        self._drag_last_xy: Optional[Tuple[float, float]] = None
        self._drag_mode: str = "XY"
        self._drag_action: str = "move"  # move | rotate

        # Drag state 3D
        self._drag3d_active = False
        self._drag3d_idx: Optional[int] = None
        self._drag3d_last_px: Optional[Tuple[float, float]] = None
        self._drag3d_action: str = "move_xy"  # move_xy | move_depth

        self._build_ui()
        self._refresh_object_list()
        self._reload_selected_into_edit()
        self.draw_viewport()

        self.canvas.mpl_connect("scroll_event", self._on_mpl_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_mpl_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mpl_release)

    # ------------------ Theme ------------------
    def _apply_theme(self):
        self.configure(bg=BG)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=BG, foreground=FG, fieldbackground=FIELD)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=FG)
        style.configure("TLabelframe.Label", background=BG, foreground=FG)
        style.configure("TButton", padding=6)
        style.configure("TEntry", fieldbackground=FIELD, foreground=FG)
        style.configure("TSpinbox", fieldbackground=FIELD, foreground=FG)
        style.map("TButton", background=[("active", "#3a3a3a")])

        # Matplotlib defaults for dark-ish viewport
        plt.rcParams["figure.facecolor"] = MPL_BG
        plt.rcParams["axes.facecolor"] = MPL_BG
        plt.rcParams["axes.labelcolor"] = FG
        plt.rcParams["text.color"] = FG
        plt.rcParams["xtick.color"] = "#cfcfcf"
        plt.rcParams["ytick.color"] = "#cfcfcf"
        plt.rcParams["grid.color"] = "#555555"

    # ------------------ Seed scene ------------------
    def _seed_scene(self):
        self.scene.clear()
        self.scene.add(Object3D(
            name="Сфера",
            kind="sphere",
            material=Material(name="mat_sphere", base_color=(0.23, 0.63, 0.95)),
            sdf_params={"center": (0.0, 0.2, 0.0), "radius": 1.1},
        ))
        self.scene.add(Object3D(
            name="Куб",
            kind="box",
            material=Material(name="mat_box", base_color=(0.75, 0.35, 0.95)),
            sdf_params={"center": (-2.2, -0.2, -0.2), "half_size": (0.7, 0.7, 0.7)},
        ))
        self.scene.add(Object3D(
            name="Пол",
            kind="plane",
            material=Material(name="mat_plane", base_color=(0.70, 0.70, 0.70)),
            sdf_params={"normal": (0.0, 1.0, 0.0), "h": 1.45},
        ))

    # ------------------ UI ------------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Toolbar
        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 10))

        ttk.Button(toolbar, text="Новая сцена", command=self._on_new_scene).pack(side="left")
        ttk.Button(toolbar, text="Импорт OBJ", command=self.import_obj_clicked).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Фон…", command=self.choose_background).pack(side="left")
        ttk.Button(toolbar, text="Очистить фон", command=self.clear_background).pack(side="left", padx=6)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(toolbar, text="Вид:").pack(side="left", padx=(0, 6))
        ttk.OptionMenu(
            toolbar, self.viewport_mode, self.viewport_mode.get(),
            "3D", "XY", "XZ", "YZ",
            command=lambda *_: self.draw_viewport()
        ).pack(side="left")

        ttk.Button(toolbar, text="Фокус", command=self.focus_scene).pack(side="left", padx=6)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(toolbar, text="Рендер", command=self.render_clicked).pack(side="left")
        ttk.Button(toolbar, text="Сохранить PNG", command=self.save_last_render).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Экспорт CSV", command=self.export_scene_csv).pack(side="left")

        ttk.Label(toolbar, textvariable=self.status, anchor="e").pack(side="right", fill="x", expand=True)

        # Body
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left_scroll = ScrollableFrame(body, width=560, bg=BG)
        left_scroll.pack(side="left", fill="y")
        left = left_scroll.inner

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Viewport
        pane = ttk.LabelFrame(right, text="Окно просмотра", padding=8)
        pane.pack(fill="both", expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=pane)
        self.canvas.draw()
        w = self.canvas.get_tk_widget()
        w.configure(bg=BG, highlightthickness=0)
        w.pack(fill="both", expand=True)

        # Help
        help_box = ttk.LabelFrame(left, text="Справка", padding=10)
        help_box.pack(fill="x", pady=6)
        ttk.Label(
            help_box,
            text=(
                "• Масштаб: колесо мыши\n"
                "• 2D перемещение: ЛКМ перетаскивание\n"
                "• 2D поворот OBJ: Ctrl + ЛКМ перетаскивание\n"
                "• 3D перемещение: выберите объект → ЛКМ перетаскивание\n"
                "  Shift + ЛКМ — перемещение в глубину\n"
                "• Редактирование: выберите объект → измените поля → Применить\n"
            ),
            justify="left",
            wraplength=530
        ).pack(anchor="w")

        # Output folder
        io = ttk.LabelFrame(left, text="Вывод", padding=10)
        io.pack(fill="x", pady=6)
        self._path_row(io, "Папка:", self.out_dir, self.choose_out_dir)

        # Camera
        cam = ttk.LabelFrame(left, text="Камера", padding=10)
        cam.pack(fill="x", pady=6)
        self._spin(cam, "Азимут", self.cam_yaw, -180, 180, 5)
        self._spin(cam, "Подъём", self.cam_pitch, -89, 89, 2)
        self._spin(cam, "Дистанция", self.cam_dist, 2, 60, 1)
        ttk.Checkbutton(cam, text="Показывать оси", variable=self.show_axes, command=self.draw_viewport).pack(anchor="w", pady=4)
        ttk.Button(cam, text="Применить камеру", command=self.draw_viewport).pack(fill="x")

        # Scene list
        sl = ttk.LabelFrame(left, text="Сцена", padding=10)
        sl.pack(fill="both", expand=True, pady=6)

        self.listbox = tk.Listbox(
            sl, height=10,
            bg=FIELD, fg=FG,
            selectbackground=ACCENT,
            highlightthickness=1, relief="flat",
            exportselection=False,
        )
        self.listbox.pack(fill="both", expand=True, pady=(0, 6))
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._reload_selected_into_edit())

        rowb = ttk.Frame(sl)
        rowb.pack(fill="x")
        ttk.Button(rowb, text="Удалить", command=self.delete_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(rowb, text="Дублировать", command=self.duplicate_selected).pack(side="left", fill="x", expand=True, padx=6)

        # Add panel
        add = ttk.LabelFrame(left, text="Добавить", padding=10)
        add.pack(fill="x", pady=6)

        ttk.Label(add, text="Тип:").pack(anchor="w")
        ttk.OptionMenu(
            add, self.new_kind, self.new_kind.get(),
            "sphere", "box", "plane", "cylinder", "torus", "obj",
            command=lambda *_: self._refresh_add_form()
        ).pack(fill="x", pady=4)

        self.add_form = ttk.Frame(add)
        self.add_form.pack(fill="x")
        self._refresh_add_form()

        color_row = ttk.Frame(add)
        color_row.pack(fill="x", pady=6)
        ttk.Label(color_row, text="Цвет:", width=10).pack(side="left")
        self.add_color_btn = ttk.Button(color_row, text="Выбрать…", command=self.pick_add_color)
        self.add_color_btn.pack(side="left")
        self._sync_color_button(self.add_color_btn, self.add_color_hex.get())

        tex = ttk.Frame(add)
        tex.pack(fill="x", pady=4)
        ttk.Entry(tex, textvariable=self.add_tex_path).pack(side="left", fill="x", expand=True)
        ttk.Button(tex, text="Текстура…", command=self.choose_add_texture).pack(side="left", padx=6)

        ttk.Button(add, text="Добавить объект", command=self.add_object).pack(fill="x", pady=6)

        # Edit selected
        edit = ttk.LabelFrame(left, text="Редактирование", padding=10)
        edit.pack(fill="x", pady=6)

        ttk.Label(edit, textvariable=self.edit_kind).pack(anchor="w", pady=(0, 6))

        ttk.Label(edit, text="Имя:").pack(anchor="w")
        ttk.Entry(edit, textvariable=self.edit_name).pack(fill="x", pady=3)

        pos = ttk.LabelFrame(edit, text="Позиция", padding=8)
        pos.pack(fill="x", pady=6)
        self._spin(pos, "X", self.edit_x, -50, 50, 0.1)
        self._spin(pos, "Y", self.edit_y, -50, 50, 0.1)
        self._spin(pos, "Z", self.edit_z, -50, 50, 0.1)

        self.edit_params = ttk.Frame(edit)
        self.edit_params.pack(fill="x", pady=4)

        col = ttk.Frame(edit)
        col.pack(fill="x", pady=6)
        ttk.Label(col, text="Цвет:", width=10).pack(side="left")
        self.edit_color_btn = ttk.Button(col, text="Выбрать…", command=self.pick_edit_color)
        self.edit_color_btn.pack(side="left")
        self._sync_color_button(self.edit_color_btn, self.edit_color_hex.get())

        tex2 = ttk.Frame(edit)
        tex2.pack(fill="x", pady=4)
        ttk.Entry(tex2, textvariable=self.edit_tex_path).pack(side="left", fill="x", expand=True)
        ttk.Button(tex2, text="Текстура…", command=self.choose_edit_texture).pack(side="left", padx=6)

        ttk.Button(edit, text="Применить", command=self.apply_edit).pack(fill="x", pady=6)

        # Render settings
        ren = ttk.LabelFrame(left, text="Параметры рендера", padding=10)
        ren.pack(fill="x", pady=6)

        self._spin_int(ren, "Ширина", self.r_w, 200, 1920, 10)
        self._spin_int(ren, "Высота", self.r_h, 150, 1080, 10)
        self._spin_int(ren, "Шаги", self.r_steps, 16, 512, 1)
        self._spin(ren, "FOV", self.r_fov, 20, 95, 1)
        self._spin(ren, "Yaw", self.r_yaw, -180, 180, 5)
        self._spin(ren, "Pitch", self.r_pitch, -89, 89, 2)

    # ------------------ Toolbar actions ------------------
    def _on_new_scene(self):
        if messagebox.askyesno("Новая сцена", "Сбросить сцену к состоянию по умолчанию?"):
            self._seed_scene()
            self._refresh_object_list()
            self._reload_selected_into_edit()
            self.focus_scene()
            self.draw_viewport()
            self.status.set("Сцена создана заново.")

    # ------------------ UI helpers ------------------
    def _path_row(self, parent, label, var, choose_cmd):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=10).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать…", command=choose_cmd).pack(side="left", padx=6)

    def _spin(self, parent, label, var, vmin, vmax, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=10).pack(side="left")
        ttk.Spinbox(row, textvariable=var, from_=vmin, to=vmax, increment=step, width=10).pack(side="left")

    def _spin_int(self, parent, label, var, vmin, vmax, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=10).pack(side="left")
        ttk.Spinbox(row, textvariable=var, from_=vmin, to=vmax, increment=step, width=10).pack(side="left")

    def _sync_color_button(self, btn: ttk.Button, hex_color: str):
        btn.configure(text=f"Выбрать ({hex_color})")

    # ------------------ Files ------------------
    def choose_out_dir(self):
        d = filedialog.askdirectory(title="Выберите папку вывода")
        if d:
            self.out_dir.set(d)

    def choose_background(self):
        p = filedialog.askopenfilename(
            title="Выберите фоновое изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")]
        )
        if p:
            self.scene.background_path = p
            self.status.set("Фон установлен.")

    def clear_background(self):
        self.scene.background_path = None
        self.status.set("Фон очищен.")

    def choose_add_texture(self):
        p = filedialog.askopenfilename(
            title="Выберите текстуру (метаданные)",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")]
        )
        if p:
            self.add_tex_path.set(p)

    def choose_edit_texture(self):
        p = filedialog.askopenfilename(
            title="Выберите текстуру (метаданные)",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")]
        )
        if p:
            self.edit_tex_path.set(p)

    # ------------------ Scene list ------------------
    def _refresh_object_list(self):
        self.listbox.delete(0, tk.END)
        for i, o in enumerate(self.scene.objects):
            extra = ""
            if o.kind == "mesh_obj" and o.vertices is not None and o.faces is not None:
                extra = f" (v={o.vertices.shape[0]} f={o.faces.shape[0]})"
            self.listbox.insert(tk.END, f"{i}: {o.name} [{o.kind}]{extra}")

    def _get_selected_index(self) -> Optional[int]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        return int(sel[0])

    def delete_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        self.scene.remove_by_index(idx)
        self._refresh_object_list()
        self._reload_selected_into_edit()
        self.draw_viewport()

    def duplicate_selected(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        from copy import deepcopy
        o = deepcopy(self.scene.objects[idx])
        o.name = f"{o.name}_копия"
        if o.kind in ("sphere", "box", "cylinder", "torus"):
            cx, cy, cz = o.sdf_params.get("center", (0, 0, 0))
            o.sdf_params["center"] = (float(cx) + 0.3, float(cy), float(cz) + 0.3)
        elif o.kind == "mesh_obj" and o.vertices is not None:
            o.vertices = o.vertices + np.array([0.3, 0.0, 0.3])
        self.scene.add(o)
        self._refresh_object_list()
        self.draw_viewport()

    def export_scene_csv(self):
        ensure_dir(self.out_dir.get())
        df = pd.DataFrame(self.scene.summary_rows())
        path = os.path.join(self.out_dir.get(), "scene_objects.csv")
        df.to_csv(path, index=False)
        self.status.set("CSV экспортирован.")

    # ------------------ Camera ------------------
    def focus_scene(self):
        mn, mx = _scene_bounds(self.scene)
        radius = float(np.linalg.norm(mx - mn) * 0.5 + 1e-6)
        radius = max(radius, 1.0)
        self.cam_dist.set(radius * 2.6)
        self.draw_viewport()
        self.status.set("Фокус на сцену.")

    # ------------------ Add object ------------------
    def _refresh_add_form(self):
        for w in self.add_form.winfo_children():
            w.destroy()

        kind = self.new_kind.get()

        if kind in ("sphere", "box", "cylinder", "torus", "obj"):
            pos = ttk.LabelFrame(self.add_form, text="Позиция", padding=8)
            pos.pack(fill="x", pady=4)
            self._spin(pos, "X", self.add_x, -50, 50, 0.1)
            self._spin(pos, "Y", self.add_y, -50, 50, 0.1)
            self._spin(pos, "Z", self.add_z, -50, 50, 0.1)

        if kind == "sphere":
            p = ttk.LabelFrame(self.add_form, text="Сфера", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "Радиус", self.add_a, 0.05, 20, 0.05)

        elif kind == "box":
            p = ttk.LabelFrame(self.add_form, text="Куб (half-size)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "hx", self.add_b, 0.05, 20, 0.05)
            self._spin(p, "hy", self.add_c, 0.05, 20, 0.05)
            self._spin(p, "hz", self.add_d, 0.05, 20, 0.05)

        elif kind == "cylinder":
            p = ttk.LabelFrame(self.add_form, text="Цилиндр (ось Y)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "Радиус", self.add_a, 0.05, 20, 0.05)
            self._spin(p, "Половина высоты", self.add_b, 0.05, 20, 0.05)

        elif kind == "torus":
            p = ttk.LabelFrame(self.add_form, text="Тор (ось Y)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "R", self.add_a, 0.05, 40, 0.05)
            self._spin(p, "r", self.add_b, 0.02, 20, 0.02)

        elif kind == "plane":
            p = ttk.LabelFrame(self.add_form, text="Плоскость", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "nx", self.add_plane_nx, -1, 1, 0.05)
            self._spin(p, "ny", self.add_plane_ny, -1, 1, 0.05)
            self._spin(p, "nz", self.add_plane_nz, -1, 1, 0.05)
            self._spin(p, "h", self.add_plane_h, -50, 50, 0.05)

        elif kind == "obj":
            p = ttk.LabelFrame(self.add_form, text="OBJ", padding=8)
            p.pack(fill="x", pady=4)
            ttk.Button(p, text="Импорт OBJ…", command=self.import_obj_clicked).pack(fill="x")

    def pick_add_color(self):
        col = colorchooser.askcolor(title="Выберите цвет")
        if not col or not col[1]:
            return
        self.add_color_hex.set(col[1])
        self._sync_color_button(self.add_color_btn, col[1])

    def add_object(self):
        kind = self.new_kind.get()
        color = hex_to_rgb(self.add_color_hex.get())
        tex = self.add_tex_path.get().strip() or None

        if kind == "sphere":
            obj = Object3D(
                name=f"Сфера_{len(self.scene.objects)}",
                kind="sphere",
                material=Material(name="mat", base_color=color, texture_path=tex),
                sdf_params={"center": (float(self.add_x.get()), float(self.add_y.get()), float(self.add_z.get())),
                            "radius": float(self.add_a.get())}
            )
            self.scene.add(obj)

        elif kind == "box":
            obj = Object3D(
                name=f"Куб_{len(self.scene.objects)}",
                kind="box",
                material=Material(name="mat", base_color=color, texture_path=tex),
                sdf_params={"center": (float(self.add_x.get()), float(self.add_y.get()), float(self.add_z.get())),
                            "half_size": (float(self.add_b.get()), float(self.add_c.get()), float(self.add_d.get()))}
            )
            self.scene.add(obj)

        elif kind == "cylinder":
            obj = Object3D(
                name=f"Цилиндр_{len(self.scene.objects)}",
                kind="cylinder",
                material=Material(name="mat", base_color=color, texture_path=tex),
                sdf_params={"center": (float(self.add_x.get()), float(self.add_y.get()), float(self.add_z.get())),
                            "radius": float(self.add_a.get()), "half_height": float(self.add_b.get())}
            )
            self.scene.add(obj)

        elif kind == "torus":
            obj = Object3D(
                name=f"Тор_{len(self.scene.objects)}",
                kind="torus",
                material=Material(name="mat", base_color=color, texture_path=tex),
                sdf_params={"center": (float(self.add_x.get()), float(self.add_y.get()), float(self.add_z.get())),
                            "major_radius": float(self.add_a.get()), "minor_radius": float(self.add_b.get())}
            )
            self.scene.add(obj)

        elif kind == "plane":
            obj = Object3D(
                name=f"Плоскость_{len(self.scene.objects)}",
                kind="plane",
                material=Material(name="mat", base_color=color, texture_path=tex),
                sdf_params={"normal": (float(self.add_plane_nx.get()), float(self.add_plane_ny.get()), float(self.add_plane_nz.get())),
                            "h": float(self.add_plane_h.get())}
            )
            self.scene.add(obj)

        elif kind == "obj":
            messagebox.showinfo("OBJ", "Используйте «Импорт OBJ».")
            return

        self._refresh_object_list()
        self.draw_viewport()

    def import_obj_clicked(self):
        path = filedialog.askopenfilename(
            title="Выберите OBJ-файл",
            filetypes=[("OBJ", "*.obj"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            v, f = load_obj(path)
        except Exception as e:
            messagebox.showerror("Ошибка импорта OBJ", str(e))
            return

        v = v - v.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(v, axis=1)) + 1e-12
        v = v / scale * 2.0
        v = v + np.array([float(self.add_x.get()), float(self.add_y.get()), float(self.add_z.get())], dtype=np.float64)

        color = hex_to_rgb(self.add_color_hex.get())
        tex = self.add_tex_path.get().strip() or None

        obj = Object3D(
            name=os.path.basename(path),
            kind="mesh_obj",
            material=Material(name="mat_obj", base_color=color, texture_path=tex),
            vertices=v,
            faces=f,
        )
        self.scene.add(obj)
        self._refresh_object_list()
        self.draw_viewport()
        self.status.set("OBJ импортирован.")

    # ------------------ Edit selected ------------------
    def pick_edit_color(self):
        col = colorchooser.askcolor(title="Выберите цвет")
        if not col or not col[1]:
            return
        self.edit_color_hex.set(col[1])
        self._sync_color_button(self.edit_color_btn, col[1])

    def _reload_selected_into_edit(self):
        idx = self._get_selected_index()
        if idx is None:
            self.edit_kind.set("Нет выбора")
            self.edit_name.set("")
            self._rebuild_edit_params(None)
            return

        o = self.scene.objects[idx]
        self.edit_kind.set(f"{idx}: {o.name} [{o.kind}]")
        self.edit_name.set(o.name)
        self.edit_tex_path.set(o.material.texture_path or "")
        self.edit_color_hex.set(rgb_to_hex(o.material.base_color))
        self._sync_color_button(self.edit_color_btn, self.edit_color_hex.get())

        if o.kind in ("sphere", "box", "cylinder", "torus"):
            cx, cy, cz = o.sdf_params.get("center", (0.0, 0.0, 0.0))
            self.edit_x.set(float(cx)); self.edit_y.set(float(cy)); self.edit_z.set(float(cz))
        elif o.kind == "mesh_obj" and o.vertices is not None:
            c = o.vertices.mean(axis=0)
            self.edit_x.set(float(c[0])); self.edit_y.set(float(c[1])); self.edit_z.set(float(c[2]))
        else:
            self.edit_x.set(0.0); self.edit_y.set(0.0); self.edit_z.set(0.0)

        self._rebuild_edit_params(o.kind)

        if o.kind == "sphere":
            self.edit_a.set(float(o.sdf_params.get("radius", 1.0)))
        elif o.kind == "box":
            hx, hy, hz = o.sdf_params.get("half_size", (0.8, 0.8, 0.8))
            self.edit_b.set(float(hx)); self.edit_c.set(float(hy)); self.edit_d.set(float(hz))
        elif o.kind == "cylinder":
            self.edit_a.set(float(o.sdf_params.get("radius", 0.6)))
            self.edit_b.set(float(o.sdf_params.get("half_height", 0.9)))
        elif o.kind == "torus":
            self.edit_a.set(float(o.sdf_params.get("major_radius", 1.0)))
            self.edit_b.set(float(o.sdf_params.get("minor_radius", 0.28)))
        elif o.kind == "plane":
            nx, ny, nz = o.sdf_params.get("normal", (0.0, 1.0, 0.0))
            self.edit_plane_nx.set(float(nx)); self.edit_plane_ny.set(float(ny)); self.edit_plane_nz.set(float(nz))
            self.edit_plane_h.set(float(o.sdf_params.get("h", 1.2)))

    def _rebuild_edit_params(self, kind: Optional[str]):
        for w in self.edit_params.winfo_children():
            w.destroy()

        if kind == "sphere":
            p = ttk.LabelFrame(self.edit_params, text="Сфера", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "Радиус", self.edit_a, 0.05, 20, 0.05)
        elif kind == "box":
            p = ttk.LabelFrame(self.edit_params, text="Куб (half-size)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "hx", self.edit_b, 0.05, 20, 0.05)
            self._spin(p, "hy", self.edit_c, 0.05, 20, 0.05)
            self._spin(p, "hz", self.edit_d, 0.05, 20, 0.05)
        elif kind == "cylinder":
            p = ttk.LabelFrame(self.edit_params, text="Цилиндр (ось Y)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "Радиус", self.edit_a, 0.05, 20, 0.05)
            self._spin(p, "Половина высоты", self.edit_b, 0.05, 20, 0.05)
        elif kind == "torus":
            p = ttk.LabelFrame(self.edit_params, text="Тор (ось Y)", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "R", self.edit_a, 0.05, 40, 0.05)
            self._spin(p, "r", self.edit_b, 0.02, 20, 0.02)
        elif kind == "plane":
            p = ttk.LabelFrame(self.edit_params, text="Плоскость", padding=8)
            p.pack(fill="x", pady=4)
            self._spin(p, "nx", self.edit_plane_nx, -1, 1, 0.05)
            self._spin(p, "ny", self.edit_plane_ny, -1, 1, 0.05)
            self._spin(p, "nz", self.edit_plane_nz, -1, 1, 0.05)
            self._spin(p, "h", self.edit_plane_h, -50, 50, 0.05)
        else:
            ttk.Label(self.edit_params, text="").pack()

    def apply_edit(self):
        idx = self._get_selected_index()
        if idx is None:
            return
        o = self.scene.objects[idx]

        name = self.edit_name.get().strip()
        if name:
            o.name = name

        o.material.base_color = hex_to_rgb(self.edit_color_hex.get())
        tex = self.edit_tex_path.get().strip()
        o.material.texture_path = tex if tex else None

        px, py, pz = float(self.edit_x.get()), float(self.edit_y.get()), float(self.edit_z.get())

        if o.kind in ("sphere", "box", "cylinder", "torus"):
            o.sdf_params["center"] = (clamp_pos(px), clamp_pos(py), clamp_pos(pz))
        elif o.kind == "mesh_obj" and o.vertices is not None:
            c = o.vertices.mean(axis=0)
            delta = np.array([px, py, pz], dtype=np.float64) - c
            o.vertices = o.vertices + delta

        if o.kind == "sphere":
            o.sdf_params["radius"] = max(0.01, float(self.edit_a.get()))
        elif o.kind == "box":
            o.sdf_params["half_size"] = (max(0.01, float(self.edit_b.get())),
                                         max(0.01, float(self.edit_c.get())),
                                         max(0.01, float(self.edit_d.get())))
        elif o.kind == "cylinder":
            o.sdf_params["radius"] = max(0.01, float(self.edit_a.get()))
            o.sdf_params["half_height"] = max(0.01, float(self.edit_b.get()))
        elif o.kind == "torus":
            o.sdf_params["major_radius"] = max(0.01, float(self.edit_a.get()))
            o.sdf_params["minor_radius"] = max(0.01, float(self.edit_b.get()))
        elif o.kind == "plane":
            o.sdf_params["normal"] = (float(self.edit_plane_nx.get()),
                                      float(self.edit_plane_ny.get()),
                                      float(self.edit_plane_nz.get()))
            o.sdf_params["h"] = float(self.edit_plane_h.get())

        self._refresh_object_list()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.draw_viewport()
        self.status.set("Применено.")

    # ------------------ Render ------------------
    def render_clicked(self):
        ensure_dir(self.out_dir.get())
        cam_pos, cam_target, max_dist = _fit_camera_for_scene(self.scene, yaw_deg=float(self.r_yaw.get()), pitch_deg=float(self.r_pitch.get()))

        cfg = RenderConfig(
            width=int(self.r_w.get()),
            height=int(self.r_h.get()),
            fov_deg=float(self.r_fov.get()),
            max_steps=int(self.r_steps.get()),
            max_dist=float(max_dist),
            camera_pos=cam_pos,
            camera_target=cam_target,
        )

        def worker():
            try:
                self.status.set("Рендер…")
                img, metrics = render(self.scene, cfg, out_dir=self.out_dir.get(), save_prefix="render")
                self._last_render_img = img
                self.after(0, lambda: self._draw_image(img))
                self.after(0, lambda: self.status.set(f"Готово. {metrics['render_time_sec']:.3f} сек, hit={metrics['hit_ratio']:.3f}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка рендера", str(e)))
                self.after(0, lambda: self.status.set("Ошибка рендера."))

        threading.Thread(target=worker, daemon=True).start()

    def save_last_render(self):
        if self._last_render_img is None:
            messagebox.showinfo("Нет изображения", "Сначала выполните рендер.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not path:
            return
        plt.imsave(path, np.clip(self._last_render_img, 0.0, 1.0))
        self.status.set("PNG сохранён.")

    # ------------------ Viewport draw ------------------
    def _reset_axes(self):
        self.fig.clear()
        if self.viewport_mode.get() == "3D":
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)

    def draw_viewport(self):
        self._reset_axes()
        mode = self.viewport_mode.get()
        if mode == "3D":
            self._draw_3d()
        else:
            self._draw_2d(mode)
        self.canvas.draw()

    def _draw_3d(self):
        ax = self.ax
        ax.set_title("3D")

        ax.view_init(elev=float(self.cam_pitch.get()), azim=float(self.cam_yaw.get()))
        try:
            ax.dist = float(self.cam_dist.get())
        except Exception:
            pass

        # soften 3D panes a bit
        try:
            ax.xaxis.pane.set_facecolor((0.18, 0.18, 0.18, 1.0))
            ax.yaxis.pane.set_facecolor((0.18, 0.18, 0.18, 1.0))
            ax.zaxis.pane.set_facecolor((0.18, 0.18, 0.18, 1.0))
        except Exception:
            pass

        for o in self.scene.objects:
            if o.kind == "sphere":
                self._plot_sphere(ax, o)
            elif o.kind == "box":
                self._plot_box(ax, o)
            elif o.kind == "plane":
                self._plot_plane(ax, o)
            elif o.kind == "cylinder":
                self._plot_cylinder(ax, o)
            elif o.kind == "torus":
                self._plot_torus(ax, o)
            elif o.kind == "mesh_obj":
                self._plot_mesh_obj(ax, o)

        self._set_3d_equal(ax)

        if not self.show_axes.get():
            ax.set_axis_off()
        else:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    def _draw_2d(self, mode: str):
        ax = self.ax
        ax.set_title(mode)

        for o in self.scene.objects:
            c = o.material.base_color

            if o.kind == "sphere":
                center = np.array(o.sdf_params["center"], dtype=np.float64)
                r = float(o.sdf_params["radius"])
                p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                ax.add_patch(Circle((p2[0], p2[1]), radius=r, facecolor=c, edgecolor=(0, 0, 0, 0.35), alpha=0.9))

            elif o.kind == "box":
                center = np.array(o.sdf_params["center"], dtype=np.float64)
                hs = np.array(o.sdf_params["half_size"], dtype=np.float64)
                if mode == "XY":
                    w, h = 2 * hs[0], 2 * hs[1]
                    pos = (center[0] - hs[0], center[1] - hs[1])
                elif mode == "XZ":
                    w, h = 2 * hs[0], 2 * hs[2]
                    pos = (center[0] - hs[0], center[2] - hs[2])
                else:
                    w, h = 2 * hs[1], 2 * hs[2]
                    pos = (center[1] - hs[1], center[2] - hs[2])
                ax.add_patch(Rectangle(pos, w, h, facecolor=c, edgecolor=(0, 0, 0, 0.35), alpha=0.9))

            elif o.kind == "plane":
                n = np.array(o.sdf_params["normal"], dtype=np.float64)
                h = float(o.sdf_params["h"])
                n = n / (np.linalg.norm(n) + 1e-12)
                p0 = -h * n
                a = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, n)) > 0.9:
                    a = np.array([0.0, 0.0, 1.0])
                t1 = np.cross(n, a); t1 = t1 / (np.linalg.norm(t1) + 1e-12)
                t2 = np.cross(n, t1)
                s = 4.0
                pts3 = np.array([p0 + s*t1 + s*t2,
                                 p0 - s*t1 + s*t2,
                                 p0 - s*t1 - s*t2,
                                 p0 + s*t1 - s*t2], dtype=np.float64)
                pts2 = project_points_ortho(pts3, mode=mode)
                ax.add_patch(Polygon(pts2, closed=True, facecolor=c, edgecolor=(0, 0, 0, 0.2), alpha=0.2))

            elif o.kind == "cylinder":
                center = np.array(o.sdf_params["center"], dtype=np.float64)
                r = float(o.sdf_params["radius"])
                p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                ax.add_patch(Circle((p2[0], p2[1]), radius=r, facecolor=c, edgecolor=(0, 0, 0, 0.35), alpha=0.8))

            elif o.kind == "torus":
                center = np.array(o.sdf_params["center"], dtype=np.float64)
                R = float(o.sdf_params["major_radius"])
                p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                ax.add_patch(Circle((p2[0], p2[1]), radius=R, facecolor=c, edgecolor=(0, 0, 0, 0.35), alpha=0.25))

            elif o.kind == "mesh_obj" and o.vertices is not None:
                pts2 = project_points_ortho(o.vertices, mode=mode)
                ax.scatter(pts2[:, 0], pts2[:, 1], s=1, c=[c], alpha=0.5)

        ax.grid(True, alpha=0.20)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()

    # ------------------ Mpl interactions ------------------
    def _on_mpl_scroll(self, event):
        mode = self.viewport_mode.get()
        step = getattr(event, "step", None)
        if step is None:
            step = 1 if getattr(event, "button", "") == "up" else -1

        if mode == "3D":
            d = float(self.cam_dist.get())
            d *= 0.90 if step > 0 else 1.10
            self.cam_dist.set(max(2.0, min(60.0, d)))
            self.draw_viewport()
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        ax = self.ax
        scale = 0.9 if step > 0 else 1.1
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        ax.set_xlim(cx + (x0 - cx) * scale, cx + (x1 - cx) * scale)
        ax.set_ylim(cy + (y0 - cy) * scale, cy + (y1 - cy) * scale)
        self.canvas.draw()

    def _on_mpl_press(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return

        mode = self.viewport_mode.get()

        if mode == "3D":
            idx = self._get_selected_index()
            if idx is None:
                return
            key = (event.key or "").lower()
            action = "move_depth" if ("shift" in key) else "move_xy"
            self._drag3d_active = True
            self._drag3d_idx = idx
            self._drag3d_last_px = (float(event.x), float(event.y))
            self._drag3d_action = action
            return

        if event.xdata is None or event.ydata is None:
            return

        action = "move"
        key = (event.key or "").lower()
        if "control" in key or "ctrl" in key:
            action = "rotate"

        idx = self._pick_object_2d(mode, float(event.xdata), float(event.ydata))
        if idx is None:
            return

        self._drag_active = True
        self._drag_idx = idx
        self._drag_mode = mode
        self._drag_last_xy = (float(event.xdata), float(event.ydata))
        self._drag_action = action

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self._reload_selected_into_edit()

    def _on_mpl_motion(self, event):
        mode = self.viewport_mode.get()

        if mode == "3D" and self._drag3d_active and self._drag3d_idx is not None:
            if event.inaxes != self.ax:
                return
            if self._drag3d_last_px is None:
                self._drag3d_last_px = (float(event.x), float(event.y))
                return

            x0, y0 = self._drag3d_last_px
            x1, y1 = float(event.x), float(event.y)
            dxp, dyp = (x1 - x0), (y1 - y0)
            self._drag3d_last_px = (x1, y1)

            o = self.scene.objects[self._drag3d_idx]
            d = float(self.cam_dist.get())
            k_xy = 0.004 * d
            k_depth = 0.010 * d

            if self._drag3d_action == "move_xy":
                self._move_object(o, dx=dxp * k_xy, dy=-dyp * k_xy, dz=0.0)
            else:
                self._move_object(o, dx=0.0, dy=0.0, dz=-dyp * k_depth)

            self._reload_selected_into_edit()
            self.draw_viewport()
            return

        if not self._drag_active or self._drag_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._drag_last_xy is None:
            self._drag_last_xy = (float(event.xdata), float(event.ydata))
            return

        x0, y0 = self._drag_last_xy
        x1, y1 = float(event.xdata), float(event.ydata)
        dx, dy = x1 - x0, y1 - y0
        self._drag_last_xy = (x1, y1)

        o = self.scene.objects[self._drag_idx]
        a0, a1 = _mode_axes(self._drag_mode)

        if self._drag_action == "move":
            if o.kind in ("sphere", "box", "cylinder", "torus"):
                cx, cy, cz = o.sdf_params.get("center", (0.0, 0.0, 0.0))
                c = np.array([cx, cy, cz], dtype=np.float64)
                c[a0] += dx
                c[a1] += dy
                o.sdf_params["center"] = (float(c[0]), float(c[1]), float(c[2]))
            elif o.kind == "mesh_obj" and o.vertices is not None:
                delta = np.zeros(3, dtype=np.float64)
                delta[a0] = dx
                delta[a1] = dy
                o.vertices = o.vertices + delta
        else:
            if o.kind == "mesh_obj" and o.vertices is not None:
                if self._drag_mode == "XY":
                    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                elif self._drag_mode == "XZ":
                    axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                else:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

                angle = dx * 0.6
                R = _rotation_matrix_axis_angle(axis, angle)
                c = o.vertices.mean(axis=0)
                V = o.vertices - c[None, :]
                o.vertices = (V @ R.T) + c[None, :]

        self._reload_selected_into_edit()
        self.draw_viewport()

    def _on_mpl_release(self, event):
        if event.button != 1:
            return
        self._drag_active = False
        self._drag_idx = None
        self._drag_last_xy = None
        self._drag3d_active = False
        self._drag3d_idx = None
        self._drag3d_last_px = None

    def _pick_object_2d(self, mode: str, x: float, y: float) -> Optional[int]:
        best_idx = None
        best_d = 1e9
        for i, o in enumerate(self.scene.objects):
            if o.kind == "plane":
                continue
            if o.kind in ("sphere", "cylinder"):
                center = np.array(o.sdf_params.get("center", (0.0, 0.0, 0.0)), dtype=np.float64)
                r = float(o.sdf_params.get("radius", 0.8))
                p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                d = float(np.hypot(x - p2[0], y - p2[1]))
                if d <= max(0.15, r * 1.1) and d < best_d:
                    best_d, best_idx = d, i
            elif o.kind == "torus":
                center = np.array(o.sdf_params.get("center", (0.0, 0.0, 0.0)), dtype=np.float64)
                R = float(o.sdf_params.get("major_radius", 1.0))
                p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                d = float(np.hypot(x - p2[0], y - p2[1]))
                if d <= max(0.2, R * 1.05) and d < best_d:
                    best_d, best_idx = d, i
            elif o.kind == "box":
                center = np.array(o.sdf_params.get("center", (0.0, 0.0, 0.0)), dtype=np.float64)
                hs = np.array(o.sdf_params.get("half_size", (0.6, 0.6, 0.6)), dtype=np.float64)
                if mode == "XY":
                    xmin, xmax = center[0] - hs[0], center[0] + hs[0]
                    ymin, ymax = center[1] - hs[1], center[1] + hs[1]
                elif mode == "XZ":
                    xmin, xmax = center[0] - hs[0], center[0] + hs[0]
                    ymin, ymax = center[2] - hs[2], center[2] + hs[2]
                else:
                    xmin, xmax = center[1] - hs[1], center[1] + hs[1]
                    ymin, ymax = center[2] - hs[2], center[2] + hs[2]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    p2 = project_points_ortho(center.reshape(1, 3), mode=mode)[0]
                    d = float(np.hypot(x - p2[0], y - p2[1]))
                    if d < best_d:
                        best_d, best_idx = d, i
            elif o.kind == "mesh_obj" and o.vertices is not None:
                c = o.vertices.mean(axis=0)
                p2 = project_points_ortho(c.reshape(1, 3), mode=mode)[0]
                d = float(np.hypot(x - p2[0], y - p2[1]))
                if d <= 0.6 and d < best_d:
                    best_d, best_idx = d, i
        return best_idx

    def _move_object(self, o: Object3D, dx: float, dy: float, dz: float):
        if o.kind in ("sphere", "box", "cylinder", "torus"):
            cx, cy, cz = o.sdf_params.get("center", (0, 0, 0))
            o.sdf_params["center"] = (float(cx + dx), float(cy + dy), float(cz + dz))
        elif o.kind == "mesh_obj" and o.vertices is not None:
            o.vertices = o.vertices + np.array([dx, dy, dz], dtype=np.float64)

    # ------------------ 3D plot helpers ------------------
    def _plot_sphere(self, ax, o: Object3D):
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        r = float(o.sdf_params["radius"])
        col = o.material.base_color
        u = np.linspace(0, 2*np.pi, 44)
        v = np.linspace(0, np.pi, 22)
        x = c[0] + r * np.outer(np.cos(u), np.sin(v))
        y = c[1] + r * np.outer(np.sin(u), np.sin(v))
        z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=col, linewidth=0, antialiased=True, shade=True, alpha=0.98)

    def _plot_box(self, ax, o: Object3D):
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        hs = np.array(o.sdf_params["half_size"], dtype=np.float64)
        col = o.material.base_color

        corners = np.array([
            c + [-hs[0], -hs[1], -hs[2]],
            c + [ hs[0], -hs[1], -hs[2]],
            c + [ hs[0],  hs[1], -hs[2]],
            c + [-hs[0],  hs[1], -hs[2]],
            c + [-hs[0], -hs[1],  hs[2]],
            c + [ hs[0], -hs[1],  hs[2]],
            c + [ hs[0],  hs[1],  hs[2]],
            c + [-hs[0],  hs[1],  hs[2]],
        ], dtype=np.float64)

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
        polys = [[corners[i] for i in f] for f in faces]
        pc = Poly3DCollection(polys, facecolors=[col], edgecolors=(0, 0, 0, 0.25), linewidths=0.6, alpha=0.98)
        ax.add_collection3d(pc)

    def _plot_plane(self, ax, o: Object3D):
        n = np.array(o.sdf_params["normal"], dtype=np.float64)
        h = float(o.sdf_params["h"])
        n = n / (np.linalg.norm(n) + 1e-12)
        p0 = -h * n
        col = o.material.base_color

        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 0.0, 1.0])
        t1 = np.cross(n, a); t1 = t1 / (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(n, t1)

        s = 6.0
        pts = np.array([
            p0 + s*t1 + s*t2,
            p0 - s*t1 + s*t2,
            p0 - s*t1 - s*t2,
            p0 + s*t1 - s*t2
        ], dtype=np.float64)

        poly = Poly3DCollection([pts], facecolors=[col], edgecolors=(0, 0, 0, 0.12), linewidths=0.4, alpha=0.20)
        ax.add_collection3d(poly)

    def _plot_cylinder(self, ax, o: Object3D):
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        r = float(o.sdf_params["radius"])
        hh = float(o.sdf_params["half_height"])
        col = o.material.base_color

        theta = np.linspace(0, 2*np.pi, 44)
        y = np.linspace(c[1] - hh, c[1] + hh, 18)
        T, Y = np.meshgrid(theta, y)
        X = c[0] + r * np.cos(T)
        Z = c[2] + r * np.sin(T)
        ax.plot_surface(X, Y, Z, color=col, linewidth=0, antialiased=True, shade=True, alpha=0.98)

    def _plot_torus(self, ax, o: Object3D):
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        R = float(o.sdf_params["major_radius"])
        r = float(o.sdf_params["minor_radius"])
        col = o.material.base_color

        u = np.linspace(0, 2*np.pi, 56)
        v = np.linspace(0, 2*np.pi, 28)
        U, V = np.meshgrid(u, v)
        X = c[0] + (R + r*np.cos(V)) * np.cos(U)
        Y = c[1] + r*np.sin(V)
        Z = c[2] + (R + r*np.cos(V)) * np.sin(U)
        ax.plot_surface(X, Y, Z, color=col, linewidth=0, antialiased=True, shade=True, alpha=0.98)

    def _plot_mesh_obj(self, ax, o: Object3D):
        if o.vertices is None or o.faces is None:
            return
        V = o.vertices
        F = o.faces
        col = o.material.base_color

        tris = V[F]
        max_tris = 9000
        if tris.shape[0] > max_tris:
            step = tris.shape[0] // max_tris
            tris = tris[::max(1, step)]

        pc = Poly3DCollection(tris, facecolors=[col], edgecolors=(0, 0, 0, 0.18), linewidths=0.15, alpha=0.92)
        ax.add_collection3d(pc)

    def _set_3d_equal(self, ax):
        mn, mx = _scene_bounds(self.scene)
        c = (mn + mx) / 2.0
        r = float(np.max(mx - mn) / 2.0 + 1e-6)
        r = max(r, 1.0)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)

    # ------------------ Render preview ------------------
    def _draw_image(self, img: np.ndarray):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.imshow(img)
        ax.set_title("Рендер")
        ax.axis("off")
        self.ax = ax
        self.canvas.draw()


def main():
    ensure_dir("outputs")
    app = MiniBlenderLike()
    app.mainloop()


if __name__ == "__main__":
    main()