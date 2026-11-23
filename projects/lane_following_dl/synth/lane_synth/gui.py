"""Desktop GUI (PySide6): course / trajectory / camera settings, preview, dataset generation.

  python3 scripts/gui.py [project.yaml]
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# opencv-python (non-headless) points Qt to its own plugins, which breaks PySide6
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import yaml
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSlider, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .camera import Camera
from .dataset import default_project, generate
from .generator import DEFAULT_CONFIG, Generator, load_config
from .render import CameraRenderer, WorldRaster, base_pose_matrix
from .labels import compute_labels
from .driving import follow_exact
from .track import course_from_dict

COURSE_DIR = Path(DEFAULT_CONFIG).parent / "courses"
SEG_TYPES = ["straight", "arc", "clothoid"]
SEG_COLS = ["種類", "長さ [m]", "半径 [m]", "角度 [deg]", "κ始め [1/m]", "κ終わり [1/m]"]
SEG_USED = {"straight": [1], "arc": [2, 3], "clothoid": [1, 4, 5]}
SEG_KEYS = {1: "length", 2: "radius", 3: "angle_deg", 4: "k0", 5: "k1"}


def dspin(lo, hi, val, step=0.01, dec=3, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(" " + suffix)
    w.setKeyboardTracking(False)
    return w


def parse_list(text):
    return [float(x) for x in text.replace("、", ",").split(",") if x.strip()]


def fmt_list(vals):
    return ", ".join(f"{v:g}" for v in vals)


def to_qimage(gray: np.ndarray) -> QImage:
    g = np.ascontiguousarray(gray)
    return QImage(g.data, g.shape[1], g.shape[0], g.strides[0], QImage.Format_Grayscale8).copy()


def to_qimage_rgb(rgb: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgb)
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()


class GenWorker(QObject):
    progress = Signal(int, int, int)
    finished = Signal(int, str)

    def __init__(self, project):
        super().__init__()
        self.project = project
        self.stop = False

    def run(self):
        try:
            n = generate(self.project, lambda d, t, f: self.progress.emit(d, t, f), lambda: self.stop)
            self.finished.emit(n, "")
        except Exception as e:  # report to the GUI
            self.finished.emit(0, repr(e))


class MainWindow(QMainWindow):
    def __init__(self, project_path=None):
        super().__init__()
        self.setWindowTitle("A1 白線データセット生成")
        self.resize(1500, 950)
        self.project = default_project(load_config())
        self.project["course"] = yaml.safe_load(open(COURSE_DIR / "oval.yaml"))
        self.project["course"]["name"] = "oval"
        self._renderer_key = None
        self._frames = []

        self.tabs = QTabWidget()
        self.tabs.addTab(self._course_tab(), "コース")
        self.tabs.addTab(self._traj_tab(), "走行経路")
        self.tabs.addTab(self._camera_tab(), "カメラ")
        self.tabs.addTab(self._bev_tab(), "鳥瞰・ラベル")
        self.tabs.addTab(self._output_tab(), "データ生成")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.tabs)
        hb = QHBoxLayout()
        for text, fn in [("プロジェクトを開く", self.open_project), ("プロジェクトを保存", self.save_project)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        lv.addLayout(hb)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._preview_panel())
        splitter.setSizes([620, 880])
        self.setCentralWidget(splitter)

        if project_path:
            self.load_project_file(project_path)
        else:
            self.project_to_ui()
        self.update_preview()

    # ------------------------------------------------------------------ tabs
    def _course_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.course_name = QLineEdit()
        self.closed_chk = QCheckBox("周回コース（終点を始点につなぐ）")
        self.lane_w = dspin(0.1, 2.0, 0.40, 0.01, 3, "m")
        self.line_w = dspin(0.005, 0.1, 0.02, 0.001, 3, "m")
        self.center_line = QComboBox()
        self.center_line.addItems(["none（白線 2 本だけ）", "dashed（中央破線あり）"])
        f.addRow("コース名", self.course_name)
        f.addRow("", self.closed_chk)
        f.addRow("車線幅（白線の間隔）", self.lane_w)
        f.addRow("白線の幅", self.line_w)
        f.addRow("中央線", self.center_line)
        v.addLayout(f)

        v.addWidget(QLabel("道路中心線のセグメント（上から順につなぐ。角度・曲率は左カーブが正）"))
        self.seg_table = QTableWidget(0, len(SEG_COLS))
        self.seg_table.setHorizontalHeaderLabels(SEG_COLS)
        self.seg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.seg_table)
        hb = QHBoxLayout()
        for text, fn in [("追加", self.seg_add), ("削除", self.seg_del), ("↑", lambda: self.seg_move(-1)),
                         ("↓", lambda: self.seg_move(1))]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        v.addLayout(hb)
        hb = QHBoxLayout()
        for text, fn in [("コースYAMLを読込", self.load_course), ("コースYAMLを保存", self.save_course),
                         ("プレビュー更新", self.update_preview)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        v.addLayout(hb)
        self.course_status = QLabel("")
        self.course_status.setWordWrap(True)
        v.addWidget(self.course_status)
        return w

    def _traj_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("プレビューの走行（設定した軌跡どおりに走る）")
        f = QFormLayout(g)
        self.direction = QComboBox()
        self.direction.addItems(["forward（セグメントの順）", "reverse（逆回り）"])
        self.speed = dspin(0.05, 5.0, 0.5, 0.05, 2, "m/s")
        self.fps = dspin(1, 120, 15, 1, 1, "fps")
        self.offset_y = dspin(-0.5, 0.5, 0.0, 0.01, 3, "m")
        self.heading_off = dspin(-45, 45, 0.0, 0.5, 2, "deg")
        self.start_s = dspin(0, 100, 0.0, 0.1, 2, "m")
        self.distance = dspin(0, 1000, 0.0, 0.5, 2, "m")
        self.distance.setSpecialValueText("1 周（開いたコースは終点まで）")
        f.addRow("走る向き", self.direction)
        f.addRow("速度", self.speed)
        f.addRow("フレームレート", self.fps)
        f.addRow("横ずれ（左が正）", self.offset_y)
        f.addRow("向きのずれ（左向きが正）", self.heading_off)
        f.addRow("スタート位置（経路長）", self.start_s)
        f.addRow("走る距離", self.distance)
        v.addWidget(g)
        g2 = QGroupBox("データ生成時の走行バリエーション（すべての組合せで 1 回ずつ走る）")
        f2 = QFormLayout(g2)
        self.dirs_fwd = QCheckBox("forward")
        self.dirs_rev = QCheckBox("reverse")
        hb = QHBoxLayout()
        hb.addWidget(self.dirs_fwd)
        hb.addWidget(self.dirs_rev)
        hw = QWidget()
        hw.setLayout(hb)
        self.speeds_txt = QLineEdit()
        self.offsets_txt = QLineEdit()
        self.headings_txt = QLineEdit()
        f2.addRow("走る向き", hw)
        f2.addRow("速度 [m/s]（カンマ区切り）", self.speeds_txt)
        f2.addRow("横ずれ [m]", self.offsets_txt)
        f2.addRow("向きのずれ [deg]", self.headings_txt)
        self.runs_label = QLabel("")
        f2.addRow("", self.runs_label)
        for wd in (self.speeds_txt, self.offsets_txt, self.headings_txt):
            wd.editingFinished.connect(self.update_run_count)
        for wd in (self.dirs_fwd, self.dirs_rev):
            wd.toggled.connect(self.update_run_count)
        v.addWidget(g2)
        b = QPushButton("プレビュー更新")
        b.clicked.connect(self.update_preview)
        v.addWidget(b)
        v.addStretch()
        return w

    def _camera_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("内部パラメータ（画素単位）")
        f = QFormLayout(g)
        self.cam_w = QSpinBox()
        self.cam_w.setRange(16, 8192)
        self.cam_h = QSpinBox()
        self.cam_h.setRange(16, 8192)
        self.fx = dspin(1, 20000, 460, 1, 2)
        self.fy = dspin(1, 20000, 460, 1, 2)
        self.cx = dspin(0, 10000, 320, 0.5, 2)
        self.cy = dspin(0, 10000, 240, 0.5, 2)
        f.addRow("幅 × 高さ [px]", self._pair(self.cam_w, self.cam_h))
        f.addRow("fx, fy", self._pair(self.fx, self.fy))
        f.addRow("cx, cy", self._pair(self.cx, self.cy))
        self.dist = [dspin(-10, 10, 0.0, 0.001, 6) for _ in range(5)]
        f.addRow("歪み k1, k2", self._pair(self.dist[0], self.dist[1]))
        f.addRow("歪み p1, p2", self._pair(self.dist[2], self.dist[3]))
        f.addRow("歪み k3", self.dist[4])
        self.fov_label = QLabel("")
        f.addRow("画角", self.fov_label)
        v.addWidget(g)
        g2 = QGroupBox("外部パラメータ（後輪車軸中心の床面から見たカメラの位置・姿勢）")
        f2 = QFormLayout(g2)
        self.cam_xyz = [dspin(-2, 2, 0.0, 0.005, 4, "m") for _ in range(3)]
        self.cam_rpy = [dspin(-180, 180, 0.0, 0.5, 3, "deg") for _ in range(3)]
        f2.addRow("x（前方）", self.cam_xyz[0])
        f2.addRow("y（左）", self.cam_xyz[1])
        f2.addRow("z（高さ）", self.cam_xyz[2])
        f2.addRow("roll", self.cam_rpy[0])
        f2.addRow("pitch（下向きが正）", self.cam_rpy[1])
        f2.addRow("yaw（左向きが正）", self.cam_rpy[2])
        v.addWidget(g2)
        hb = QHBoxLayout()
        b = QPushButton("キャリブレーション結果を読込（ROS camera_info / OpenCV YAML）")
        b.clicked.connect(self.load_calibration)
        hb.addWidget(b)
        b2 = QPushButton("プレビュー更新")
        b2.clicked.connect(self.update_preview)
        hb.addWidget(b2)
        v.addLayout(hb)
        v.addStretch()
        for wd in [self.fx, self.fy, self.cam_w, self.cam_h]:
            wd.valueChanged.connect(self.update_fov)
        return w

    def _bev_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("鳥瞰画像の範囲（車両座標：x 前方・y 左、原点は後輪車軸中心）")
        f = QFormLayout(g)
        self.bev_x = [dspin(0, 10, 0.3, 0.05, 3, "m"), dspin(0, 10, 1.0, 0.05, 3, "m")]
        self.bev_y = [dspin(-5, 5, -0.45, 0.05, 3, "m"), dspin(-5, 5, 0.45, 0.05, 3, "m")]
        self.bev_res = dspin(0.001, 0.1, 0.005, 0.001, 4, "m/px")
        f.addRow("x 最小 / 最大", self._pair(*self.bev_x))
        f.addRow("y 最小 / 最大", self._pair(*self.bev_y))
        f.addRow("解像度", self.bev_res)
        self.bev_size = QLabel("")
        f.addRow("画像サイズ", self.bev_size)
        v.addWidget(g)
        g2 = QGroupBox("正解ラベル")
        f2 = QFormLayout(g2)
        self.preview_s_txt = QLineEdit()
        self.path_s_txt = QLineEdit()
        f2.addRow("曲率 κ を記録する前方距離 s [m]", self.preview_s_txt)
        f2.addRow("中心線の点を記録する前方距離 s [m]", self.path_s_txt)
        f2.addRow(QLabel("s は車両の真横（目標経路上の最近傍点）から経路に沿って測る。κ は左カーブが正。"))
        v.addWidget(g2)
        b = QPushButton("プレビュー更新")
        b.clicked.connect(self.update_preview)
        v.addWidget(b)
        v.addStretch()
        for wd in self.bev_x + self.bev_y + [self.bev_res]:
            wd.valueChanged.connect(self.update_bev_size)
        return w

    def _output_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("生成方法")
        gv = QVBoxLayout(g)
        self.mode_course = QRadioButton("設定したコース × 走行バリエーション")
        self.mode_random = QRadioButton("ランダムなコース（ランダムな速度・位置）")
        gv.addWidget(self.mode_course)
        gv.addWidget(self.mode_random)
        f = QFormLayout()
        self.random_n = QSpinBox()
        self.random_n.setRange(1, 10_000_000)
        self.random_n.setSingleStep(1000)
        f.addRow("ランダム時の枚数", self.random_n)
        gv.addLayout(f)
        v.addWidget(g)
        g2 = QGroupBox("出力")
        f2 = QFormLayout(g2)
        self.out_dir = QLineEdit()
        b = QPushButton("参照…")
        b.clicked.connect(self.browse_out)
        hb = QHBoxLayout()
        hb.addWidget(self.out_dir)
        hb.addWidget(b)
        hw = QWidget()
        hw.setLayout(hb)
        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.save_cam = QCheckBox("カメラ視点の画像（鳥瞰変換前）も保存する")
        f2.addRow("出力フォルダ", hw)
        f2.addRow("並列プロセス数", self.workers)
        f2.addRow("", self.save_cam)
        v.addWidget(g2)
        info = QLabel("出力：bev/*.png（鳥瞰の白線画像）、camera/*.png（変換前）、labels.csv（κ・e_y・e_ψ・中心線の点）、"
                      "project.yaml（設定一式）、LABELS.txt（列の説明）")
        info.setWordWrap(True)
        v.addWidget(info)
        self.gen_btn = QPushButton("データ生成")
        self.gen_btn.setMinimumHeight(40)
        self.gen_btn.clicked.connect(self.start_generation)
        self.stop_btn = QPushButton("中止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        hb2 = QHBoxLayout()
        hb2.addWidget(self.gen_btn, 3)
        hb2.addWidget(self.stop_btn, 1)
        v.addLayout(hb2)
        self.pbar = QProgressBar()
        v.addWidget(self.pbar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(self.log)
        return w

    def _preview_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.fig = Figure(figsize=(6, 3.6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 3)
        hb = QHBoxLayout()
        self.cam_view = QLabel("カメラ画像（鳥瞰変換前）")
        self.bev_view = QLabel("鳥瞰画像（変換後）")
        for lab in (self.cam_view, self.bev_view):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(300, 240)
            lab.setStyleSheet("background:#111; color:#888")
        hb.addWidget(self.cam_view, 4)
        hb.addWidget(self.bev_view, 3)
        v.addLayout(hb, 3)
        hb2 = QHBoxLayout()
        hb2.addWidget(QLabel("フレーム"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        hb2.addWidget(self.slider)
        v.addLayout(hb2)
        self.info = QLabel("")
        self.info.setStyleSheet("font-family: monospace")
        v.addWidget(self.info)
        return w

    @staticmethod
    def _pair(a, b):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    # ------------------------------------------------------- course table
    def seg_add(self, seg=None):
        seg = seg or {"type": "straight", "length": 1.0}
        r = self.seg_table.rowCount()
        self.seg_table.insertRow(r)
        cb = QComboBox()
        cb.addItems(SEG_TYPES)
        cb.setCurrentText(seg["type"])
        cb.currentTextChanged.connect(lambda _t, row_cb=cb: self._seg_enable(self._row_of(row_cb)))
        self.seg_table.setCellWidget(r, 0, cb)
        defaults = {"length": 1.0, "radius": 0.7, "angle_deg": 90.0, "k0": 0.0, "k1": 0.0}
        for c, key in SEG_KEYS.items():
            self.seg_table.setItem(r, c, QTableWidgetItem(f"{float(seg.get(key, defaults[key])):g}"))
        self._seg_enable(r)

    def _row_of(self, widget):
        for r in range(self.seg_table.rowCount()):
            if self.seg_table.cellWidget(r, 0) is widget:
                return r
        return -1

    def _seg_enable(self, r):
        if r < 0:
            return
        t = self.seg_table.cellWidget(r, 0).currentText()
        for c in SEG_KEYS:
            it = self.seg_table.item(r, c)
            if it is None:
                continue
            used = c in SEG_USED[t]
            it.setFlags((it.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled) if used
                        else (it.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled))

    def seg_del(self):
        r = self.seg_table.currentRow()
        if r >= 0:
            self.seg_table.removeRow(r)

    def seg_move(self, d):
        r = self.seg_table.currentRow()
        segs = self.segments_from_table()
        if r < 0 or not (0 <= r + d < len(segs)):
            return
        segs[r], segs[r + d] = segs[r + d], segs[r]
        self.set_segments(segs)
        self.seg_table.selectRow(r + d)

    def segments_from_table(self):
        segs = []
        for r in range(self.seg_table.rowCount()):
            t = self.seg_table.cellWidget(r, 0).currentText()
            seg = {"type": t}
            for c in SEG_USED[t]:
                seg[SEG_KEYS[c]] = float(self.seg_table.item(r, c).text())
            segs.append(seg)
        return segs

    def set_segments(self, segs):
        self.seg_table.setRowCount(0)
        for s in segs:
            self.seg_add(s)

    # -------------------------------------------------- project <-> UI
    def project_to_ui(self):
        p = self.project
        cfg, c, d = p["config"], p["course"] or {}, p["dataset"]
        self.course_name.setText(c.get("name", "course"))
        self.closed_chk.setChecked(bool(c.get("closed", True)))
        self.lane_w.setValue(float(c.get("lane_width", 0.4)))
        self.line_w.setValue(float(c.get("line_width", 0.02)))
        self.center_line.setCurrentIndex(1 if c.get("center_line", "none") == "dashed" else 0)
        self.set_segments(c.get("segments", []))
        dr = cfg["driving"]
        self.speed.setValue(float(dr["speed"]))
        self.fps.setValue(float(dr["fps"]))
        self.offset_y.setValue(float(dr.get("offset_y", 0.0)))
        self.heading_off.setValue(float(dr.get("heading_offset_deg", 0.0)))
        self.start_s.setValue(float(dr.get("start_s", 0.0)))
        self.distance.setValue(float(dr.get("distance") or 0.0))
        self.dirs_fwd.setChecked("forward" in d["directions"])
        self.dirs_rev.setChecked("reverse" in d["directions"])
        self.speeds_txt.setText(fmt_list(d["speeds"]))
        self.offsets_txt.setText(fmt_list(d["offsets_y"]))
        self.headings_txt.setText(fmt_list(d["heading_offsets_deg"]))
        cm = cfg["camera"]
        self.cam_w.setValue(int(cm["width"]))
        self.cam_h.setValue(int(cm["height"]))
        for w_, k in [(self.fx, "fx"), (self.fy, "fy"), (self.cx, "cx"), (self.cy, "cy")]:
            w_.setValue(float(cm[k]))
        for w_, val in zip(self.dist, cm["dist"]):
            w_.setValue(float(val))
        for w_, val in zip(self.cam_xyz, cm["xyz"]):
            w_.setValue(float(val))
        for w_, val in zip(self.cam_rpy, cm["rpy_deg"]):
            w_.setValue(float(val))
        b = cfg["bev"]
        self.bev_x[0].setValue(b["x_min"]); self.bev_x[1].setValue(b["x_max"])
        self.bev_y[0].setValue(b["y_min"]); self.bev_y[1].setValue(b["y_max"])
        self.bev_res.setValue(b["res"])
        self.preview_s_txt.setText(fmt_list(cfg["labels"]["preview_s"]))
        self.path_s_txt.setText(fmt_list(cfg["labels"].get("path_s", [])))
        (self.mode_course if d["mode"] == "course" else self.mode_random).setChecked(True)
        self.random_n.setValue(int(d["random_samples"]))
        self.out_dir.setText(str(d["out_dir"]))
        self.workers.setValue(int(d["workers"]))
        self.save_cam.setChecked(bool(d.get("save_camera", True)))
        self.update_fov()
        self.update_bev_size()
        self.update_run_count()

    def ui_to_project(self) -> dict:
        p = copy.deepcopy(self.project)
        cfg = p["config"]
        p["course"] = dict(name=self.course_name.text() or "course", closed=self.closed_chk.isChecked(),
                           center_line="dashed" if self.center_line.currentIndex() == 1 else "none",
                           lane_width=self.lane_w.value(), line_width=self.line_w.value(),
                           segments=self.segments_from_table())
        cfg["track"]["center_line"] = p["course"]["center_line"]
        dr = cfg["driving"]
        dr.update(mode="exact", speed=self.speed.value(), fps=self.fps.value(), offset_y=self.offset_y.value(),
                  heading_offset_deg=self.heading_off.value(), start_s=self.start_s.value(),
                  distance=self.distance.value() or None)
        cfg["camera"].update(width=self.cam_w.value(), height=self.cam_h.value(), fx=self.fx.value(),
                             fy=self.fy.value(), cx=self.cx.value(), cy=self.cy.value(),
                             dist=[w_.value() for w_ in self.dist], xyz=[w_.value() for w_ in self.cam_xyz],
                             rpy_deg=[w_.value() for w_ in self.cam_rpy])
        cfg["bev"].update(x_min=self.bev_x[0].value(), x_max=self.bev_x[1].value(), y_min=self.bev_y[0].value(),
                          y_max=self.bev_y[1].value(), res=self.bev_res.value())
        cfg["labels"]["preview_s"] = parse_list(self.preview_s_txt.text())
        cfg["labels"]["path_s"] = parse_list(self.path_s_txt.text())
        cfg["labels"]["stations_x"] = cfg["labels"].get("stations_x", [])
        d = p["dataset"]
        d.update(mode="course" if self.mode_course.isChecked() else "random",
                 directions=[n for n, chk in (("forward", self.dirs_fwd), ("reverse", self.dirs_rev)) if chk.isChecked()],
                 speeds=parse_list(self.speeds_txt.text()), offsets_y=parse_list(self.offsets_txt.text()),
                 heading_offsets_deg=parse_list(self.headings_txt.text()), random_samples=self.random_n.value(),
                 out_dir=self.out_dir.text(), workers=self.workers.value(), save_camera=self.save_cam.isChecked())
        return p

    # ------------------------------------------------------------ helpers
    def update_fov(self):
        hf = 2 * np.degrees(np.arctan(self.cam_w.value() / 2 / self.fx.value()))
        vf = 2 * np.degrees(np.arctan(self.cam_h.value() / 2 / self.fy.value()))
        self.fov_label.setText(f"水平 {hf:.1f}°、垂直 {vf:.1f}°（歪みを除く）")

    def update_bev_size(self):
        w = int(round((self.bev_y[1].value() - self.bev_y[0].value()) / self.bev_res.value()))
        h = int(round((self.bev_x[1].value() - self.bev_x[0].value()) / self.bev_res.value()))
        self.bev_size.setText(f"{w} × {h} px（幅 × 高さ）")

    def update_run_count(self):
        try:
            n = (int(self.dirs_fwd.isChecked()) + int(self.dirs_rev.isChecked())) * len(parse_list(self.speeds_txt.text())) \
                * len(parse_list(self.offsets_txt.text())) * len(parse_list(self.headings_txt.text()))
            self.runs_label.setText(f"→ {n} 回走行")
        except ValueError:
            self.runs_label.setText("数値をカンマ区切りで入力してください")

    def log_msg(self, text):
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + text)

    # ----------------------------------------------------------- preview
    def update_preview(self):
        try:
            p = self.ui_to_project()
            cfg = p["config"]
            course = course_from_dict(p["course"], cfg["track"])
        except Exception as e:
            self.course_status.setText(f"<span style='color:#d33'>コースを作れません：{e}</span>")
            return
        c = course.center
        self.course_status.setText(
            f"中心線の長さ {c.length:.2f} m、最小半径 {1 / max(np.abs(c.kappa).max(), 1e-9):.2f} m"
            + ("（周回OK）" if c.closed else "（開いたコース）"))
        self.p_cfg, self.p_course = cfg, course
        self.gen = Generator(cfg, keep_camera=True)
        cam = self.gen.cam_nominal
        key = (tuple(cam.K.ravel()), tuple(cam.dist), cam.width, cam.height, cfg["render"]["supersample"])
        if key != self._renderer_key:
            self._rays = cam.pixel_rays(cfg["render"]["supersample"])
            self._renderer_key = key
        self.renderer = CameraRenderer(cam, cfg["render"]["supersample"], cfg["render"]["max_range"], self._rays)
        self.world = WorldRaster(course, cfg["render"]["world_res"])
        forward = self.direction.currentIndex() == 0
        self.path = course.lane_path(0, forward)
        self._frames = list(follow_exact(self.path, cfg["driving"]))
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self._frames) - 1))
        self.slider.setValue(min(self.slider.value(), len(self._frames) - 1))
        self.slider.blockSignals(False)
        self.draw_course()
        self.show_frame(self.slider.value())

    def draw_course(self, cur=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.tick_params(labelsize=8)
        course, cfg = self.p_course, self.p_cfg
        for o in course.boundary_offsets():
            q = course.center.offset(o)
            xy = np.vstack([q.xy, q.xy[:1]]) if q.closed else q.xy
            ax.plot(xy[:, 0], xy[:, 1], "k-", lw=1.5)
        if course.dashed_center:
            ax.plot(course.center.xy[:, 0], course.center.xy[:, 1], "k--", lw=0.8)
        ax.plot(self.path.xy[:, 0], self.path.xy[:, 1], "c-", lw=1, label="target path")
        if self._frames:
            P = np.array([(f["x"], f["y"]) for f in self._frames])
            ax.plot(P[:, 0], P[:, 1], "r-", lw=0.8, alpha=0.6, label="vehicle")
        if cur is not None:
            x, y, yaw = cur
            b = cfg["bev"]
            roi = np.array([[b["x_min"], b["y_min"]], [b["x_max"], b["y_min"]], [b["x_max"], b["y_max"]],
                            [b["x_min"], b["y_max"]], [b["x_min"], b["y_min"]]])
            cth, sth = np.cos(yaw), np.sin(yaw)
            ax.plot(x + cth * roi[:, 0] - sth * roi[:, 1], y + sth * roi[:, 0] + cth * roi[:, 1], "m-", lw=1)
            ax.arrow(x, y, 0.2 * cth, 0.2 * sth, width=0.025, color="r")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def show_frame(self, i):
        if not self._frames:
            return
        st = self._frames[int(i)]
        cfg = self.p_cfg
        R, t = base_pose_matrix(st["x"], st["y"], st["yaw"])
        prob, _ = self.renderer.render(self.world, R, t)
        cam_img = (np.clip(prob, 0, 1) * 255 + 0.5).astype(np.uint8)
        bev = self.gen.bev.warp(cam_img)
        lab = compute_labels(self.path, (st["x"], st["y"], st["yaw"]), st["idx"], cfg["labels"], cfg["bev"],
                             self.gen.cam_nominal, R, t, half_width=self.p_course.lane_width / 2)
        self.cam_view.setPixmap(QPixmap.fromImage(to_qimage(cam_img)).scaled(
            self.cam_view.width(), self.cam_view.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        rgb = np.repeat(bev[..., None], 3, axis=2)
        scale = 3
        rgb = np.kron(rgb, np.ones((scale, scale, 1), np.uint8))
        for (px, py), pv in zip(lab["path_xy"], lab["path_valid"]):
            u, v = self.gen.bev.base_to_px(px, py)
            pt = (int((u + 0.5) * scale), int((v + 0.5) * scale))
            if pv:
                cv2.circle(rgb, pt, 5, (0, 220, 0), -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(rgb, pt, (230, 40, 40), cv2.MARKER_TILTED_CROSS, 10, 2)
        self.bev_view.setPixmap(QPixmap.fromImage(to_qimage_rgb(rgb)).scaled(
            self.bev_view.width(), self.bev_view.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        ks = "  ".join(f"κ({s:g})={k:+.2f}" for s, k in zip(cfg["labels"]["preview_s"], lab["kappa"]))
        self.info.setText(f"frame {i}/{len(self._frames) - 1}  s={st['s']:.2f} m  e_y={lab['e_y'] * 100:+.1f} cm  "
                          f"e_ψ={np.degrees(lab['e_psi']):+.1f}°\n{ks}  [1/m]")
        self.draw_course((st["x"], st["y"], st["yaw"]))

    # -------------------------------------------------------------- files
    def load_course(self):
        fn, _ = QFileDialog.getOpenFileName(self, "コースYAML", str(COURSE_DIR), "YAML (*.yaml *.yml)")
        if fn:
            c = yaml.safe_load(open(fn))
            c.setdefault("name", Path(fn).stem)
            self.project = self.ui_to_project()
            self.project["course"] = c
            self.project_to_ui()
            self.update_preview()

    def save_course(self):
        fn, _ = QFileDialog.getSaveFileName(self, "コースYAMLを保存", str(COURSE_DIR / f"{self.course_name.text()}.yaml"),
                                            "YAML (*.yaml)")
        if fn:
            yaml.safe_dump(self.ui_to_project()["course"], open(fn, "w"), allow_unicode=True, sort_keys=False)
            self.log_msg(f"コースを保存：{fn}")

    def load_calibration(self):
        fn, _ = QFileDialog.getOpenFileName(self, "キャリブレーション結果", "", "YAML (*.yaml *.yml)")
        if not fn:
            return
        try:
            K, D, w, h = read_calibration(fn)
        except Exception as e:
            QMessageBox.warning(self, "読込エラー", str(e))
            return
        self.fx.setValue(K[0, 0]); self.fy.setValue(K[1, 1]); self.cx.setValue(K[0, 2]); self.cy.setValue(K[1, 2])
        for w_, val in zip(self.dist, list(D) + [0.0] * 5):
            w_.setValue(float(val))
        if w and h:
            self.cam_w.setValue(int(w)); self.cam_h.setValue(int(h))
        self.log_msg(f"内部パラメータを読込：{fn}")
        self.update_preview()

    def open_project(self):
        fn, _ = QFileDialog.getOpenFileName(self, "プロジェクト", "", "YAML (*.yaml *.yml)")
        if fn:
            self.load_project_file(fn)
            self.update_preview()

    def load_project_file(self, fn):
        p = yaml.safe_load(open(fn))
        base = default_project(load_config())
        base["config"].update(p.get("config", {}))
        base["dataset"].update(p.get("dataset", {}))
        base["course"] = p.get("course") or base["course"]
        self.project = base
        self.project_to_ui()
        self.log_msg(f"プロジェクトを読込：{fn}")

    def save_project(self):
        fn, _ = QFileDialog.getSaveFileName(self, "プロジェクトを保存", "project.yaml", "YAML (*.yaml)")
        if fn:
            yaml.safe_dump(self.ui_to_project(), open(fn, "w"), allow_unicode=True, sort_keys=False)
            self.log_msg(f"プロジェクトを保存：{fn}（scripts/make_dataset.py でも生成できます）")

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "出力フォルダ", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    # --------------------------------------------------------- generation
    def start_generation(self):
        try:
            p = self.ui_to_project()
            course_from_dict(p["course"], p["config"]["track"])
        except Exception as e:
            QMessageBox.warning(self, "設定エラー", str(e))
            return
        out = Path(p["dataset"]["out_dir"])
        if (out / "labels.csv").exists():
            if QMessageBox.question(self, "上書き確認", f"{out} には既にデータセットがあります。上書きしますか？") \
                    != QMessageBox.Yes:
                return
        self.project = p
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pbar.setValue(0)
        self.t0 = time.time()
        self.log_msg(f"データ生成を開始：{out}")
        self.thread = QThread()
        self.worker = GenWorker(p)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def stop_generation(self):
        if getattr(self, "worker", None):
            self.worker.stop = True
            self.log_msg("中止します…")

    def on_progress(self, done, total, frames):
        self.pbar.setMaximum(total)
        self.pbar.setValue(done)
        el = time.time() - self.t0
        eta = el / done * (total - done) if done else 0
        self.pbar.setFormat(f"{done}/{total} ジョブ  {frames} 枚  残り約 {eta:.0f} 秒")

    def on_finished(self, n, err):
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if err:
            self.log_msg(f"エラー：{err}")
            QMessageBox.critical(self, "エラー", err)
        else:
            self.log_msg(f"完了：{n} 枚（{time.time() - self.t0:.0f} 秒）→ {self.project['dataset']['out_dir']}")


def read_calibration(fn):
    """Read K, D (and image size) from a ROS camera_info YAML or an OpenCV FileStorage YAML."""
    text = open(fn).read()
    if text.lstrip().startswith("%YAML"):
        fs = cv2.FileStorage(fn, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        if D is None:
            D = fs.getNode("dist_coeffs").mat()
        w = fs.getNode("image_width").real() or 0
        h = fs.getNode("image_height").real() or 0
        return K, np.asarray(D).ravel(), w, h
    d = yaml.safe_load(text)
    K = np.asarray(d["camera_matrix"]["data"], float).reshape(3, 3)
    D = np.asarray(d["distortion_coefficients"]["data"], float)
    return K, D, d.get("image_width"), d.get("image_height")


def main():
    app = QApplication(sys.argv)
    win = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    win.show()
    sys.exit(app.exec())
