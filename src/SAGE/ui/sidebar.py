# sam2_gui/ui/sidebar.py
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QButtonGroup,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QRadioButton,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal


class Sidebar(QWidget):
    image_selected = pyqtSignal(str)
    save_all_coco_requested = pyqtSignal()
    load_seed_mask_requested = pyqtSignal()
    clear_seed_mask_requested = pyqtSignal()
    seed_points_now_requested = pyqtSignal()
    auto_seed_toggled = pyqtSignal(bool)
    eraser_toggled = pyqtSignal(bool)
    segmentation_mode_changed = pyqtSignal(str)  # "points", "polygon", "paint", or "manual_polygon"
    polygon_sampling_changed = pyqtSignal(str)  # "dense", "random", "poisson"

    def __init__(
            self,
            controller,
            on_run_segmentation,
            on_opacity_changed,
            on_visibility_changed,
            on_clear_points,
            parent=None
    ):
        super().__init__(parent)
        self.controller = controller
        self.on_run_segmentation = on_run_segmentation
        self.on_opacity_changed = on_opacity_changed
        self.on_visibility_changed = on_visibility_changed
        self.on_clear_points = on_clear_points

        layout = QVBoxLayout()

        # ----------------------------------------------------
        # Image list (top)
        # ----------------------------------------------------
        layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self._on_image_double_clicked)
        layout.addWidget(self.image_list)

        # ----------------------------------------------------
        # Seed/Prompt Controls
        # ----------------------------------------------------
        seed_groupbox = QGroupBox("Seed/Prompt Controls")
        seed_layout = QVBoxLayout(seed_groupbox)
        seed_layout.setContentsMargins(8, 10, 8, 8)
        seed_layout.setSpacing(6)

        # Row: Load + Clear
        row = QHBoxLayout()

        self.load_seed_btn = QPushButton("Load Seed Mask")
        self.load_seed_btn.clicked.connect(self.load_seed_mask_requested.emit)
        row.addWidget(self.load_seed_btn)

        self.clear_seed_btn = QPushButton("Unload")
        self.clear_seed_btn.setEnabled(False)  # disabled until something is loaded
        self.clear_seed_btn.clicked.connect(self.clear_seed_mask_requested.emit)
        row.addWidget(self.clear_seed_btn)

        seed_layout.addLayout(row)

        # status line (the label you added earlier)
        self.seed_mask_status = QLabel("")
        self.seed_mask_status.setStyleSheet("color: gray; font-size: 24px;")
        self.seed_mask_status.setVisible(False)
        seed_layout.addWidget(self.seed_mask_status)

        self.seed_now_btn = QPushButton("Apply Seeds")
        self.seed_now_btn.clicked.connect(self.seed_points_now_requested.emit)

        # Make buttons fully visible + consistent height
        for b in (self.load_seed_btn, self.seed_now_btn):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(28)  # tweak 26–32 if you want
            b.setMinimumWidth(0)
            b.setContentsMargins(6, 4, 6, 4)

        seed_layout.addWidget(self.seed_now_btn)

        self.auto_seed_checkbox = QCheckBox("Auto-Seed: ON")
        self.auto_seed_checkbox.setChecked(True)
        self.seed_now_btn.setEnabled(False)
        self.auto_seed_checkbox.toggled.connect(self._on_auto_seed_changed)
        seed_layout.addWidget(self.auto_seed_checkbox)

        # Prevent this groupbox from being squashed too much
        seed_groupbox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout.addWidget(seed_groupbox)

        # ----------------------------------------------------
        # Mask list (bottom)
        # ----------------------------------------------------
        layout.addWidget(QLabel("Masks / Labels:"))
        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.mask_list)

        # ----------------------------------------------------
        # Segment button + Clear Points button
        # ----------------------------------------------------
        seg_groupbox = QGroupBox("Segment")
        seg_layout = QVBoxLayout(seg_groupbox)
        seg_layout.setContentsMargins(8, 10, 8, 8)
        seg_layout.setSpacing(6)

        # Row 1: Erase toggle button + Clear Points
        top_row = QHBoxLayout()

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(self._on_erase_clicked)
        top_row.addWidget(self.erase_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.on_clear_points)
        top_row.addWidget(self.clear_points_btn)

        seg_layout.addLayout(top_row)

        # Row 2: Segment button centered / full width
        self.run_btn = QPushButton("Segment (Enter)")
        self.run_btn.clicked.connect(self.on_run_segmentation)
        seg_layout.addWidget(self.run_btn)

        opacity_row = QVBoxLayout()
        opacity_row.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(120)
        self.opacity_slider.setMinimumHeight(18)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider)

        seg_layout.addLayout(opacity_row)

        layout.addWidget(seg_groupbox)

        # ----------------------------------------------------
        # Segmentation mode (2x2 grid in groupbox)
        # ----------------------------------------------------
        mode_groupbox = QGroupBox("Segmentation Mode")
        mode_grid = QGridLayout()

        mode_button_group = QButtonGroup(self)
        mode_button_group.setExclusive(True)

        self.points_btn = QPushButton("SAM2 Points")
        self.points_btn.setCheckable(True)
        self.points_btn.setChecked(True)
        self.points_btn.clicked.connect(lambda: self._on_mode_button_clicked("points"))
        mode_button_group.addButton(self.points_btn)
        mode_grid.addWidget(self.points_btn, 0, 0)

        self.polygon_btn = QPushButton("SAM2 Polygon")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(lambda: self._on_mode_button_clicked("polygon"))
        mode_button_group.addButton(self.polygon_btn)
        mode_grid.addWidget(self.polygon_btn, 0, 1)

        self.paint_btn = QPushButton("SAM2 Paint")
        self.paint_btn.setCheckable(True)
        self.paint_btn.clicked.connect(lambda: self._on_mode_button_clicked("paint"))
        mode_button_group.addButton(self.paint_btn)
        mode_grid.addWidget(self.paint_btn, 1, 0)

        self.manual_polygon_btn = QPushButton("Manual Polygon")
        self.manual_polygon_btn.setCheckable(True)
        self.manual_polygon_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_polygon"))
        mode_button_group.addButton(self.manual_polygon_btn)
        mode_grid.addWidget(self.manual_polygon_btn, 1, 1)

        mode_groupbox.setLayout(mode_grid)
        layout.addWidget(mode_groupbox)

        # ----------------------------------------------------
        # Polygon sampling strategy (horizontal row)
        # ----------------------------------------------------
        self.sampling_groupbox = QGroupBox("Polygon Sampling")
        sampling_row = QHBoxLayout()

        sampling_group = QButtonGroup(self)
        sampling_group.setExclusive(True)

        self.dense_radio = QPushButton("Grid")
        self.dense_radio.setCheckable(True)
        self.dense_radio.setChecked(True)
        self.dense_radio.clicked.connect(lambda: self._on_sampling_button_clicked("dense"))
        sampling_group.addButton(self.dense_radio)
        sampling_row.addWidget(self.dense_radio)

        self.random_radio = QPushButton("Random")
        self.random_radio.setCheckable(True)
        self.random_radio.clicked.connect(lambda: self._on_sampling_button_clicked("random"))
        sampling_group.addButton(self.random_radio)
        sampling_row.addWidget(self.random_radio)

        self.poisson_radio = QPushButton("Disc")
        self.poisson_radio.setCheckable(True)
        self.poisson_radio.clicked.connect(lambda: self._on_sampling_button_clicked("poisson"))
        sampling_group.addButton(self.poisson_radio)
        sampling_row.addWidget(self.poisson_radio)

        self.sampling_groupbox.setLayout(sampling_row)
        layout.addWidget(self.sampling_groupbox)

        self._update_sampling_enabled_state(False)  # start disabled because default mode = points

        # ----------------------------------------------------
        # Save ALL COCO button → emits signal
        # ----------------------------------------------------
        self.save_all_btn = QPushButton("Save COCO 1.0 (All Images)")
        self.save_all_btn.clicked.connect(self.save_all_coco_requested.emit)
        layout.addWidget(self.save_all_btn)
        self.save_all_btn.setMinimumHeight(42)
        self.save_all_btn.setStyleSheet(
            "QPushButton { background-color: #5cb85c; color: white; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #4cae4c; }"
        )

        self.setLayout(layout)

    def _on_erase_clicked(self, checked: bool):
        self.eraser_toggled.emit(checked)
    # ----------------------------------------------------
    def update_segment_button_state(self):
        """Enable/disable segment button based on whether points exist"""
        if self.controller is None:
            self.run_btn.setEnabled(False)
        else:
            has_points = len(self.controller.fg_points) > 0 or len(self.controller.bg_points) > 0
            self.run_btn.setEnabled(has_points)

    # ----------------------------------------------------
    # Mode button handlers
    # ----------------------------------------------------
    def _on_mode_button_clicked(self, mode):
        """Handle mode button clicks"""
        self.segmentation_mode_changed.emit(mode)

        # Enable/disable sampling options based on mode
        if mode == "polygon":
            self._update_sampling_enabled_state(True)
            # Make polygon button and sampling buttons green with black text
            self.polygon_btn.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: black; }")
            self._update_sampling_button_style("#4CAF50")
        else:
            self._update_sampling_enabled_state(False)
            # Reset polygon button style
            self.polygon_btn.setStyleSheet("")
            self._update_sampling_button_style("")

    def _update_sampling_enabled_state(self, enabled: bool):
        self.dense_radio.setEnabled(enabled)
        self.random_radio.setEnabled(enabled)
        self.poisson_radio.setEnabled(enabled)

    def _update_sampling_button_style(self, color):
        """Update the style of sampling buttons"""
        if color:
            style = f"QPushButton:checked {{ background-color: {color}; color: black; }}"
            self.dense_radio.setStyleSheet(style)
            self.random_radio.setStyleSheet(style)
            self.poisson_radio.setStyleSheet(style)
        else:
            self.dense_radio.setStyleSheet("")
            self.random_radio.setStyleSheet("")
            self.poisson_radio.setStyleSheet("")

    def _on_sampling_button_clicked(self, mode):
        """Handle sampling button clicks"""
        self.polygon_sampling_changed.emit(mode)

    # ----------------------------------------------------
    # Double-click image → emit filename ONLY
    # ----------------------------------------------------
    def _on_image_double_clicked(self, item):
        filename = item.text()
        self.image_selected.emit(filename)

    # ----------------------------------------------------
    # Mask list refresh
    # ----------------------------------------------------
    def refresh_masks(self):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for m in self.controller.masks:
            item = QListWidgetItem(m["label"])
            item.setFlags(
                item.flags()
                | Qt.ItemIsEditable
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
            )
            item.setCheckState(Qt.Checked if m["visible"] else Qt.Unchecked)
            item.setData(Qt.UserRole, m["id"])
            self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)

    # ----------------------------------------------------
    # Mask visibility / label changes
    # ----------------------------------------------------
    def _on_item_changed(self, item: QListWidgetItem):
        mask_id = item.data(Qt.UserRole)
        label = item.text()
        visible = item.checkState() == Qt.Checked

        self.controller.set_mask_label(mask_id, label)
        self.controller.set_mask_visibility(mask_id, visible)

        self.on_visibility_changed()

    # ---------------- Seed/eraser UI helpers ----------------
    def _on_auto_seed_changed(self, on: bool):
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if on else "Auto-Seed: OFF")
        self.seed_now_btn.setEnabled(not on)
        self.auto_seed_toggled.emit(on)

    def set_seed_controls_state(self, auto_on: bool):
        """Called by MainWindow when an image changes."""
        self.auto_seed_checkbox.blockSignals(True)

        self.auto_seed_checkbox.setChecked(auto_on)
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if auto_on else "Auto-Seed: OFF")

        self.auto_seed_checkbox.blockSignals(False)

    # ------------- Mask path showing UI helpers -------------
    def set_seed_mask_status(self, mask_path):
        if mask_path:
            import os
            name = os.path.basename(mask_path)
            self.seed_mask_status.setText(f"Loaded: {name}")
            self.seed_mask_status.setVisible(True)
            self.clear_seed_btn.setEnabled(True)
        else:
            self.seed_mask_status.setText("")
            self.seed_mask_status.setVisible(False)
            self.clear_seed_btn.setEnabled(False)