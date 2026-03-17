# sam2_gui/ui/sidebar.py
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QPushButton,
    QButtonGroup,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QStackedWidget,
    QSizePolicy,
    QFrame,
    QLineEdit,
    QInputDialog,
    QMenu,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import os


# ---------------------------------------------------------------------------
# Sub-panel state constants
# ---------------------------------------------------------------------------
MASK_STATE_LOCKED   = "locked"    # mode active, no mask loaded yet
MASK_STATE_LOADED   = "loaded"    # mask loaded, seeds placed, ready to segment
MASK_STATE_PERSISTS = "persists"  # image changed, mask carried forward


class Sidebar(QWidget):
    # Existing signals
    image_selected              = pyqtSignal(str)
    save_all_coco_requested     = pyqtSignal()
    eraser_toggled              = pyqtSignal(bool)
    segmentation_mode_changed   = pyqtSignal(str)   # "points","polygon","paint","manual_polygon","mask"
    polygon_sampling_changed    = pyqtSignal(str)   # "dense","random","poisson"
    mask_selected               = pyqtSignal(int)   # mask_id; -1 = none selected
    mask_renamed                = pyqtSignal(int, str)  # mask_id, new_name
    label_class_renamed         = pyqtSignal(str, str)  # old_name, new_name

    # Seed-mask signals (previously lived in seed_groupbox)
    load_seed_mask_requested    = pyqtSignal()
    clear_seed_mask_requested   = pyqtSignal()
    seed_points_now_requested   = pyqtSignal()
    auto_seed_toggled           = pyqtSignal(bool)

    def __init__(
            self,
            controller,
            on_run_segmentation,
            on_visibility_changed,
            on_clear_points,
            parent=None
    ):
        super().__init__(parent)
        self.controller           = controller
        self.on_run_segmentation  = on_run_segmentation
        self.on_visibility_changed = on_visibility_changed
        self.on_clear_points      = on_clear_points

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ----------------------------------------------------
        # Image list
        # ----------------------------------------------------
        layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self._on_image_double_clicked)
        layout.addWidget(self.image_list)

        # ----------------------------------------------------
        # Label Classes panel
        # ----------------------------------------------------
        lc_header_row = QHBoxLayout()
        lc_header_row.addWidget(QLabel("Label Classes:"))
        layout.addLayout(lc_header_row)

        self.label_class_list = QListWidget()
        self.label_class_list.setMaximumHeight(90)
        self.label_class_list.itemClicked.connect(self._on_label_class_clicked)
        self.label_class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_class_list.customContextMenuRequested.connect(self._on_label_class_context_menu)
        layout.addWidget(self.label_class_list)

        add_label_row = QHBoxLayout()
        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("New label name…")
        self.new_label_input.returnPressed.connect(self._add_label_class)
        add_label_row.addWidget(self.new_label_input)
        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(38)
        add_btn.clicked.connect(self._add_label_class)
        add_label_row.addWidget(add_btn)
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(26)
        del_btn.setToolTip("Delete selected label class")
        del_btn.clicked.connect(self._delete_label_class)
        add_label_row.addWidget(del_btn)
        layout.addLayout(add_label_row)

        # ----------------------------------------------------
        # Masks on this image
        # ----------------------------------------------------
        layout.addWidget(QLabel("Masks on this image:"))
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(100)
        self.mask_list.itemChanged.connect(self._on_item_changed)
        self.mask_list.itemSelectionChanged.connect(self._on_mask_selection_changed)
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self._on_mask_context_menu)
        layout.addWidget(self.mask_list)

        # Internal state
        self._label_classes = []   # list of str, ordered
        self._active_label  = None  # str or None

        # ----------------------------------------------------
        # Segment controls
        # ----------------------------------------------------
        seg_groupbox = QGroupBox("Segment")
        seg_layout   = QVBoxLayout(seg_groupbox)
        seg_layout.setContentsMargins(8, 10, 8, 8)
        seg_layout.setSpacing(6)

        top_row = QHBoxLayout()

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(self._on_erase_clicked)
        top_row.addWidget(self.erase_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.on_clear_points)
        top_row.addWidget(self.clear_points_btn)

        seg_layout.addLayout(top_row)

        self.run_btn = QPushButton("Segment (Enter)")
        self.run_btn.clicked.connect(self.on_run_segmentation)
        seg_layout.addWidget(self.run_btn)

        layout.addWidget(seg_groupbox)

        # ----------------------------------------------------
        # Segmentation Mode  (2-col grid + full-width SAM2 Mask)
        # ----------------------------------------------------
        mode_groupbox = QGroupBox("Segmentation Mode")
        mode_outer    = QVBoxLayout(mode_groupbox)
        mode_outer.setContentsMargins(8, 10, 8, 8)
        mode_outer.setSpacing(6)

        mode_grid         = QGridLayout()
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

        # Full-width SAM2 Mask button on its own row
        self.mask_btn = QPushButton("SAM2 Mask")
        self.mask_btn.setCheckable(True)
        self.mask_btn.clicked.connect(lambda: self._on_mode_button_clicked("mask"))
        mode_button_group.addButton(self.mask_btn)
        mode_grid.addWidget(self.mask_btn, 2, 0)

        self.manual_draw_btn = QPushButton("SAM2 Freehand")
        self.manual_draw_btn.setCheckable(True)
        self.manual_draw_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_draw"))
        mode_button_group.addButton(self.manual_draw_btn)
        mode_grid.addWidget(self.manual_draw_btn, 1, 1)

        self.manual_polygon_btn = QPushButton("Manual Polygon")
        self.manual_polygon_btn.setCheckable(True)
        self.manual_polygon_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_polygon"))
        mode_button_group.addButton(self.manual_polygon_btn)
        mode_grid.addWidget(self.manual_polygon_btn, 2, 1)

        mode_outer.addLayout(mode_grid)

        # ---- Mask sub-panel (QStackedWidget with 3 pages) ----
        self.mask_subpanel = QStackedWidget()
        self.mask_subpanel.setVisible(False)   # hidden until SAM2 Mask mode is active

        # Page 0: LOCKED — load gate
        self.mask_subpanel.addWidget(self._build_subpanel_locked())

        # Page 1: LOADED — full controls
        self.mask_subpanel.addWidget(self._build_subpanel_loaded())

        # Page 2: PERSISTS — amber carry-forward warning
        self.mask_subpanel.addWidget(self._build_subpanel_persists())

        mode_outer.addWidget(self.mask_subpanel)
        layout.addWidget(mode_groupbox)

        # ----------------------------------------------------
        # Polygon Sampling
        # ----------------------------------------------------
        self.sampling_groupbox = QGroupBox("Polygon Sampling")
        sampling_row   = QHBoxLayout()
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

        # start disabled (default mode = points)
        self._update_sampling_enabled_state(False)

        # ----------------------------------------------------
        # Save COCO
        # ----------------------------------------------------
        self.save_all_btn = QPushButton("Save COCO 1.0 (All Images)")
        self.save_all_btn.clicked.connect(self.save_all_coco_requested.emit)
        self.save_all_btn.setMinimumHeight(42)
        self.save_all_btn.setStyleSheet(
            "QPushButton { background-color: #5cb85c; color: white;"
            "              font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #4cae4c; }"
        )
        layout.addWidget(self.save_all_btn)

        self.setLayout(layout)

    # =========================================================
    # Sub-panel builders
    # =========================================================

    def _build_subpanel_locked(self):
        """
        State A: SAM2 Mask mode active, no mask loaded yet.
        Only 'Load Mask File...' is active. Everything else is ghosted.
        """
        w = QFrame()
        w.setObjectName("subpanel_locked")
        w.setStyleSheet(
            "#subpanel_locked { background: #efefef;"
            " border: 1px dashed #bbb; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        hint = QLabel("Load a mask to continue")
        hint.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        vbox.addWidget(hint)

        self.load_mask_btn_locked = QPushButton("Load Mask File...")
        self.load_mask_btn_locked.clicked.connect(self.load_seed_mask_requested.emit)
        vbox.addWidget(self.load_mask_btn_locked)

        ghost_label = QLabel("Sampling and auto-seed unlock after loading")
        ghost_label.setStyleSheet("color: #bbb; font-size: 10px;")
        ghost_label.setWordWrap(True)
        vbox.addWidget(ghost_label)

        return w

    def _build_subpanel_loaded(self):
        """
        State B: Mask confirmed, seeds placed. Full controls active.
        """
        w = QFrame()
        w.setObjectName("subpanel_loaded")
        w.setStyleSheet(
            "#subpanel_loaded { background: #f0faf0;"
            " border: 1px solid #a5d6a7; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        self.loaded_filename_label = QLabel("No file")
        self.loaded_filename_label.setStyleSheet(
            "color: #2e7d32; font-weight: bold; font-size: 11px;"
        )
        self.loaded_filename_label.setWordWrap(True)
        vbox.addWidget(self.loaded_filename_label)

        btn_row = QHBoxLayout()
        self.load_new_btn = QPushButton("Load New...")
        self.load_new_btn.clicked.connect(self.load_seed_mask_requested.emit)
        btn_row.addWidget(self.load_new_btn)

        self.unload_btn = QPushButton("Unload")
        self.unload_btn.clicked.connect(self.clear_seed_mask_requested.emit)
        btn_row.addWidget(self.unload_btn)
        vbox.addLayout(btn_row)

        self.auto_seed_checkbox = QCheckBox("Auto-Seed: ON")
        self.auto_seed_checkbox.setChecked(True)
        self.auto_seed_checkbox.toggled.connect(self._on_auto_seed_changed)
        vbox.addWidget(self.auto_seed_checkbox)

        self.seed_now_btn = QPushButton("Apply Seeds")
        self.seed_now_btn.setEnabled(False)   # only active when Auto-Seed is OFF
        self.seed_now_btn.clicked.connect(self.seed_points_now_requested.emit)
        vbox.addWidget(self.seed_now_btn)

        return w

    def _build_subpanel_persists(self):
        """
        State C: Image changed, mask carried forward. Amber warning.
        """
        w = QFrame()
        w.setObjectName("subpanel_persists")
        w.setStyleSheet(
            "#subpanel_persists { background: #fffde7;"
            " border: 1px solid #ffe082; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        title = QLabel("Mask carried forward")
        title.setStyleSheet("color: #b45309; font-weight: bold; font-size: 11px;")
        vbox.addWidget(title)

        self.persists_filename_label = QLabel("")
        self.persists_filename_label.setStyleSheet("color: #78350f; font-size: 10px;")
        self.persists_filename_label.setWordWrap(True)
        vbox.addWidget(self.persists_filename_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ffe082;")
        vbox.addWidget(sep)

        btn_row = QHBoxLayout()
        self.load_new_btn_persists = QPushButton("Load New...")
        self.load_new_btn_persists.clicked.connect(self.load_seed_mask_requested.emit)
        btn_row.addWidget(self.load_new_btn_persists)

        self.unload_btn_persists = QPushButton("Unload")
        self.unload_btn_persists.clicked.connect(self.clear_seed_mask_requested.emit)
        btn_row.addWidget(self.unload_btn_persists)
        vbox.addLayout(btn_row)

        # Mirror the auto-seed checkbox state (read-only display; actual checkbox is in loaded panel)
        self.auto_seed_persists_label = QLabel("Auto-Seed: ON")
        self.auto_seed_persists_label.setStyleSheet("color: #78350f; font-size: 10px;")
        vbox.addWidget(self.auto_seed_persists_label)

        return w

    # =========================================================
    # Public method: MainWindow calls this on image change or
    # mask load/unload to drive sub-panel state.
    # =========================================================

    def set_mask_subpanel_state(self, state: str, mask_path: str = None):
        """
        state: MASK_STATE_LOCKED | MASK_STATE_LOADED | MASK_STATE_PERSISTS
        mask_path: full path to loaded mask (used for filename display)
        """
        name = os.path.basename(mask_path) if mask_path else ""

        if state == MASK_STATE_LOCKED:
            self.mask_subpanel.setCurrentIndex(0)
        elif state == MASK_STATE_LOADED:
            self.loaded_filename_label.setText(f"✓ {name}")
            self.mask_subpanel.setCurrentIndex(1)
        elif state == MASK_STATE_PERSISTS:
            self.persists_filename_label.setText(
                f"{name} is still loaded from the previous image. "
                "Seeds have been re-applied automatically."
            )
            # Keep auto-seed label in sync
            auto_on = self.auto_seed_checkbox.isChecked()
            self.auto_seed_persists_label.setText(
                "Auto-Seed: ON" if auto_on else "Auto-Seed: OFF"
            )
            self.mask_subpanel.setCurrentIndex(2)

    # =========================================================
    # Segment button gating
    # =========================================================

    def update_segment_button_state(self):
        if self.controller is None:
            self.run_btn.setEnabled(False)
        else:
            has_points = (
                len(self.controller.fg_points) > 0
                or len(self.controller.bg_points) > 0
            )
            self.run_btn.setEnabled(has_points)

    # =========================================================
    # Mode button handler
    # =========================================================

    def _on_mode_button_clicked(self, mode: str):
        self.segmentation_mode_changed.emit(mode)

        is_mask    = mode == "mask"
        is_polygon = mode in ("polygon", "paint", "mask", "manual_draw")

        # Show / hide the mask sub-panel
        self.mask_subpanel.setVisible(is_mask)

        # Sampling enabled for polygon, paint, mask, and manual_draw modes
        self._update_sampling_enabled_state(is_polygon)

        # SAM2 Mask defaults to Random sampling
        if is_mask:
            self.random_radio.setChecked(True)
            self._on_sampling_button_clicked("random")

        # Style: green for polygon-family modes
        if is_polygon:
            color = "#4CAF50"
            for btn in (self.polygon_btn, self.paint_btn, self.mask_btn, self.manual_draw_btn):
                btn.setStyleSheet(
                    f"QPushButton:checked {{ background-color: {color}; color: black; }}"
                )
            self._update_sampling_button_style(color)
        else:
            for btn in (self.polygon_btn, self.paint_btn, self.mask_btn, self.manual_draw_btn):
                btn.setStyleSheet("")
            self._update_sampling_button_style("")

    def _update_sampling_enabled_state(self, enabled: bool):
        for btn in (self.dense_radio, self.random_radio, self.poisson_radio):
            btn.setEnabled(enabled)

    def _update_sampling_button_style(self, color: str):
        style = (
            f"QPushButton:checked {{ background-color: {color}; color: black; }}"
            if color else ""
        )
        for btn in (self.dense_radio, self.random_radio, self.poisson_radio):
            btn.setStyleSheet(style)

    def _on_sampling_button_clicked(self, mode: str):
        self.polygon_sampling_changed.emit(mode)

    # =========================================================
    # Erase button
    # =========================================================

    def _on_erase_clicked(self, checked: bool):
        self.eraser_toggled.emit(checked)

    # =========================================================
    # Auto-seed checkbox (lives inside the loaded sub-panel)
    # =========================================================

    def _on_auto_seed_changed(self, on: bool):
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if on else "Auto-Seed: OFF")
        self.seed_now_btn.setEnabled(not on)
        self.auto_seed_toggled.emit(on)

    # =========================================================
    # Called by MainWindow on image change
    # (replaces the old set_seed_controls_state signature)
    # =========================================================

    def set_seed_controls_state(self, auto_on: bool):
        """Sync auto-seed checkbox without triggering signals."""
        self.auto_seed_checkbox.blockSignals(True)
        self.auto_seed_checkbox.setChecked(auto_on)
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if auto_on else "Auto-Seed: OFF")
        self.seed_now_btn.setEnabled(not auto_on)
        self.auto_seed_checkbox.blockSignals(False)

    # =========================================================
    # Label Class management
    # =========================================================

    # Predefined distinct colors for label classes
    _LABEL_COLORS = [
        "#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa",
        "#00acc1", "#f4511e", "#6d4c41", "#3949ab", "#00897b",
    ]

    def _label_color(self, index: int) -> str:
        return self._LABEL_COLORS[index % len(self._LABEL_COLORS)]

    def _add_label_class(self):
        """Add a new label class from the input field."""
        name = self.new_label_input.text().strip()
        if not name or name in self._label_classes:
            self.new_label_input.clear()
            return
        self._label_classes.append(name)
        self._rebuild_label_class_list()
        if len(self._label_classes) == 1:
            self._set_active_label(name)
        self.new_label_input.clear()

    def add_label_classes_from_mask(self, names: list):
        """Called when SAM2 Mask file is loaded — populate classes from mask."""
        for name in names:
            if name and name not in self._label_classes:
                self._label_classes.append(name)
        self._rebuild_label_class_list()
        if not self._active_label and self._label_classes:
            self._set_active_label(self._label_classes[0])

    def _delete_label_class(self):
        row = self.label_class_list.currentRow()
        if row < 0:
            return
        name = self._label_classes[row]
        self._label_classes.pop(row)
        if self._active_label == name:
            self._active_label = self._label_classes[0] if self._label_classes else None
        self._rebuild_label_class_list()

    def _rebuild_label_class_list(self):
        self.label_class_list.blockSignals(True)
        self.label_class_list.clear()
        for i, name in enumerate(self._label_classes):
            item = QListWidgetItem(f"  {name}")
            item.setData(Qt.UserRole, name)
            color = QColor(self._label_color(i))
            item.setForeground(color)
            if name == self._active_label:
                item.setBackground(QColor("#dbeafe"))
            self.label_class_list.addItem(item)
        self.label_class_list.blockSignals(False)

    def _on_label_class_clicked(self, item: QListWidgetItem):
        self._set_active_label(item.data(Qt.UserRole))

    def _on_label_class_context_menu(self, pos):
        item = self.label_class_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename_action = QAction("Rename…", self)
        rename_action.triggered.connect(lambda: self._rename_label_class(item))
        menu.addAction(rename_action)
        menu.exec_(self.label_class_list.mapToGlobal(pos))

    def _rename_label_class(self, item: QListWidgetItem):
        old_name = item.data(Qt.UserRole)
        new_name, ok = QInputDialog.getText(
            self, "Rename Label", "New label name:", text=old_name
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return
        if new_name in self._label_classes:
            return  # already exists
        idx = self._label_classes.index(old_name)
        self._label_classes[idx] = new_name
        if self._active_label == old_name:
            self._active_label = new_name
        self._rebuild_label_class_list()
        self.label_class_renamed.emit(old_name, new_name)

    def _set_active_label(self, name: str):
        self._active_label = name
        self._rebuild_label_class_list()

    def get_active_label(self) -> str | None:
        """Return active label, or None if no label classes have been defined."""
        return self._active_label or None

    def get_label_classes(self) -> list:
        """Return the current ordered list of label class names."""
        return list(self._label_classes)

    def set_label_classes(self, names: list):
        """Replace all label classes with a new list (e.g. from CSV import)."""
        self._label_classes = list(names)
        # Reset active label to first class if current one no longer exists
        if self._active_label not in self._label_classes:
            self._active_label = self._label_classes[0] if self._label_classes else None
        self._rebuild_label_class_list()

    def get_color_for_label(self, name: str) -> tuple:
        """Return (r, g, b) tuple for a label class, consistent across all instances."""
        try:
            idx = self._label_classes.index(name)
        except ValueError:
            idx = abs(hash(name)) % len(self._LABEL_COLORS)
        hex_color = self._label_color(idx)
        # Convert hex string to (r, g, b)
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # =========================================================
    # Image list
    # =========================================================

    def _on_image_double_clicked(self, item):
        self.image_selected.emit(item.text())

    # =========================================================
    # Mask list (instances on this image)
    # =========================================================

    def refresh_masks(self):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for m in self.controller.masks:
            item = QListWidgetItem(m["label"])
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
            )
            item.setCheckState(Qt.Checked if m["visible"] else Qt.Unchecked)
            item.setData(Qt.UserRole, m["id"])
            self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem):
        mask_id = item.data(Qt.UserRole)
        self.controller.set_mask_visibility(mask_id, item.checkState() == Qt.Checked)
        self.on_visibility_changed()

    def _on_mask_selection_changed(self):
        items = self.mask_list.selectedItems()
        if items:
            self.mask_selected.emit(items[0].data(Qt.UserRole))
        else:
            self.mask_selected.emit(-1)

    def _on_mask_context_menu(self, pos):
        item = self.mask_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename_action = QAction("Rename…", self)
        rename_action.triggered.connect(lambda: self._rename_mask(item))
        menu.addAction(rename_action)
        menu.exec_(self.mask_list.mapToGlobal(pos))

    def _rename_mask(self, item: QListWidgetItem):
        if not self._label_classes:
            QInputDialog.getText(self, "No Labels", "No label classes defined.")
            return
        mask_id = item.data(Qt.UserRole)
        old_name = item.text()
        current_idx = self._label_classes.index(old_name) if old_name in self._label_classes else 0
        new_name, ok = QInputDialog.getItem(
            self, "Rename Mask", "Select label:",
            self._label_classes, current_idx, editable=False
        )
        if not ok or new_name == old_name:
            return
        self.mask_renamed.emit(mask_id, new_name)
