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
)
from PyQt5.QtCore import Qt, pyqtSignal


class Sidebar(QWidget):
    image_selected = pyqtSignal(str)
    save_all_coco_requested = pyqtSignal()
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
        # Mask list (bottom)
        # ----------------------------------------------------
        layout.addWidget(QLabel("Masks / Labels:"))
        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.mask_list)

        # ----------------------------------------------------
        # Segment button + Clear Points button
        # ----------------------------------------------------
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Segment (Enter)")
        self.run_btn.clicked.connect(self.on_run_segmentation)
        btn_layout.addWidget(self.run_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.on_clear_points)
        btn_layout.addWidget(self.clear_points_btn)

        layout.addLayout(btn_layout)

        # ----------------------------------------------------
        # Opacity slider
        # ----------------------------------------------------
        layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(120)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        layout.addWidget(self.opacity_slider)

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

        self.setLayout(layout)

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