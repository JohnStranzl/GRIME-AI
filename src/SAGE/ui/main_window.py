# sam2_gui/ui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
QSizePolicy
)
from PyQt5.QtCore import Qt
import os
import copy
import json
import math
import random

import numpy as np

from SAGE.utils.image_io import load_image_rgb
from SAGE.core.controller import SegmentationController
from SAGE.core.renderer import Renderer
from SAGE.ui.canvas import Canvas
from SAGE.ui.sidebar import Sidebar
from SAGE.ui.dialogs import ask_for_label
from SAGE.ui.mask_item import MaskItem
from SAGE.settings_manager import SettingsManager
from SAGE.utils.colors import get_color_for_index
from SAGE.utils.mask_ops import compute_mask_stats


class MainWindow(QMainWindow):
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation & Annotation for Geospatial Ecohydrology (SAGE)")

        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Store model manager for reloading images
        self.model_manager = model_manager

        # Mask store + current image tracking
        self.mask_store = {}
        self.current_image_path = None

        # Polygon sampling mode
        self.polygon_sampling_mode = "dense"  # "dense", "random", "poisson"

        # Controller and renderer will be created when first image is loaded
        self.controller = None
        self.renderer = None
        self.image_np = None

        # Seed mask (global / site-level)
        self.seed_mask_path = None  # path to .tif/.tiff mask
        self.seed_mask_bool = None  # cached boolean mask resized to current image
        self.auto_seed_enabled = True  # auto-apply when each image loads
        self.eraser_radius = 18

        central = QWidget()

        # ---------------------------------------------------------
        # TOP-LEVEL LAYOUT: VERTICAL (folder row above everything)
        # ---------------------------------------------------------
        main_layout = QVBoxLayout(central)

        # Folder row ABOVE the image
        ROW_HEIGHT = 28
        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(4, 4, 4, 4)
        folder_row.setSpacing(6)
        self.folder_edit = QLineEdit()
        self.folder_edit.setFixedHeight(ROW_HEIGHT)
        self.folder_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.folder_browse_btn = QPushButton("Browse")
        self.folder_browse_btn.setFixedHeight(ROW_HEIGHT)
        self.folder_browse_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.folder_browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.folder_browse_btn)
        main_layout.addLayout(folder_row)

        # ---------------------------------------------------------
        # BELOW THAT: Canvas + Sidebar horizontally
        # ---------------------------------------------------------
        layout = QHBoxLayout()

        self.canvas = Canvas(
            on_left_click=self._on_left_click,
            on_right_click=self._on_right_click_handler,
            parent=self
        )
        self.canvas.eraser_move.connect(self._erase_seeds_at)
        layout.addWidget(self.canvas, stretch=4)

        self.sidebar = Sidebar(
            controller=None,  # Will be set when image is loaded
            on_run_segmentation=self._run_segmentation,
            on_opacity_changed=self._on_opacity_changed,
            on_visibility_changed=self._update_canvas,
            on_clear_points=self._clear_points,
            parent=self,
        )

        # Sidebar emits ONLY the filename → MainWindow builds full path
        self.sidebar.image_selected.connect(self._load_new_image)

        # Sidebar requests COCO save → MainWindow handles it
        self.sidebar.save_all_coco_requested.connect(self.save_all_coco)

        # Sidebar seed/prompt control
        self.sidebar.load_seed_mask_requested.connect(self._browse_seed_mask)
        self.sidebar.clear_seed_mask_requested.connect(self._clear_seed_mask)
        self.sidebar.seed_points_now_requested.connect(self._seed_points_from_mask_current_image)
        self.sidebar.auto_seed_toggled.connect(self._on_auto_seed_toggled)
        self.sidebar.eraser_toggled.connect(self._on_eraser_toggled)

        # New: segmentation mode + sampling mode signals
        self.sidebar.segmentation_mode_changed.connect(self._on_segmentation_mode_changed)
        self.sidebar.polygon_sampling_changed.connect(self._on_polygon_sampling_changed)

        # Canvas polygon signal - handle both SAM2 and manual
        self.canvas.polygon_drawn.connect(self._on_polygon_drawn_dispatcher)

        layout.addWidget(self.sidebar, stretch=1)
        main_layout.addLayout(layout)

        self.setCentralWidget(central)
        self.showMaximized()

        # Load saved folder path and populate
        saved_folder = self.settings_manager.get_folder_path()
        if saved_folder and os.path.isdir(saved_folder):
            self.folder_edit.setText(saved_folder)
            self._populate_image_list(saved_folder)

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_auto_seed_toggled(self, on: bool):
        self.auto_seed_enabled = on

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_eraser_toggled(self, on: bool):
        self.canvas.set_eraser_enabled(on)
        # optional: change cursor
        self.canvas.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_polygon_drawn_dispatcher(self, points):
        """Route to SAM2 or manual polygon handler based on mode"""
        if self.canvas._segmentation_mode == "manual_polygon":
            self._on_manual_polygon_drawn(points)
        else:
            self._on_polygon_drawn(points)  # Existing SAM2 polygon handler

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_manual_polygon_drawn(self, points):
        """
        Create mask directly from polygon without SAM2.
        For ambiguous features like grass where SAM2 fails.
        """
        if self.controller is None:
            return

        if len(points) < 3:
            return

        import cv2

        # Ask for label first
        default_label = f"Region {len(self.controller.masks) + 1}"
        label = ask_for_label(self, default_label)

        if not label:  # User cancelled
            return

        # Create mask from polygon
        h, w = self.image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_array], 1)
        mask = mask.astype(bool)

        # Create mask entry manually (without SAM2)
        mask_id = next(self.controller._mask_id_counter)
        color = get_color_for_index(len(self.controller.masks))
        stats = compute_mask_stats(mask)

        mask_entry = {
            "id": mask_id,
            "label": label,
            "mask": mask,
            "color": color,
            "visible": True,
            "stats": stats,
        }

        self.controller.masks.append(mask_entry)

        self.sidebar.refresh_masks()
        self._update_canvas()

    # ---------------------------------------------------------
    # Folder browsing
    # ---------------------------------------------------------
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_edit.setText(folder)

            # Save folder path to settings
            self.settings_manager.set_folder_path(folder)

            # Populate image list
            self._populate_image_list(folder)

    # ---------------------------------------------------------
    # Seed mask browsing
    # ---------------------------------------------------------
    def _browse_seed_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Seed Mask",
            "",
            "Mask Files (*.tif *.tiff *.png *.jpg *.jpeg *.npy)"
        )
        if not path:
            return

        self.seed_mask_path = path
        self.sidebar.set_seed_mask_status(self.seed_mask_path)

        # Load once (resized when image is loaded)
        self.seed_mask_bool = None  # reset cache
        QMessageBox.information(self, "Seed Mask Loaded", f"Loaded:\n{path}")

    def _load_seed_mask_bool(self, target_shape):
        """
        Returns boolean mask (H,W) resized to target_shape using nearest-neighbor.
        Caches result per-image-size to avoid re-reading on every image.
        """
        if self.seed_mask_path is None:
            return None

        # If cached and already correct size, reuse
        if self.seed_mask_bool is not None and self.seed_mask_bool.shape == target_shape:
            return self.seed_mask_bool

        path = self.seed_mask_path
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext in [".tif", ".tiff"]:
                from PIL import Image
                m = np.array(Image.open(path))
            elif ext == ".npy":
                m = np.load(path)
            else:
                import cv2
                m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if m is None:
                    return None
                if m.ndim == 3:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

            mask = self._normalize_seed_mask(m)
            print("Seed mask foreground fraction:", mask.mean())

            # Resize if needed
            if mask.shape != target_shape:
                import cv2
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = (mask_u8 > 0)

            self.seed_mask_bool = mask
            return mask

        except Exception as e:
            print("Seed mask load error:", e)
            return None

    def _normalize_seed_mask(self, mask):
        """
        Returns boolean mask where True = ROI (foreground).

        Handles common cases:
        - binary 0/255 masks
        - 0/1 masks
        - float masks
        - masks with huge white background (auto-invert)
        """
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]

        m = mask.astype(np.float32)
        m = np.nan_to_num(m, nan=0.0)

        # If it looks binary-ish, threshold at >0
        uniq = np.unique(m)
        if uniq.size <= 10:
            roi = m > 0
        else:
            # otherwise use Otsu (robust for grayscale)
            import cv2
            mm = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, thr = cv2.threshold(mm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi = thr > 0

        # Auto-invert if ROI covers most of the image (common when background is white)
        # If > 50% is True, it's probably inverted.
        if roi.mean() > 0.5:
            roi = ~roi

        return roi

    def _clear_seed_mask(self):
        # Reset seed mask state
        self.seed_mask_path = None
        self.seed_mask_bool = None

        # Clear any UI text field if you still have it (safe even if removed)
        if hasattr(self, "seed_mask_edit"):
            self.seed_mask_edit.setText("")

        # Update sidebar status line
        self.sidebar.set_seed_mask_status(None)

    # ---------------------------------------------------------
    # Seed points from input mask
    # ---------------------------------------------------------
    def _seed_points_from_mask_current_image(self):
        # If user manually requests seeding, ensure this image is not skipped
        # if self.current_image_path in self.skip_seed_for_images:
        #     self.skip_seed_for_images.remove(self.current_image_path)
            # self._update_seed_skip_button_text()

        if self.controller is None or self.image_np is None:
            return
        if not self.seed_mask_path:
            QMessageBox.warning(self, "No Seed Mask", "Please load a seed mask (.tif) first.")
            return

        # Load / resize seed mask to match current image
        mask_bool = self._load_seed_mask_bool(target_shape=self.image_np.shape[:2])
        if mask_bool is None or mask_bool.sum() == 0:
            QMessageBox.warning(self, "Invalid Seed Mask", "Seed mask is empty or could not be loaded.")
            return

        seed = abs(hash(self.current_image_path)) % (2 ** 32)
        fg_points, bg_points = self._sample_points_from_mask(mask_bool, n_fg=30, n_bg=0, seed=seed)

        # Put points into the existing controller path (supported)
        self.controller.clear_points()
        for x, y in fg_points:
            self.controller.add_point(x, y, is_fg=True)
        for x, y in bg_points:
            self.controller.add_point(x, y, is_fg=False)

        print("Seed mask stats:", mask_bool.shape, "fg_frac=", float(mask_bool.mean()))
        self._update_canvas()

    def _sample_points_from_mask(self, mask_bool, n_fg=30, n_bg=30, seed=0):
        rng = np.random.default_rng(seed)
        h, w = mask_bool.shape

        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return [], []

        # FG points inside ROI
        k_fg = min(n_fg, len(xs))
        idx_fg = rng.choice(len(xs), size=k_fg, replace=False)
        fg_points = [(int(xs[i]), int(ys[i])) for i in idx_fg]

        if n_bg <= 0:
            return fg_points, []

        # BG candidates: outside ROI
        outside = ~mask_bool

        # Build "safe BG zones"
        safe = np.zeros_like(mask_bool, dtype=bool)

        # sky strip
        safe[: int(0.15 * h), :] = True
        # bottom strip
        safe[int(0.90 * h):, :] = True
        # left/right margins
        safe[:, : int(0.05 * w)] = True
        safe[:, int(0.95 * w):] = True

        bg_candidate = outside & safe
        bys, bxs = np.where(bg_candidate)

        # fallback: if safe zones empty, use any outside
        if len(bxs) == 0:
            bys, bxs = np.where(outside)

        if len(bxs) == 0:
            return fg_points, []

        k_bg = min(n_bg, len(bxs))
        idx_bg = rng.choice(len(bxs), size=k_bg, replace=False)
        bg_points = [(int(bxs[i]), int(bys[i])) for i in idx_bg]

        return fg_points, bg_points

    # ---------------------------------------------------------
    # Populate mask name
    # ---------------------------------------------------------
    def _default_label_from_seed_mask(self):
        """
        Extract class name from seed mask filename.

        Expected pattern:
          site_CLASS_otherinfo.tif
          e.g. ninemileprairie_GR_2000_03.tif

        Returns:
          "GR"
        """
        if not self.seed_mask_path:
            return None

        name = os.path.basename(self.seed_mask_path)
        stem = os.path.splitext(name)[0]

        parts = stem.split("_")
        if len(parts) < 2:
            return None

        class_name = parts[1]  # <-- CLASS POSITION (fixed)
        return class_name

    # ---------------------------------------------------------
    # Erase seed points
    # ---------------------------------------------------------
    def _erase_seeds_at(self, x, y):
        if self.controller is None:
            return

        # 1) remove SAM points + manual points from controller
        self.controller.remove_points_in_circle(x, y, radius=self.eraser_radius)

        # 2) also remove paint stroke points (works for (x,y) or (x,y,is_fg))
        if hasattr(self.canvas, "_paint_points") and self.canvas._paint_points:
            r2 = float(self.eraser_radius) ** 2
            new_pts = []
            for p in self.canvas._paint_points:
                px, py = p[0], p[1]  # <-- only take first two always
                dx = px - x
                dy = py - y
                if (dx * dx + dy * dy) > r2:
                    new_pts.append(p)  # keep original tuple as-is (2 or 3 items)
            self.canvas._paint_points = new_pts

        self._update_canvas()

    def _toggle_seed_eraser(self):
        self.eraser_enabled = self.seed_eraser_btn.isChecked()
        self.seed_eraser_btn.setText("Eraser: ON" if self.eraser_enabled else "Eraser: OFF")
        self.canvas.set_eraser_enabled(self.eraser_enabled)

    # ---------------------------------------------------------
    # Populate image list
    # ---------------------------------------------------------
    def _populate_image_list(self, folder):
        """Populate the sidebar image list from a folder"""
        self.sidebar.image_list.clear()
        image_files = []
        for name in os.listdir(folder):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_files.append(name)
                self.sidebar.image_list.addItem(name)

        # Automatically load the first image if any exist
        if image_files:
            self._load_new_image(image_files[0])

    # ---------------------------------------------------------
    # Clear Points
    # ---------------------------------------------------------
    def _clear_points(self):
        if self.controller is None:
            return
        self.controller.clear_points()

        # Clear paint stroke visuals if they exist
        if hasattr(self.canvas, '_paint_points'):
            self.canvas._paint_points = []

        self._update_canvas()

    # ---------------------------------------------------------
    # Interaction handlers
    # ---------------------------------------------------------
    def _on_left_click(self, x, y):
        if self.controller is None:
            return
        self.controller.add_point(x, y, is_fg=True)
        self._update_canvas()

    def _on_right_click_handler(self, x_or_mask_id, y=None):
        """
        Handle right-clicks: either on a mask (mask_id only)
        or on empty space (x, y coordinates)
        """
        if self.controller is None:
            return

        if y is None:
            # It's a mask_id - delete it
            reply = QMessageBox.question(
                self,
                'Delete Mask',
                'Delete this mask?',
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.controller.delete_mask(x_or_mask_id)
                self.sidebar.refresh_masks()
                self._update_canvas()
        else:
            # It's coordinates - add negative point
            self.controller.add_point(x_or_mask_id, y, is_fg=False)
            self._update_canvas()

    def _run_segmentation(self):
        if self.controller is None:
            return

        seed_label = self._default_label_from_seed_mask()
        fallback = f"Region {len(self.controller.masks) + 1}"
        default_label = seed_label or fallback
        label = ask_for_label(self, default_label)
        mask_entry = self.controller.run_segmentation(label=label)

        if mask_entry is not None:
            # If in paint mode, constrain to FOREGROUND painted area only
            if self.canvas._segmentation_mode == "paint" and len(self.controller.fg_points) > 0:
                import cv2

                # Get bounding box of ONLY foreground (green) painted points
                fg_array = np.array(self.controller.fg_points, dtype=np.int32)
                x_min = max(0, int(fg_array[:, 0].min()) - 30)
                x_max = min(self.image_np.shape[1], int(fg_array[:, 0].max()) + 30)
                y_min = max(0, int(fg_array[:, 1].min()) - 30)
                y_max = min(self.image_np.shape[0], int(fg_array[:, 1].max()) + 30)

                # Create constraint mask
                h, w = self.image_np.shape[:2]
                constraint_mask = np.zeros((h, w), dtype=np.uint8)
                constraint_mask[y_min:y_max, x_min:x_max] = 1

                # Apply spatial constraint
                mask_entry["mask"] = mask_entry["mask"] & constraint_mask.astype(bool)

                # Recompute stats
                mask_entry["stats"] = compute_mask_stats(mask_entry["mask"])

                # Clear paint visuals
                if hasattr(self.canvas, '_paint_points'):
                    self.canvas._paint_points = []

            self.sidebar.refresh_masks()
            self._update_canvas()

    def _on_opacity_changed(self, value):
        if self.controller is None:
            return
        self.controller.set_opacity(value)
        self._update_canvas()

    # ---------------------------------------------------------
    # Segmentation mode + polygon sampling handlers
    # ---------------------------------------------------------
    def _on_segmentation_mode_changed(self, mode: str):
        self.canvas.set_segmentation_mode(mode)

    def _on_polygon_sampling_changed(self, mode: str):
        if mode in ("dense", "random", "poisson"):
            self.polygon_sampling_mode = mode

    # ---------------------------------------------------------
    # Polygon drawn handler
    # ---------------------------------------------------------
    def _on_polygon_drawn(self, points):
        """
        points: list of (x, y) tuples defining a closed polygon in image coords.
        We sample interior points, run segmentation, then SPATIALLY CONSTRAIN
        the result to only include pixels inside the polygon boundary.
        """
        if self.controller is None:
            return

        if len(points) < 3:
            return

        # Sample interior points
        interior_points = self._sample_points_inside_polygon(points)

        if not interior_points:
            return

        for x, y in interior_points:
            self.controller.add_point(x, y, is_fg=True)

        # Run segmentation with polygon constraint
        seed_label = self._default_label_from_seed_mask()
        fallback = f"Region {len(self.controller.masks) + 1}"
        default_label = seed_label or fallback
        label = ask_for_label(self, default_label)
        mask_entry = self.controller.run_segmentation(label=label)

        if mask_entry is not None:
            # CRITICAL: Constrain mask to polygon area
            import cv2

            # Create polygon mask
            h, w = self.image_np.shape[:2]
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            polygon_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(poly_mask, [polygon_array], 1)

            # Apply spatial constraint: intersect SAM2 result with polygon
            mask_entry["mask"] = mask_entry["mask"] & poly_mask.astype(bool)

            # Recompute stats for the constrained mask
            mask_entry["stats"] = compute_mask_stats(mask_entry["mask"])

            self.sidebar.refresh_masks()
            self._update_canvas()

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------
    def _update_canvas(self):
        if self.controller is None or self.renderer is None:
            return

        base_pixmap = self.renderer.base_pixmap()
        masks = self.controller.get_visible_masks()
        pixmap_with_masks = self.renderer.overlay_masks(
            base_pixmap, masks, opacity=self.controller.opacity
        )
        pixmap_with_points = self.renderer.draw_points(
            pixmap_with_masks,
            self.controller.fg_points,
            self.controller.bg_points,
        )
        self.canvas.set_pixmap(pixmap_with_points)

        # Add invisible mask items for right-click detection
        self._add_mask_items_to_canvas()

        # Update segment button state
        self.sidebar.update_segment_button_state()

    def _add_mask_items_to_canvas(self):
        """Add invisible MaskItem polygons to canvas for right-click detection"""
        import cv2

        # First, remove any existing mask items
        for item in self.canvas._scene.items():
            if isinstance(item, MaskItem):
                self.canvas._scene.removeItem(item)

        # Add new mask items for all masks (not just visible ones, so you can delete hidden ones too)
        if self.controller is None:
            return

        for m in self.controller.masks:
            mask = m["mask"].astype(np.uint8)

            # Extract contours to get polygon points
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Add a MaskItem for each contour
            for cnt in contours:
                if len(cnt) < 3:
                    continue

                # Convert contour to list of (x, y) tuples
                polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]

                # Create and add the invisible mask item
                mask_item = MaskItem(polygon_points, m["id"])
                self.canvas._scene.addItem(mask_item)

    def keyPressEvent(self, event):
        key = event.key()

        # Enter keys - run segmentation
        if key in (13, 16777220):
            self._run_segmentation()

        # Backspace - remove last point
        elif key == Qt.Key_Backspace:
            if self.controller is not None:
                self.controller.remove_last_point()
                self._update_canvas()

        super().keyPressEvent(event)

    # ---------------------------------------------------------
    # Polygon sampling strategies
    # ---------------------------------------------------------
    def _sample_points_inside_polygon(self, polygon_points):
        polygon = np.array(polygon_points, dtype=float)
        xs = polygon[:, 0]
        ys = polygon[:, 1]

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        if max_x <= min_x or max_y <= min_y:
            return []

        if self.polygon_sampling_mode == "dense":
            return self._sample_dense_grid(polygon, min_x, max_x, min_y, max_y)
        elif self.polygon_sampling_mode == "random":
            return self._sample_random_uniform(polygon, min_x, max_x, min_y, max_y)
        else:  # "poisson"
            return self._sample_poisson_disk(polygon, min_x, max_x, min_y, max_y)

    def _point_in_polygon(self, x, y, polygon):
        """
        Standard ray-casting point-in-polygon test.
        polygon: Nx2 array
        """
        num = len(polygon)
        inside = False
        j = num - 1
        for i in range(num):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            # Check if point is between yi and yj in y, and to the left of the edge
            intersect = ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def _sample_dense_grid(self, polygon, min_x, max_x, min_y, max_y, step=5):
        points = []
        # Step as int pixels
        min_x_int = int(math.floor(min_x))
        max_x_int = int(math.ceil(max_x))
        min_y_int = int(math.floor(min_y))
        max_y_int = int(math.ceil(max_y))

        for x in range(min_x_int, max_x_int + 1, step):
            for y in range(min_y_int, max_y_int + 1, step):
                if self._point_in_polygon(x + 0.5, y + 0.5, polygon):
                    points.append((x + 0.5, y + 0.5))
        return points

    def _sample_random_uniform(
            self, polygon, min_x, max_x, min_y, max_y, num_points=300
    ):
        points = []
        attempts = 0
        max_attempts = num_points * 20

        while len(points) < num_points and attempts < max_attempts:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._point_in_polygon(x, y, polygon):
                points.append((x, y))
            attempts += 1

        return points

    def _sample_poisson_disk(
            self, polygon, min_x, max_x, min_y, max_y, radius=8.0, k=30
    ):
        """
        Simple Poisson disk sampling (Bridson) restricted to polygon.
        radius: minimum distance between points.
        k: attempts per active point.
        """
        cell_size = radius / math.sqrt(2)
        grid_width = int(math.ceil((max_x - min_x) / cell_size))
        grid_height = int(math.ceil((max_y - min_y) / cell_size))

        # Grid cells store indices into samples list or -1
        grid = [[-1 for _ in range(grid_height)] for _ in range(grid_width)]
        samples = []
        active = []

        def grid_coords(px, py):
            gx = int((px - min_x) / cell_size)
            gy = int((py - min_y) / cell_size)
            return gx, gy

        # Initialize with one random point inside polygon
        init_attempts = 0
        while True:
            if init_attempts > 1000:
                return []  # fallback
            init_x = random.uniform(min_x, max_x)
            init_y = random.uniform(min_y, max_y)
            if self._point_in_polygon(init_x, init_y, polygon):
                samples.append((init_x, init_y))
                gx, gy = grid_coords(init_x, init_y)
                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                    grid[gx][gy] = 0
                active.append(0)
                break
            init_attempts += 1

        while active:
            idx = random.choice(active)
            base_x, base_y = samples[idx]
            found = False

            for _ in range(k):
                r = random.uniform(radius, 2 * radius)
                theta = random.uniform(0, 2 * math.pi)
                nx = base_x + r * math.cos(theta)
                ny = base_y + r * math.sin(theta)

                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                if not self._point_in_polygon(nx, ny, polygon):
                    continue

                gx, gy = grid_coords(nx, ny)
                if not (0 <= gx < grid_width and 0 <= gy < grid_height):
                    continue

                ok = True
                # Check neighbors in grid
                for ix in range(max(gx - 2, 0), min(gx + 3, grid_width)):
                    for iy in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                        s_idx = grid[ix][iy]
                        if s_idx != -1:
                            sx, sy = samples[s_idx]
                            if (sx - nx) ** 2 + (sy - ny) ** 2 < radius ** 2:
                                ok = False
                                break
                    if not ok:
                        break

                if ok:
                    samples.append((nx, ny))
                    grid[gx][gy] = len(samples) - 1
                    active.append(len(samples) - 1)
                    found = True

            if not found:
                active.remove(idx)

        return samples

    # ---------------------------------------------------------
    # Save COCO for ALL images with labels
    # ---------------------------------------------------------
    def save_all_coco(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            return

        # Save current image's masks into mask_store
        if self.current_image_path and self.controller is not None:
            self.mask_store[self.current_image_path] = copy.deepcopy(self.controller.masks)

        # All images in folder
        all_images = [
            name for name in os.listdir(folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]
        total_images = len(all_images)

        # Ask user where to save the COCO file
        default_path = os.path.join(folder, "Annotations.json")
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save COCO Annotation File",
            default_path,
            "JSON Files (*.json)",
        )
        if not filepath:
            return

        import cv2

        # Step 1: collect all unique labels
        label_set = set()
        for masks in self.mask_store.values():
            for m in masks:
                label_set.add(m["label"])

        label_to_id = {label: i + 1 for i, label in enumerate(sorted(label_set))}
        categories = [{"id": cid, "name": label} for label, cid in label_to_id.items()]

        coco = {
            "images": [],
            "annotations": [],
            "categories": categories,
        }

        image_id = 1
        ann_id = 1
        saved = 0
        skipped = 0

        for filename in all_images:
            full_path = os.path.join(folder, filename)
            masks = self.mask_store.get(full_path, [])

            if not masks:
                skipped += 1
                continue

            image_np = load_image_rgb(full_path)
            height, width = image_np.shape[:2]

            coco["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            })

            # Process masks directly - no need to create a controller
            for m in masks:
                mask = m["mask"].astype("uint8")

                import cv2 as _cv2
                contours, _ = _cv2.findContours(
                    mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
                )

                segmentation = []
                area = 0
                bbox = None

                for cnt in contours:
                    if len(cnt) < 3:
                        continue

                    poly = cnt.reshape(-1, 2).tolist()
                    segmentation.append([coord for point in poly for coord in point])

                    area += _cv2.contourArea(cnt)
                    x, y, w, h = _cv2.boundingRect(cnt)
                    if bbox is None:
                        bbox = [x, y, w, h]

                if not segmentation:
                    continue

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": label_to_id[m["label"]],
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "label": m["label"],  # optional, for your own tools
                })

                ann_id += 1

            saved += 1
            image_id += 1

        # Write final COCO file
        with open(filepath, "w") as f:
            json.dump(coco, f, indent=2)

        # Summary dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("COCO Export Summary")
        msg.setText(
            f"Total images: {total_images}\n"
            f"Saved with labels: {saved}\n"
            f"Skipped (no labels): {skipped}\n\n"
            f"Saved to:\n{filepath}"
        )
        msg.exec_()

    # ---------------------------------------------------------
    # Load a new image when double-clicked in the sidebar
    # ---------------------------------------------------------
    def _load_new_image(self, filename):
        folder = self.folder_edit.text().strip()
        if not folder:
            return

        full_path = os.path.join(folder, filename)

        # Save masks for current image
        if self.current_image_path and self.controller is not None:
            self.mask_store[self.current_image_path] = copy.deepcopy(self.controller.masks)

        # Update current image path
        self.current_image_path = full_path

        # Load new image
        self.image_np = load_image_rgb(full_path)

        # Reset controller + renderer
        self.controller = SegmentationController(self.model_manager, self.image_np)
        self.renderer = Renderer(self.image_np)

        # Restore masks if they exist
        if full_path in self.mask_store:
            self.controller.masks = copy.deepcopy(self.mask_store[full_path])

        # Auto-seed points from seed mask if available
        if self.seed_mask_path and self.auto_seed_enabled:
            self._seed_points_from_mask_current_image()

        # update sidebar checkbox state to reflect this image
        self.sidebar.set_seed_controls_state(
            auto_on=self.auto_seed_enabled,
            # eraser_on=self.canvas._eraser_enabled
        )

        # Update sidebar controller reference
        self.sidebar.controller = self.controller
        self.sidebar.refresh_masks()

        # Update canvas
        self._update_canvas()