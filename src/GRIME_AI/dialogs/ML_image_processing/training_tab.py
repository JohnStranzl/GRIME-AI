import os

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import json

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QAbstractItemView, QSizePolicy, QListWidget, QMessageBox

from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE, BUTTON_CSS_DARK_RED
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

# Optional: if there is a training entry point, import it. Replace with the actual path/class.
#try:
#    from GRIME_AI.training.trainer import Trainer
#except Exception:
#    Trainer = None


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====   MODULE LEVEL HELPERS    =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
def _check_folder(folder: Path) -> Tuple[bool, List[str]]:
    """
    Validates a single folder by:
      1. Finding at least one .json COCO file and some .jpg/.jpeg images.
      2. Parsing the JSON’s "images" list of dicts to pull out file_name.
      3. Verifying every listed file_name exists in that folder.

    Returns (is_valid, missing_files_list).
    """
    # 1) List JSONs and JPGs via os.scandir
    jsons = [e.name for e in os.scandir(folder)
             if e.is_file() and e.name.lower().endswith(".json")]
    jpgs = {e.name for e in os.scandir(folder)
            if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg"))}

    print(f"Scanning `{folder}` -> JSONs: {jsons}, JPGs: {list(jpgs)[:5]}…")  # debug

    if not jsons or not jpgs:
        return False, []

    # 2) Load the first JSON file
    path_json = folder / jsons[0]
    try:
        data = json.loads(path_json.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Cannot parse {jsons[0]}: {e}"]

    # 3) Extract expected filenames from COCO "images" list
    raw_images = data.get("images")
    if not isinstance(raw_images, list):
        return False, [f"'images' key missing or not a list in {jsons[0]}"]

    expected_files = []
    for item in raw_images:
        if isinstance(item, dict):
            fname = item.get("file_name") or item.get("filename")
            if not fname:
                return False, [f"Missing 'file_name' in entry: {item}"]
            expected_files.append(Path(fname).name)
        elif isinstance(item, str):
            expected_files.append(item)
        else:
            return False, [f"Unsupported image entry type: {type(item)}"]

    # 4) Compare against the actual JPGs on disk
    missing = [f for f in expected_files if f not in jpgs]
    if missing:
        return False, missing

    return True, []

def _iter_dirs(root: Path) -> List[Path]:
    """
    Recursively yield every subdirectory under root using os.scandir.
    """
    for entry in os.scandir(root):
        if entry.is_dir():
            sub = Path(entry.path)
            yield sub
            yield from _iter_dirs(sub)

# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
# Custom ListWidget classes with drag and drop support.
class DraggableListWidget(QListWidget):
    def mimeData(self, items):
        mimeData = QtCore.QMimeData()
        texts = "\n".join(sorted(set(item.text() for item in self.selectedItems())))
        mimeData.setText(texts)
        return mimeData


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class DroppableListWidget(QListWidget):
    def dropEvent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text()
            items_to_drop = [line.strip() for line in text.splitlines() if line.strip()]
            dlg = self.parent()
            if dlg is not None:
                for item_text in items_to_drop:
                    # Remove matching item from available list.
                    available = dlg.listWidget_availableFolders
                    for idx in range(available.count()):
                        avail_item = available.item(idx)
                        if avail_item.text() == item_text:
                            available.takeItem(idx)
                            break
                    # Add the item if not already transferred.
                    if item_text not in dlg.transferred_items:
                        self.addItem(item_text)
                        dlg.transferred_items.add(item_text)
                        print(f"Dragged '{item_text}' from available to selected folders via drop.")
                dlg.listWidget_selectedFolders.repaint()
                dlg._refresh_annotations_from_selection()
                dlg.updateTrainButtonState()
            event.accept()
        else:
            event.ignore()

# ======================================================================================================================
# Main training tab widget
# ======================================================================================================================
class TrainingTab(QtWidgets.QWidget):
    ml_train_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).parent / "training_tab.ui"
        uic.loadUi(str(ui_path), self)

        # Default selection
        self.selected_training_model = "sam2"

        # Replace default QListWidgets with drag/drop versions while preserving object names and layout positions
        self._install_drag_drop_lists()

        # State holders
        self.categories_available: bool = False
        self.annotation_list: List[Dict[str, Any]] = []
        self.unique_training_labels: List[str] = []
        self.transferred_items: Set[str] = set()
        self.original_folders = []
        self.categories_available = False

        # Detect optional labels list widget
        self._init_labels_widget_reference()

        # Init config manager
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        config_file = Path(settings_folder) / "site_config.json"
        self._mgr = ModelConfigManager(str(config_file))

        # Load config (creates/writes template if missing/empty/invalid)
        self.site_config: Dict[str, Any] = self._mgr.load_config(return_type="dict")

        # --- LoRA UI: keep all LoRA controls together so we can enable/disable them ---
        self._lora_widgets = []

        # Main LoRA hyperparam group
        if hasattr(self, "groupBox_loraHyperparameters"):
            self._lora_widgets.append(self.groupBox_loraHyperparameters)

        # Start with LoRA controls disabled; they will be enabled only when LoRA is selected
        self._set_lora_enabled(False)

        # Populate UI from config and initialize state
        self._populate_ui_from_config(self.site_config)
        #self.setup_custom_list_widgets()
        self.setup_ui_properties()
        self.setup_drag_and_drop()
        self.setup_connections()

        self.reset_selection()
        self.populate_available_folders()

        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.listWidget_availableFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setMinimumHeight(200)

        self.listWidget_selectedFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_selectedFolders.setMinimumHeight(200)

        self.adjustSize()

        self.setMinimumSize(self.size())

        self.verticalTabParametersLayout.setStretch(0, 1)
        self.verticalTabParametersLayout.setStretch(1, 0)
        self.horizontalMainLayout.setStretch(0, 1)
        self.horizontalMainLayout.setStretch(1, 3)
        self.horizontalListLayout.setStretch(0, 1)
        self.horizontalListLayout.setStretch(1, 0)
        self.horizontalListLayout.setStretch(2, 1)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_drag_and_drop(self):
        """Configure drag & drop for folder lists and set style for the Segment button."""
        self.listWidget_availableFolders.setDragEnabled(True)
        self.listWidget_availableFolders.setDragDropMode(QAbstractItemView.DragOnly)
        self.listWidget_selectedFolders.setAcceptDrops(True)
        self.listWidget_selectedFolders.setDragDropMode(QAbstractItemView.DropOnly)
        self.listWidget_selectedFolders.installEventFilter(self)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def eventFilter(self, source, event):
        """
        Process drag-and-drop events on the selected folders list.
        """
        if source == self.listWidget_selectedFolders:
            if event.type() in (QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.DragMove):
                event.accept()
                return True
            elif event.type() == QtCore.QEvent.Type.Drop:
                if event.mimeData().hasText():
                    mime_text = event.mimeData().text()
                    dragged_items = [txt.strip() for txt in mime_text.splitlines() if txt.strip()]
                    for txt in dragged_items:
                        for idx in range(self.listWidget_availableFolders.count()):
                            avail_item = self.listWidget_availableFolders.item(idx)
                            if avail_item.text() == txt:
                                self.listWidget_availableFolders.takeItem(idx)
                                break
                        if txt not in self.transferred_items:
                            self.listWidget_selectedFolders.addItem(txt)
                            self.transferred_items.add(txt)
                            print(f"Dragged '{txt}' from available to selected folders via eventFilter.")
                event.accept()
                self._refresh_annotations_from_selection()
                self.updateTrainButtonState()
                return True

        return super().eventFilter(source, event)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    '''
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.tabWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_selectedFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setMinimumHeight(200)
        self.listWidget_selectedFolders.setMinimumHeight(200)
        self.adjustSize()
        self.setMinimumSize(self.size())
        self.verticalTabParametersLayout.setStretch(0, 1)
        self.verticalTabParametersLayout.setStretch(1, 0)
        self.horizontalMainLayout.setStretch(0, 1)
        self.horizontalMainLayout.setStretch(1, 3)
        self.horizontalListLayout.setStretch(0, 1)
        self.horizontalListLayout.setStretch(1, 0)
        self.horizontalListLayout.setStretch(2, 1)

        # Set stylesheet for the tabs to change color when a tab is selected.
        self.tabWidget.setStyleSheet("""
            QTabBar::tab {
                background-color: white;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: steelblue;
                color: white;
            }
        """)
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_connections(self):
        """Connect signals with their slot methods."""

        # Wire up signals
        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_reset.clicked.connect(self.reset_selection)
        self.pushButton_moveRight.clicked.connect(self.move_to_right)
        self.pushButton_moveLeft.clicked.connect(self.move_to_available)
        self.lineEdit_siteName.editingFinished.connect(self.save_site_name_to_json)

        self.pushButton_browse_model_training_images_folder.clicked.connect(self.browse_model_training_images_folder)
        model_training_image_folder = JsonEditor().getValue("Model_Training_Images_Folder")
        if model_training_image_folder:
            self.lineEdit_model_training_images_path.setText(model_training_image_folder)
        self.lineEdit_model_training_images_path.editingFinished.connect(self.populate_available_folders)

        self.pushButton_moveRight.clicked.connect(self.move_to_right)
        self.pushButton_moveRight.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_moveLeft.clicked.connect(self.move_to_left)
        self.pushButton_moveLeft.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_reset.clicked.connect(self.reset_lists)
        self.pushButton_reset.setStyleSheet(BUTTON_CSS_DARK_RED)

        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_train.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.listWidget_availableFolders.itemDoubleClicked.connect(self.handle_left_item_doubleclick)

        self.listWidget_selectedFolders.itemDoubleClicked.connect(self.handle_right_item_doubleclick)
        self.listWidget_selectedFolders.itemSelectionChanged.connect(self.updateTrainButtonState)

        self.comboBox_train_label_selection.currentIndexChanged.connect(self.on_train_label_changed)

        # Default selection
        self.selected_training_model = "sam2"

        # Connect signals
        self.radioButton_train_model_SAM2.toggled.connect(lambda checked: self.set_training_model("sam2", checked))
        self.radioButton_train_model_LoRA.toggled.connect(lambda checked: self.set_training_model("segformer", checked))
        self.radioButton_train_model_MaskRCNN.toggled.connect(lambda checked: self.set_training_model("maskrcnn", checked))

        #self.buttonBox_close.rejected.connect(self.reject)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def handle_left_item_doubleclick(self, item):
        self.move_items_to_selected([item])

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def handle_right_item_doubleclick(self, item):
        self.move_items_to_available([item])

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def resizeEvent(self, event):
        """
        Ensure list widgets update their geometry on dialog resize.
        """
        super().resizeEvent(event)
        self.listWidget_availableFolders.updateGeometry()
        self.listWidget_selectedFolders.updateGeometry()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def initialize_dialog_from_config(self, config):
        self.site_config = config
        self.setup_from_config_file()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_from_config_file(self):
        """
        Initialize dialog controls from a configuration dictionary.
        """
        self.lineEdit_siteName.setText(self.site_config.get("siteName", ""))
        learningRates = self.site_config.get("learningRates", [])
        lr_str = ", ".join(str(x) for x in learningRates)
        self.lineEdit_learningRates.setText(lr_str)

        optimizer = self.site_config.get("Optimizer", "")
        idx = self.comboBox_optimizer.findText(optimizer)
        if idx >= 0:
            self.comboBox_optimizer.setCurrentIndex(idx)

        loss_function = self.site_config.get("loss_function", "")
        idx = self.comboBox_lossFunction.findText(loss_function)
        if idx >= 0:
            self.comboBox_lossFunction.setCurrentIndex(idx)

        self.doubleSpinBox_weightDecay.setValue(self.site_config.get("weight_decay", 0.0))
        self.spinBox_epochs.setValue(self.site_config.get("number_of_epochs", 0))
        self.spinBox_batchSize.setValue(self.site_config.get("batch_size", 0))
        self.spinBox_saveFrequency.setValue(self.site_config.get("save_model_frequency", 0))
        self.spinBox_validationFrequency.setValue(self.site_config.get("validation_frequency", 0))
        self.checkBox_earlyStopping.setChecked(self.site_config.get("early_stopping", False))
        self.spinBox_patience.setValue(self.site_config.get("patience", 0))

        device = self.site_config.get("device", "")
        idx = self.comboBox_device.findText(device)
        if idx >= 0:
            self.comboBox_device.setCurrentIndex(idx)

        self.current_path = self.site_config.get("Path", None)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def reset_lists(self):
        """
        Clear both list widgets and restore available folders from the original list.
        """
        self.listWidget_availableFolders.clear()
        self.listWidget_selectedFolders.clear()
        self.transferred_items.clear()
        for folder in self.original_folders:
            self.listWidget_availableFolders.addItem(QListWidgetItem(folder))

        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def reject(self):
        # DO NOTHING. LET IT CLOSE. IF THE CALLING PROGRAM CREATED THE DIALOG USING EXEC, THE CALLING INSTANTIATING
        # PROGRAM CAN INSPECT THE RETURN RESULT
        super().reject()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_to_left(self):
        """
        Move selected items back from the selected folders list to the available folders list,
        then re-sort the available folders.
        """
        selected_items = self.listWidget_selectedFolders.selectedItems()
        for item in selected_items:
            if item:
                row = self.listWidget_selectedFolders.row(item)
                self.listWidget_selectedFolders.takeItem(row)
                if item.text() in self.transferred_items:
                    self.transferred_items.remove(item.text())
                available_items = [self.listWidget_availableFolders.item(i).text() for i in range(self.listWidget_availableFolders.count())]
                available_items.append(item.text())
                available_items.sort()
                self.listWidget_availableFolders.clear()
                for text in available_items:
                    self.listWidget_availableFolders.addItem(text)
                print(f"Moved '{item.text()}' from selected back to available folders (sorted, button).")

        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_items_to_available(self, items):
        for item in items:
            name = item.text()
            if name in self.transferred_items:
                self.transferred_items.remove(name)
                available_items = [self.listWidget_availableFolders.item(i).text()
                                   for i in range(self.listWidget_availableFolders.count())]
                available_items.append(name)
                available_items.sort()
                self.listWidget_availableFolders.clear()
                for text in available_items:
                    self.listWidget_availableFolders.addItem(text)
                for i in range(self.listWidget_selectedFolders.count()):
                    if self.listWidget_selectedFolders.item(i).text() == name:
                        self.listWidget_selectedFolders.takeItem(i)
                        break
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_training_model(self, model_name: str, checked: bool):
        """Update selected_training_model when a radio button is toggled on."""
        if checked:  # only update when the button is checked, not unchecked
            self.selected_training_model = model_name
            print(f"Selected training model: {self.selected_training_model}")
            self._update_lora_ui_for_model() # Turn LoRA controls on/off depending on selected model

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA-specific controls in one place."""
        for w in getattr(self, "_lora_widgets", []):
            if w is not None:
                w.setEnabled(enabled)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _update_lora_ui_for_model(self):
        """
        Turn LoRA section on only when the LoRA/SegFormer training model
        is selected; keep it greyed out for SAM2 and Mask R-CNN.
        """
        is_lora = (getattr(self, "selected_training_model", "") == "segformer")
        self._set_lora_enabled(is_lora)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_train_label_changed(self, index: int):
        selected_training_labels = self.get_selected_training_labels()
        print("\n".join(
            [f"Selected label ID: {item['label_id']}, Name: {item['label_name']}" for item in selected_training_labels]))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_selected_training_labels(self):
        """
        Fetch the selected training label from comboBox_train_label_selection
        and return it as a list of dictionaries.

        Each dictionary contains:
        - "label_id": the parsed ID from the comboBox text
        - "label_name": the parsed label name

        Returns
        -------
        list[dict]
            A list with one dictionary if selection is valid, else an empty list.
        """
        selected_text = self.comboBox_train_label_selection.currentText().strip()

        if "-" not in selected_text:
            return []  # malformed or empty selection

        label_id, label_name = map(str.strip, selected_text.split("-", 1))

        return [{
            "label_id": label_id,
            "label_name": label_name
        }]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_custom_list_widgets(self):
        """Replace default list widgets with custom draggable/droppable ones."""
        self.listWidget_availableFolders.__class__ = DraggableListWidget
        self.listWidget_selectedFolders.__class__ = DroppableListWidget

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def browse_model_training_images_folder(self):
        """Open a dialog to choose a folder and update the folder path field."""
        folder = QFileDialog.getExistingDirectory(self, "Select training images folder", os.getcwd())
        folder = os.path.normpath(folder)

        if folder:
            self.lineEdit_model_training_images_path.setText(folder)
            self.populate_available_folders()

            JsonEditor().update_json_entry("Model_Training_Images_Folder", folder)

            self.updateTrainButtonState()

    # --------------------------------------------------------------------------------------------------------------
    # Setup helpers
    # --------------------------------------------------------------------------------------------------------------
    def _find_container_layout_and_index(self, widget):
        """
        Return (layout, index) for the slot in which `widget` resides.
        Traverses parent().layout() and its sub-layouts to find the actual item index.
        """
        parent = widget.parent()
        if not parent:
            return None, -1

        root_layout = parent.layout()
        if not isinstance(root_layout, QtWidgets.QLayout):
            return None, -1

        # breadth-first search over nested layouts
        queue = [root_layout]
        while queue:
            lay = queue.pop(0)
            for i in range(lay.count()):
                item = lay.itemAt(i)
                # Direct widget slot
                if item and item.widget() is widget:
                    return lay, i
                # Nested layout slot
                child_layout = item.layout() if item else None
                if isinstance(child_layout, QtWidgets.QLayout):
                    queue.append(child_layout)

        return None, -1

    def _install_drag_drop_lists(self) -> None:
        # Replace available list
        avail_layout, avail_index = self._find_container_layout_and_index(self.listWidget_availableFolders)
        if avail_layout is None or avail_index < 0:
            print("Could not locate layout slot for listWidget_availableFolders; aborting replacement.")
        else:
            avail_dd = DraggableListWidget(self.listWidget_availableFolders.parent())
            avail_dd.setObjectName("listWidget_availableFolders")
            avail_dd.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            # transfer any existing items
            for i in range(self.listWidget_availableFolders.count()):
                avail_dd.addItem(self.listWidget_availableFolders.item(i).text())

            # Replace at the exact slot index
            avail_layout.insertWidget(avail_index, avail_dd)
            old = avail_layout.itemAt(avail_index + 1).widget()  # the original now shifted right
            avail_layout.removeWidget(old)
            old.hide()
            old.deleteLater()
            self.listWidget_availableFolders = avail_dd

        # Replace selected list
        sel_layout, sel_index = self._find_container_layout_and_index(self.listWidget_selectedFolders)
        if sel_layout is None or sel_index < 0:
            print("Could not locate layout slot for listWidget_selectedFolders; aborting replacement.")
        else:
            sel_dd = DroppableListWidget(self.listWidget_selectedFolders.parent())
            sel_dd.setObjectName("listWidget_selectedFolders")
            sel_dd.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            for i in range(self.listWidget_selectedFolders.count()):
                sel_dd.addItem(self.listWidget_selectedFolders.item(i).text())

            sel_layout.insertWidget(sel_index, sel_dd)
            old = sel_layout.itemAt(sel_index + 1).widget()
            sel_layout.removeWidget(old)
            old.hide()
            old.deleteLater()
            self.listWidget_selectedFolders = sel_dd

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _init_labels_widget_reference(self) -> None:
        """
        Detect whether a label selection list exists (listWidget_labels).
        If not, we default to using comboBox_train_label_selection only.
        """
        if hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels, QtWidgets.QListWidget):
            self.categories_available = True
        else:
            self.categories_available = False

    # --------------------------------------------------------------------------------------------------------------
    # UI population and collection
    # --------------------------------------------------------------------------------------------------------------
    def _populate_ui_from_config(self, cfg: Dict[str, Any]) -> None:
        # Path for training images
        self.lineEdit_model_training_images_path.setText(cfg.get("segmentation_images_path", ""))

        # Model selection radio buttons
        model = cfg.get("load_model", {}).get("MODEL", "sam2").lower()
        self.radioButton_train_model_SAM2.setChecked(model == "sam2")
        self.radioButton_train_model_LoRA.setChecked(model == "segformer")
        self.radioButton_train_model_MaskRCNN.setChecked(model == "maskrcnn")

        # Labels/categories
        labels = cfg.get("load_model", {}).get("TRAINING_CATEGORIES", [])
        if hasattr(self, "comboBox_train_label_selection"):
            self.comboBox_train_label_selection.clear()
            for label in labels:
                self.comboBox_train_label_selection.addItem(str(label))
        if hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels, QtWidgets.QListWidget):
            self.listWidget_labels.clear()
            for label in labels:
                self.listWidget_labels.addItem(QtWidgets.QListWidgetItem(str(label)))
            self.categories_available = True if labels else False
        else:
            self.categories_available = True if labels else False

        # Training parameters
        self.lineEdit_siteName.setText(cfg.get("siteName", ""))
        self.lineEdit_learningRates.setText(",".join(str(x) for x in cfg.get("learningRates", [0.0001])))
        self.comboBox_optimizer.setCurrentText(cfg.get("optimizer", "Adam"))
        self.comboBox_lossFunction.setCurrentText(cfg.get("loss_function", "IOU"))
        self.doubleSpinBox_weightDecay.setValue(float(cfg.get("weight_decay", 0.01) or 0.01))
        self.spinBox_epochs.setValue(int(cfg.get("number_of_epochs", 20) or 20))
        self.spinBox_batchSize.setValue(int(cfg.get("batch_size", 32) or 32))
        self.spinBox_saveFrequency.setValue(int(cfg.get("save_model_frequency", 20) or 20))
        self.spinBox_validationFrequency.setValue(int(cfg.get("validation_frequency", 20) or 20))
        self.checkBox_earlyStopping.setChecked(bool(cfg.get("early_stopping", False)))
        self.spinBox_patience.setValue(int(cfg.get("patience", 3) or 3))
        self.comboBox_device.setCurrentText(cfg.get("device", "cpu"))

        # Folder lists
        self.listWidget_availableFolders.clear()
        for p in cfg.get("available_folders", []):
            self.listWidget_availableFolders.addItem(str(p))
        self.listWidget_selectedFolders.clear()
        for p in cfg.get("selected_folders", []):
            self.listWidget_selectedFolders.addItem(str(p))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _collect_ui_values(self) -> Dict[str, Any]:
        # Model selection
        if self.radioButton_train_model_SAM2.isChecked():
            model = "sam2"
        elif self.radioButton_train_model_LoRA.isChecked():
            model = "segformer"
        elif self.radioButton_train_model_MaskRCNN.isChecked():
            model = "maskrcnn"
        else:
            model = "sam2"

        # Learning rates parsing
        lr_text = self.lineEdit_learningRates.text().strip()
        learning_rates: List[float] = []
        if lr_text:
            for tok in lr_text.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        learning_rates.append(float(tok))
                    except ValueError:
                        pass

        # Selected labels
        selected_labels: List[str] = []
        if self.categories_available and hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels,
                                                                                           QtWidgets.QListWidget):
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.isSelected():
                    selected_labels.append(item.text())

        values: Dict[str, Any] = {
            "siteName": self.lineEdit_siteName.text().strip(),
            "learningRates": learning_rates or [0.0001],
            "optimizer": self.comboBox_optimizer.currentText(),
            "loss_function": self.comboBox_lossFunction.currentText(),
            "weight_decay": float(self.doubleSpinBox_weightDecay.value()),
            "number_of_epochs": int(self.spinBox_epochs.value()),
            "batch_size": int(self.spinBox_batchSize.value()),
            "save_model_frequency": int(self.spinBox_saveFrequency.value()),
            "validation_frequency": int(self.spinBox_validationFrequency.value()),
            "early_stopping": bool(self.checkBox_earlyStopping.isChecked()),
            "patience": int(self.spinBox_patience.value()),
            "device": self.comboBox_device.currentText(),
            "segmentation_images_path": self.lineEdit_model_training_images_path.text().strip(),
            "available_folders": [self.listWidget_availableFolders.item(i).text()
                                  for i in range(self.listWidget_availableFolders.count())],
            "selected_folders": [self.listWidget_selectedFolders.item(i).text()
                                 for i in range(self.listWidget_selectedFolders.count())],
            "load_model": {
                **self.site_config.get("load_model", {}),
                "MODEL": model,
                "SELECTED_TRAINING_LABELS": selected_labels if selected_labels
                else self.site_config.get("load_model", {}).get("SELECTED_TRAINING_LABELS", []),
            }
        }

        # 🔑 NEW: update Path section based on selected folders
        root_folder = self.lineEdit_model_training_images_path.text().strip()
        if root_folder:
            self.update_path_from_selection(root_folder, values)

        return values

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def update_path_from_selection(self, root_folder: str, values: Dict[str, Any]):
        """
        Collect selected folders from listWidget_selectedFolders and update values["Path"]
        with image and annotation paths.
        """
        # 1. Gather selected folder names
        selected_folders = [
            self.listWidget_selectedFolders.item(i).text()
            for i in range(self.listWidget_selectedFolders.count())
        ]

        new_folders = []
        new_annotations = []

        # 2. Build normalized paths
        # JES: no longer using CVAT's layered convention
        for folder in selected_folders:
            folder = os.path.normpath(os.path.join(root_folder, folder))
            new_folders.append(folder)

            filepath = os.path.normpath(os.path.join(root_folder, folder, "instances_default.json"))
            new_annotations.append(filepath)

        # 3. Update values["Path"]
        values["Path"] = [{
            "siteName": "custom",
            "directoryPaths": {
                "folders": new_folders,
                "annotations": new_annotations
            }
        }]

        print("Updated Path section from selected folders.")

    # --------------------------------------------------------------------------------------------------------------
    # Actions and slots
    # --------------------------------------------------------------------------------------------------------------
    def get_selected_training_model(self):
        return self.selected_training_model

    def update_model_config(self) -> None:
        values = self._collect_ui_values()
        self._mgr.update_config(values, save=True)
        self.site_config = self._mgr.load_config(return_type="dict")

    def train(self):
        """
        Called when the Train button is clicked.
        Validates model path, images folder, label selections, and image presence.
        """
        # Verify that the user provided a site name for training
        site_name = self.lineEdit_siteName.text().strip()
        if not site_name:
            msg = "You must specify a site name before training can begin."
            msgBox = GRIME_AI_QMessageBox('Missing Site Name', msg, GRIME_AI_QMessageBox.Ok, icon=QMessageBox.Warning)
            msgBox.displayMsgBox()
            return  # stop further execution until user provides a site name

        # Label validation if categories were expected
        '''
        if self.categories_available:
            selected_labels = []
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.isSelected():
                    selected_labels.append(item.text())
        '''

        # WRITE SETTINGS TO THE SITE_CONFIG.JSON
        self.update_model_config()

        print("\nTrain button clicked. Starting training process...")
        self.ml_train_signal.emit()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def reset_selection(self):
        self.listWidget_selectedFolders.clear()
        self.listWidget_availableFolders.clear()
        self.transferred_items.clear()
        self.annotation_list = []
        self.unique_training_labels = []
        self.populate_train_label_combobox([])
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_to_selected(self):
        for item in self.listWidget_availableFolders.selectedItems():
            self.listWidget_selectedFolders.addItem(item.text())
            self.transferred_items.add(item.text())
            self.listWidget_availableFolders.takeItem(self.listWidget_availableFolders.row(item))
        self.listWidget_selectedFolders.repaint()
        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_to_right(self):
        selected_items = self.listWidget_availableFolders.selectedItems()
        self._move_items(selected_items)
        for item in selected_items:
            print(f"Moved '{item.text()}' from available to selected folders (button).")

        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_to_available(self):
        for item in self.listWidget_selectedFolders.selectedItems():
            text = item.text()
            self.listWidget_availableFolders.addItem(text)
            self.transferred_items.discard(text)
            self.listWidget_selectedFolders.takeItem(self.listWidget_selectedFolders.row(item))
        self.listWidget_availableFolders.repaint()
        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # --------------------------------------------------------------------------------------------------------------
    # Folder population and annotation aggregation
    # --------------------------------------------------------------------------------------------------------------
    def populate_available_folders(self):
        root = Path(self.lineEdit_model_training_images_path.text().strip()).resolve()
        self.listWidget_availableFolders.clear()

        if not root.is_dir():
            QMessageBox.warning(self, "Invalid Folder", f"The selected path is not a directory:\n{root}")
            return

        valid: List[Path] = []
        incomplete: Dict[str, List[str]] = {}

        # Check the root itself
        ok, missing = _check_folder(root)
        if ok:
            valid.append(root)
        elif missing:
            incomplete[str(root)] = missing

        # Recurse into subfolders
        for folder in _iter_dirs(root):
            ok, missing = _check_folder(folder)
            if ok:
                valid.append(folder)
            elif missing:
                incomplete[str(folder)] = missing

        # Populate or alert “no valid”
        if valid:
            for vf in sorted(set(valid)):
                rel = vf.relative_to(root)
                display_name = str(rel)
                self.listWidget_availableFolders.addItem(display_name)
        else:
            QMessageBox.information(
                self,
                "No Valid Training Sets",
                "No folders were found containing a COCO JSON and all its images."
            )

        # Incomplete sets popup
        if incomplete:
            lines = ["Folders missing files:"]
            for fld, miss in incomplete.items():
                lines.append(f"\n{fld}\n  Missing:")
                lines += [f"    • {m}" for m in miss]
            QMessageBox.information(self, "Incomplete Training Sets", "\n".join(lines))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _move_items(self, items):
        for item in items:
            name = item.text()
            if name not in self.transferred_items:
                self.listWidget_selectedFolders.addItem(name)
                self.transferred_items.add(name)

            row = self.listWidget_availableFolders.row(item)

            if row != -1:
                self.listWidget_availableFolders.takeItem(row)

        self._refresh_annotations_from_selection()

        self.updateTrainButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_items_to_selected(self, items):
        self._move_items(items)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _refresh_annotations_from_selection(self):
        # ENSURE THE SELECTED LIST VISUALLY UPDATES BEFORE WE COMPUTE LABELS
        self.listWidget_selectedFolders.repaint()

        # BASE PATH (ROOT FOLDER OF TRAINING IMAGES)
        base_path = self.lineEdit_model_training_images_path.text().strip()

        # CURRENT SELECTED FOLDERS (RELATIVE NAMES)
        moved_names = [self.listWidget_selectedFolders.item(i).text()
                       for i in range(self.listWidget_selectedFolders.count())]

        # BUILD ANNOTATION LIST FROM SELECTED FOLDERS
        self.annotation_list = self._build_annotation_list(base_path, moved_names)

        # COLLECT UNIQUE LABELS AND REPOPULATE COMBOBOX
        self.unique_training_labels = self.collect_unique_labels(self.annotation_list)
        self.populate_train_label_combobox(self.unique_training_labels)

    # --------------------------------------------------------------------------------------------------------------
    # Annotation helpers
    # --------------------------------------------------------------------------------------------------------------
    def _build_annotation_list(self, base_path: str, folder_names: list[str]) -> list[str]:
        root_folder = os.path.normpath(os.path.abspath(base_path))
        annotations = []

        for folder in folder_names:
            folder_path = os.path.normpath(os.path.join(root_folder, folder))
            annotation_file_path = os.path.normpath(folder_path)
            if os.path.exists(annotation_file_path):
                annotations.append(annotation_file_path)

        return annotations
        self.populate_train_label_combobox(self.unique_training_labels)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_labels_from_annotation(self, folder_path):
        annotation_file = os.path.join(folder_path, "instances_default.json")
        if not os.path.exists(annotation_file):
            return []

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = set()
        if "annotations" in data and "categories" in data:
            for cat in data["categories"]:
                labels.add(f"{cat['id']} - {cat['name']}")
        return sorted(labels)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def collect_unique_labels(self, annotation_files: list[str]) -> list[str]:
        """
        Collects all unique category labels from a list of annotation files.

        Parameters
        ----------
        annotation_files : list[str]
            Fully qualified paths to instances_default.json files.

        Returns
        -------
        list[str]
            Sorted list of unique labels in the format 'id - name'.
        """
        all_labels = set()

        for path in annotation_files:
            labels = self.load_labels_from_annotation(path)
            all_labels.update(labels)

        return sorted(all_labels)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_train_label_combobox(self, labels: list[str]):
        self.comboBox_train_label_selection.clear()
        for label in sorted(labels):
            self.comboBox_train_label_selection.addItem(label)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def updateTrainButtonState(self):
        """
        Enable Train button only when there is a site name, selected folders, and (optionally) labels.
        """
        has_site = bool(self.lineEdit_siteName.text().strip())
        has_selected = self.listWidget_selectedFolders.count() > 0

        labels_ok = True
        if self.categories_available and hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels, QtWidgets.QListWidget):
            labels_ok = any(self.listWidget_labels.item(i).isSelected() for i in range(self.listWidget_labels.count())) \
                        or self.listWidget_labels.count() == 0

        self.pushButton_train.setEnabled(has_site and has_selected and labels_ok)

    # ******************************************************************************************************************
    # *   OTHER    OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     *
    # ******************************************************************************************************************
    def save_site_name_to_json(self):
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        CONFIG_FILENAME = "site_config.json"
        config_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))

        settings = JsonEditor().load_json_file(config_file)

        # Update with the current SiteName value
        settings["siteName"] = self.lineEdit_siteName.text()

        # Write back to the config file
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)

        print(f"Updated siteName in {config_file} to '{settings['siteName']}'")

