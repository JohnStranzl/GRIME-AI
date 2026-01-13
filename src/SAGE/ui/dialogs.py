# sam2_gui/ui/dialogs.py
from PyQt5.QtWidgets import QInputDialog, QWidget


def ask_for_label(parent: QWidget, default_label: str):
    text, ok = QInputDialog.getText(parent, "Region Label", "Label:", text=default_label)
    if ok and text.strip():
        return text.strip()
    return default_label
