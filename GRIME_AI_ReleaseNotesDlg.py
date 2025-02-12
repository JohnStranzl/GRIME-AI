from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

userCancelled = "Cancelled"
userOk = "OK"

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_ReleaseNotesDlg(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi('QDialog_ReleaseNotes.ui', self)
