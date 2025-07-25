from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_QMessageBox(QMessageBox):

    def __init__(self, strTitle, strMsg, buttons=QMessageBox.Ok, icon=QMessageBox.Information):
        QMessageBox.__init__(self)

        self.strTitle = strTitle
        self.strMsg = strMsg
        self.setIcon(icon)
        self.setStandardButtons(buttons)

        self.setWindowTitle(self.strTitle)
        self.setText(strMsg)


    def displayMsgBox(self, on_top=True):
        if on_top:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True);
        else:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False);

        response = self.exec_()

        return response
