from PyQt5.QtWidgets import QVBoxLayout, QMessageBox

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_QMessageBox(QMessageBox):

    def __init__(self, strTitle, strMsg, buttons=QMessageBox.Ok):
        QMessageBox.__init__(self)

        layout = QVBoxLayout(self)

        self.strTitle = strTitle
        self.strMsg = strMsg
        self.setIcon(QMessageBox.Information)
        self.setStandardButtons(buttons)

        self.setWindowTitle(self.strTitle)
        self.setText(strMsg)

    def displayMsgBox(self):
        response = self.exec_()

        return response
