from PyQt5 import QtGui
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QTableWidget, QCheckBox

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_QTableWidget(QTableWidget):

    def __init__(self, parent=None):
        QTableWidget.__init__(self, parent)
        self.chkbox1 = QCheckBox(self.horizontalHeader())

    def resizeEvent(self, event=None):
        super().resizeEvent(event)
        #self.chkbox1.setGeometry(QtCore.QRect((self.columnWidth(0)/2), 2, 16, 17))
        self.chkbox1.setGeometry(QtCore.QRect(0, 2, 16, 17))
