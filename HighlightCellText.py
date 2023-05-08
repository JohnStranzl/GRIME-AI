class HighlightDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HighlightDelegate, self).__init__(parent)
        self.doc = QtGui.QTextDocument(self)
        self._filters = []

    def paint(self, painter, option, index):
        painter.save()
        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        self.doc.setPlainText(options.text)
        self.apply_highlight()
        options.text = ""
        style = QtWidgets.QApplication.style() if options.widget is None \
            else options.widget.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        if option.state & QtWidgets.QStyle.State_Selected:
            ctx.palette.setColor(QtGui.QPalette.Text, option.palette.color(
                QtGui.QPalette.Active, QtGui.QPalette.HighlightedText))
        else:
            ctx.palette.setColor(QtGui.QPalette.Text, option.palette.color(
                QtGui.QPalette.Active, QtGui.QPalette.Text))

        textRect = style.subElementRect(
            QtWidgets.QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        the_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - the_constant
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def apply_highlight(self):
        cursor = QtGui.QTextCursor(self.doc)
        cursor.beginEditBlock()
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtCore.Qt.red)
        for f in self.filters():
            highlightCursor = QtGui.QTextCursor(self.doc)
            while not highlightCursor.isNull() and not highlightCursor.atEnd():
                highlightCursor = self.doc.find(f, highlightCursor)
                if not highlightCursor.isNull():
                    highlightCursor.mergeCharFormat(fmt)
        cursor.endEditBlock()

    @QtCore.pyqtSlot(list)
    def setFilters(self, filters):
        if self._filters == filters: return
        self._filters = filters

    def filters(self):
        return self._filters

class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        self.table = QtWidgets.QTableWidget(30, 6)
        self._delegate = HighlightDelegate(self.table)
        self.table.setItemDelegate(self._delegate)
        for i in range(self.table.rowCount()):
            for j in range(self.table.columnCount()):
                it = QtWidgets.QTableWidgetItem(random.choice(words))
                self.table.setItem(i, j, it)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        le = QtWidgets.QLineEdit()
        le.textChanged.connect(self.on_textChanged)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(le)
        lay.addWidget(self.table)

        le.setText("ello ack")

    @QtCore.pyqtSlot(str)
    def on_textChanged(self, text):
        self._delegate.setFilters(list(set(text.split())))
        self.table.viewport().update()

