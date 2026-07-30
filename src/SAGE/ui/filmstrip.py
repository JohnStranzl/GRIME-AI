# SAGE/ui/filmstrip.py
"""
Horizontal thumbnail filmstrip beneath the image canvas.

Ported from the GRIME AI Image Navigator pattern: single-row, no wrapping,
batched/non-blocking thumbnail loading. Emits image_clicked(filename) on
selection so MainWindow can build the full path and load it.
"""
import os
from PyQt5.QtCore import Qt, QTimer, QSize, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QImageReader
from PyQt5.QtWidgets import (
    QListWidget, QListWidgetItem, QListView, QFrame, QStyledItemDelegate,
)

# Per-item data roles beyond Qt.UserRole (which holds the filename).
ROLE_NAME = Qt.UserRole
ROLE_ANNOTATED = Qt.UserRole + 1     # bool: has >=1 saved annotation
ROLE_DIMS = Qt.UserRole + 2          # (w, h) or None until known


class _DotDelegate(QStyledItemDelegate):
    """Draws the thumbnail normally, then a small filled dot centered beneath
    it when the item is flagged annotated. Painting in the delegate (rather than
    compositing onto the icon pixmap) keeps the dot independent of the thumbnail
    - it survives selection highlight and needs no icon regeneration when the
    annotation state flips."""

    DOT_D = 7        # diameter in px
    DOT_PAD = 3      # gap between thumbnail bottom and dot

    def __init__(self, color=QColor(46, 204, 113), parent=None):
        super().__init__(parent)
        self._color = color

    def set_color(self, color):
        self._color = color

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if not index.data(ROLE_ANNOTATED):
            return
        r = option.rect
        cx = r.center().x()
        cy = r.bottom() - self.DOT_PAD - self.DOT_D / 2.0
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(
            QRectF(cx - self.DOT_D / 2.0, cy - self.DOT_D / 2.0,
                   self.DOT_D, self.DOT_D)
        )
        painter.restore()


class Filmstrip(QListWidget):
    image_clicked = pyqtSignal(str)   # emits filename (not full path)

    def __init__(self, parent=None, thumb_height=72, batch_size=12, batch_delay=40):
        super().__init__(parent)

        self._folder = ""
        self._names = []
        self._pending = []
        self._batch_size = batch_size
        self._batch_delay = batch_delay
        self._load_token = 0

        icon_w = int(thumb_height * 16 / 9)
        self.setIconSize(QSize(icon_w, thumb_height))
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setSpacing(2)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Dot painted beneath each thumbnail for annotated images.
        self._dot_delegate = _DotDelegate(parent=self)
        self.setItemDelegate(self._dot_delegate)
        self._dot_band = _DotDelegate.DOT_PAD + _DotDelegate.DOT_D + 2

        # Tooltips are built per-item on hover (see event()), so mouse tracking
        # must be on even when no button is held.
        self.setMouseTracking(True)

        # One icon row tall (+ dot band + scrollbar allowance).
        self.setFixedHeight(thumb_height + 26 + self._dot_band)

        self.itemClicked.connect(self._on_item_clicked)

    # -------------------------------------------------------------
    def populate(self, folder, names, annotated=None):
        """names: list of filenames (not full paths) in `folder`.
        annotated: optional set/container of filenames that have saved
        annotations; those get a dot. Absent -> no dots (caller can set them
        later via set_annotated / set_annotated_set)."""
        self._folder = folder
        self._names = list(names)
        annotated = annotated or set()
        self.clear()

        self._load_token += 1
        token = self._load_token
        self._pending.clear()

        band = self._dot_band
        for idx, name in enumerate(names):
            item = QListWidgetItem(QIcon(), "")
            item.setData(ROLE_NAME, name)
            item.setData(ROLE_ANNOTATED, name in annotated)
            item.setData(ROLE_DIMS, None)
            # Tooltip is composed on hover in event(); nothing static here.
            item.setSizeHint(QSize(self.iconSize().width() + 6,
                                   self.iconSize().height() + 6 + band))
            self.addItem(item)
            self._pending.append((item, os.path.join(folder, name), token))

        if self.count():
            self.setCurrentRow(0)
        QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def select_name(self, name):
        """Highlight the item for `name` without emitting a click."""
        for i in range(self.count()):
            if self.item(i).data(ROLE_NAME) == name:
                self.blockSignals(True)
                self.setCurrentRow(i)
                self.blockSignals(False)
                self.scrollToItem(self.item(i))
                return

    # -------------------------------------------------------------
    def _load_batch(self, token):
        if token != self._load_token:
            return
        for _ in range(min(self._batch_size, len(self._pending))):
            item, path, tok = self._pending.pop(0)
            if tok != self._load_token or not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            # The full-res pixmap is already decoded here; record its true
            # dimensions for the hover tooltip before scaling it down, so hover
            # never has to reopen the file.
            item.setData(ROLE_DIMS, (pix.width(), pix.height()))
            item.setIcon(QIcon(pix.scaled(self.iconSize(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)))
        if self._pending:
            QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def _on_item_clicked(self, item):
        name = item.data(ROLE_NAME)
        if name:
            self.image_clicked.emit(name)

    # -------------------------------------------------------------
    # Annotation-dot API
    # -------------------------------------------------------------
    def set_annotated(self, name, flag=True):
        """Flip the dot for a single image and repaint just its cell."""
        for i in range(self.count()):
            it = self.item(i)
            if it.data(ROLE_NAME) == name:
                if bool(it.data(ROLE_ANNOTATED)) != bool(flag):
                    it.setData(ROLE_ANNOTATED, bool(flag))
                    self.update(self.indexFromItem(it))
                return

    def set_annotated_set(self, names):
        """Bulk-set which images show a dot. `names` is any container of
        filenames; everything not in it is cleared."""
        wanted = set(names or ())
        for i in range(self.count()):
            it = self.item(i)
            flag = it.data(ROLE_NAME) in wanted
            if bool(it.data(ROLE_ANNOTATED)) != flag:
                it.setData(ROLE_ANNOTATED, flag)
        self.viewport().update()

    def set_dot_color(self, color):
        self._dot_delegate.set_color(color)
        self.viewport().update()

    # -------------------------------------------------------------
    # Hover tooltip: image name + pixel dimensions, composed on demand.
    # -------------------------------------------------------------
    def _tooltip_for(self, item):
        name = item.data(ROLE_NAME) or ""
        dims = item.data(ROLE_DIMS)
        if dims is None:
            # Thumbnail not decoded yet: read only the header, no pixel decode.
            dims = self._read_dims(os.path.join(self._folder, name))
            if dims is not None:
                item.setData(ROLE_DIMS, dims)
        if dims is None:
            return name
        w, h = dims
        annotated = " • annotated" if item.data(ROLE_ANNOTATED) else ""
        return f"{name}\n{w} × {h} px{annotated}"

    @staticmethod
    def _read_dims(path):
        """Dimensions from the image header without decoding pixels."""
        if not os.path.exists(path):
            return None
        reader = QImageReader(path)
        size = reader.size()
        if size.isValid():
            return (size.width(), size.height())
        return None

    def event(self, e):
        from PyQt5.QtCore import QEvent
        if e.type() == QEvent.ToolTip:
            item = self.itemAt(e.pos())
            if item is not None:
                from PyQt5.QtWidgets import QToolTip
                QToolTip.showText(e.globalPos(), self._tooltip_for(item), self)
            else:
                from PyQt5.QtWidgets import QToolTip
                QToolTip.hideText()
            return True
        return super().event(e)
