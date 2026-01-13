from PyQt5.QtWidgets import QGraphicsPolygonItem
from PyQt5.QtGui import QBrush, QPen, QColor, QPolygonF
from PyQt5.QtCore import Qt, QPointF


class MaskItem(QGraphicsPolygonItem):
    """
    Lightweight invisible polygon used only for hit-testing masks.
    """

    def __init__(self, polygon_points, mask_id):
        super().__init__()

        self.mask_id = mask_id

        # Build polygon
        qpoints = [QPointF(x, y) for x, y in polygon_points]
        poly = QPolygonF(qpoints)
        self.setPolygon(poly)

        # CRITICAL: Almost transparent but NOT fully transparent (alpha=1)
        # Fully transparent items (alpha=0) don't get hit-tested!
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))

        # No visible pen
        pen = QPen(Qt.NoPen)
        self.setPen(pen)

        # Disable selection to prevent dotted rectangle
        self.setFlag(self.ItemIsSelectable, False)

        # Disable hover events
        self.setAcceptHoverEvents(False)

        # CRITICAL: High z-value to ensure mask items are above the pixmap
        self.setZValue(100)