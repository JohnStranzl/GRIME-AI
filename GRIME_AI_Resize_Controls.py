# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_Resize_Controls:
    def __init__(self):
        self.className = "GRIME_AI_Resize_Controls"
        self.instance = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 0 - NEON SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_0(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidth = (int)((windowSize.width() - (gap * (3 + 1)) - (2 * border)) / 3)

        one = selfControl.NEON_listboxSites.pos()

        # PANEL 1 - SITES
        left = one.x()
        top = one.y()
        width = newWidth
        height = windowSize.height() - vertical_margin * 2
        selfControl.NEON_listboxSites.setGeometry(left, top, width, height)

        # PANEL 2A - SITE INFO
        left = left + newWidth + gap
        top = one.y()
        width = newWidth
        height = (int)(windowSize.height() / 2)
        selfControl.NEON_listboxSiteInfo.setGeometry(left, top, width, height)

        # PANEL 2B - SITE IMAGE
        top = one.y() + (int)(selfControl.NEON_listboxSites.height() / 2)
        width = newWidth
        height = (int)(windowSize.height() / 2) - vertical_margin
        selfControl.NEON_labelLatestImage.setGeometry(left, top, width, height)
        # self.NEON_labelLatestImage.setGeometry(QtCore.QRect(one.x()+newWidth+border, one.y(), newWidth, windowSize.height()))        # setGeometry(left, top, width, height)

        # PANEL 3 - SITE PRODUCTS
        left = left + newWidth + gap
        top = one.y()
        width = newWidth - border
        height = windowSize.height()
        selfControl.NEON_listboxSiteProducts.setGeometry(left, top, width, height)

        # TAB WIDGET - RESIZE THE TAB WIDGET
        left = one.x() + border
        top = one.y() - (int)(vertical_margin / 2)
        width = windowSize.width() - (2 * border)
        height = windowSize.height() - vertical_margin
        selfControl.tabWidget.setGeometry(left, top, width, height)

        # QtWidgets.resizeEvent(self, event)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 1 - NEON DOWNLOAD MANAGER
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_1(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidth = (int)(windowSize.width() - gap - (2 * border))
        newHeight = (int)(windowSize.height() * 0.50)

        topLeftCorner = selfControl.NEON_tableProducts.pos()

        # SPREADSHEET
        left = topLeftCorner.x()
        top  = topLeftCorner.y()
        width  = newWidth
        height = newHeight
        selfControl.NEON_tableProducts.setGeometry(left, top, width, height)

        # DOWNLOAD GROUPBOX
        left = left
        top  = top + newHeight + gap
        width  = selfControl.groupBox_NEONDownloadManager_OutputFolders.width()
        height = selfControl.groupBox_NEONDownloadManager_OutputFolders.height()
        selfControl.groupBox_NEONDownloadManager_OutputFolders.setGeometry(left, top, width, height)

        # DOWNLOAD BUTTON
        #DOWNLOAD GROUPBOX VERTICAL CENTER
        vertCenter = top + (int)(height / 2.0)

        width = selfControl.pushButton_RetrieveNEONData.width()
        height = selfControl.pushButton_RetrieveNEONData.height()
        left = left + selfControl.groupBox_NEONDownloadManager_OutputFolders.width() + gap
        top  = vertCenter - (int)(height / 2.0)
        selfControl.pushButton_RetrieveNEONData.setGeometry(left, top, width, height)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 2 - USGS SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_2(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidth = (int)((windowSize.width() - (gap * (2 + 1)) - (2 * border)) / 3)

        # ------------------------------------------------------------
        # TAB WIDGET - RESIZE THE TAB WIDGET
        # ------------------------------------------------------------
        left = selfControl.tabWidget.x()
        left = gap
        top = selfControl.tabWidget.y()
        top = 0
        width = (int)(windowSize.width() - (2.0 * gap))
        height = windowSize.height()
        selfControl.tabWidget.setGeometry(left, top, width, height)

        # ------------------------------------------------------------
        # Left 30%
        # ------------------------------------------------------------
        left   = gap
        top    = selfControl.USGS_listboxSites.pos().y()
        top    = gap
        width  = (int)(windowSize.width() * 0.30 - (2.0 * gap))
        height = (int)(windowSize.height() - 150)
        selfControl.USGS_listboxSites.setGeometry(left, top, width, height)

        # ------------------------------------------------------------
        # Right Top-half 70%
        # ------------------------------------------------------------
        left   = (int)(left + (windowSize.width() * 0.30) - gap)
        top = gap
        width  = (int)(windowSize.width() * 0.70 - (3.5 * gap))
        height = (int)(windowSize.height() / 2.5)
        selfControl.listboxUSGSSiteInfo.setGeometry(left, top, width, height)

        # ------------------------------------------------------------
        # Right Bottom-half 70%
        # ------------------------------------------------------------
        left   = left
        top    = (int)(top + windowSize.height() / 2.5 + gap)
        width  = (int)(windowSize.width() * 0.70 - (3.5 * gap))
        height = (int)(windowSize.height() / 2.5 + gap)
        selfControl.USGS_labelLatestImage.setGeometry(left, top, width, height)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 1 - NEON DOWNLOAD MANAGER
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_3(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidth = (int)(windowSize.width() - gap - (2 * border))
        newHeight = (int)(windowSize.height() * 0.50)

        topLeftCorner = selfControl.table_USGS_Sites.pos()

        # SPREADSHEET
        left = topLeftCorner.x()
        top  = topLeftCorner.y()
        width  = newWidth
        height = newHeight
        selfControl.table_USGS_Sites.setGeometry(left, top, width, height)

        # DOWNLOAD GROUPBOX
        left = left
        top  = top + newHeight + gap
        width  = selfControl.groupBox_USGSDownloadManager_OutputFolders.width()
        height = selfControl.groupBox_USGSDownloadManager_OutputFolders.height()
        selfControl.groupBox_USGSDownloadManager_OutputFolders.setGeometry(left, top, width, height)

        # DOWNLOAD BUTTON
        #DOWNLOAD GROUPBOX VERTICAL CENTER
        vertCenter = top + (int)(height / 2.0)

        width = selfControl.pushButton_USGSDownload.width()
        height = selfControl.pushButton_USGSDownload.height()
        left = left + selfControl.groupBox_USGSDownloadManager_OutputFolders.width() + gap
        top  = vertCenter - (int)(height / 2.0)
        selfControl.pushButton_USGSDownload.setGeometry(left, top, width, height)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 4 - NEON SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_4(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidthAvailable = (int)(windowSize.width() - (gap * (2 + 1)) - (2*border))

        one   = selfControl.labelOriginalImage.pos()
        paneWidth  = (int)(newWidthAvailable * 0.50)

        # --------------------------------------------------------------------------------
        # TAB WIDGET - RESIZE THE TAB WIDGET
        # --------------------------------------------------------------------------------
        left = selfControl.tabWidget.x()
        left = gap
        top = selfControl.tabWidget.y()
        top = 0
        width = (int)(windowSize.width() - (2.0 * gap))
        height = windowSize.height()
        selfControl.tabWidget.setGeometry(left, top, width, height)

        # --------------------------------------------------------------------------------
        # ROW 1
        # --------------------------------------------------------------------------------
        # ROW 1, COL 1 - ORIGINAL IMAGE
        left   = one.x()
        top    = one.y()
        width  = paneWidth
        height = (int)((windowSize.height() / 2.25) - vertical_margin)
        selfControl.labelOriginalImage.setGeometry(left, top, width, height)

        # ROW 1, COL 2 - EDGE IMAGE
        left   = left + paneWidth + gap
        selfControl.labelEdgeImage.setGeometry(left, top, width, height)

        # --------------------------------------------------------------------------------
        # ROW 2
        # --------------------------------------------------------------------------------
        # ROW 2, COL 1 - MASK IMAGE
        left   = one.x()
        top    = (int)(one.y() + gap + (windowSize.height() / 2.25) - vertical_margin)
        width  = paneWidth
        height = (int)((windowSize.height() / 2.25) - vertical_margin)
        selfControl.labelColorSegmentation.setGeometry(left, top, width, height)

        # ROW 2, COL 2 - ROI LIST
        left   = left + paneWidth + gap
        selfControl.tableWidget_ROIList.setGeometry(left, top, width, height)
