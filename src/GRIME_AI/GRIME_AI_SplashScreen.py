#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5.QtWidgets import QSplashScreen
from threading import Thread
import time
from PyQt5.QtGui import QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt

import cv2


# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_SplashScreen(QSplashScreen):

    def __init__(self, pixmap, strVersion='', delay=2):
        """
        The constructor for GRIME_AI_SplashScreen class.

        Parameters:
            pixmap (QPixmap): The QPixmap object to be displayed as a splash screen.
            strVersion (str, optional): The version string to be displayed on the splash screen. Defaults to ''.
            delay (int, optional): The delay time in seconds before the splash screen disappears. Defaults to 2.
        """

        QSplashScreen.__init__(self)

        self.pixmap = pixmap

        if len(strVersion) > 0:
            # CREATE A QPAINTER INSTANCE AND PASS IN THE QPIXMAP TO BEGIN PAINTING ONTO IT
            painter = QPainter(self.pixmap)

            # SELECT FONT AND FONT COLOR
            painter.setFont(QFont('Arial', 8))
            painter.setPen(Qt.black)

            # ADD THE TEXT TO THE IMAGE
            painter.drawText(475, 220, strVersion)

            painter.end()

        self.delay = delay


    def show(self, mainWin):
        Thread(target=self.splashImage(mainWin)).start()


    def splashImage(self, mainWin):
        bigSplash = QSplashScreen(self.pixmap)
        bigSplash.show()
        time.sleep(self.delay)

        bigSplash.finish(mainWin)


    '''
    def spiralImage(self, mainWin):
        self.angle += 1  # Adjust the rotation angle
        rows, cols, _ = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        rotated_image = cv2.warpAffine(self.image, M, (cols, rows))
        q_image = QImage(rotated_image.data, cols, rows, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
    '''

'''
class ImageSpinner(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.angle = 0
 
        self.setCentralWidget(self.image_label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate_image)
        self.timer.start(30)  # Adjust the rotation speed

    def rotate_image(self):
'''