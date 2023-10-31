from PyQt5.QtWidgets import QSplashScreen
from threading import Thread
from playsound import playsound
import time

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_SplashScreen(QSplashScreen):

    def __init__(self, pixmap):
        QSplashScreen.__init__(self)

        self.pixmap = pixmap

    def show(self, mainWin):
        #Thread(target=self.splashSound()).start()
        Thread(target=self.splashImage(mainWin)).start()

    def splashImage(self, mainWin):
        bigSplash = QSplashScreen(self.pixmap)
        bigSplash.show()
        time.sleep(2)
        self.splashSound()
        bigSplash.finish(mainWin)

    def splashSound(self):
        try:
           playsound('.\shall-we-play-a-game.mp3')
        except:
           pass
