from PyQt5.QtWidgets import QSplashScreen
from threading import Thread
from playsound import playsound
import time

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_SplashScreen(QSplashScreen):

    def __init__(self, pixmap, delay=2):
        QSplashScreen.__init__(self)

        self.pixmap = pixmap
        self.delay = delay

    def show(self, mainWin):
        #Thread(target=self.splashSound()).start()
        Thread(target=self.splashImage(mainWin)).start()

    def splashImage(self, mainWin):
        bigSplash = QSplashScreen(self.pixmap)
        bigSplash.show()
        time.sleep(self.delay)

        #self.splashSound()
        bigSplash.finish(mainWin)

    # ------------------------------------------------------------------------------------------------------------------
    # ARTISTIC LICENSE
    # ------------------------------------------------------------------------------------------------------------------
    def splashSound(self):
        try:
           playsound('.\shall-we-play-a-game.mp3')
        except:
           pass
