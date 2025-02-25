import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from src.ui.initial_page import Ui_MainWindow
from mtpSPECTWindow import DemoWindow as mtpSPECT

class DemoWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DemoWindow, self).__init__()
        self.setupUi(self)
        self.setFixedSize(1000, 570)  # Set the fixed size of the window

        self.show()  # Show the GUI

        # Load the pixmap
        self.pixmap = QPixmap("./big_dose_logo.png")
        self.label.setPixmap(self.pixmap)
        self.label.setFixedSize(480,420)

        # Set size policy to prevent resizing
        self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.mtpSPECT.clicked.connect(self.mtpSPECTWindow)
        # self.mtpPlanar.clicked.connect(self.mtpPlanarWindow)
        # self.StpSPECT_Planar.clicked.connect(self.StpSPECT_PlanarWindow)
        # self.PET_pred.clicked.connect(self.PET_PredWindow)
        self.aboutUsBtn.clicked.connect(self.AboutBIGDOSEWindowOpen)
        self.settingBtn.clicked.connect(self.settingWindowOpen)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def mtpSPECTWindow(self):
        self.ui = mtpSPECT()
        self.ui.show()

    # Y-90 Post-therapy Analysis Window Open
    # def mtpPlanarWindow(self):
    #     self.ui = BaseLayout()
    #     self.ui.show()
    #     print("y90")

    # def StpSPECT_PlanarWindow(self):
    #     self.ui = BaseLayout()
    #     self.ui.show()
    #     print("y90")
    #
    # def PET_PredWindow(self):
    #     self.ui = BaseLayout()
    #     self.ui.show()
    #     print("y90")

    def AboutBIGDOSEWindowOpen(self):
        print("About BIGDOSE")

    # Setting Window Open
    def settingWindowOpen(self):
        print("Setting")

def main():
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建 DemoWindow 实例
    window = DemoWindow()
    # 进入应用程序主循环
    app.exec()

if __name__ == '__main__':
    main()

