# main.py
from PyQt6.QtWidgets import QMainWindow, QApplication
from gui.main_window import PmxRkoMainWindow

app = QApplication(["PmxRkoTrading"])
win = PmxRkoMainWindow()
win.show()
app.exec()
