# main.py
import multiprocessing as mp
mp.freeze_support()
#mp.set_start_method('spawn', force=True)  # ← ÇA DOIT ÊTRE AVANT PyQt
mp.set_start_method('forkserver', force=True)  # ← alternative propre aussi

from PyQt6.QtWidgets import QApplication
from gui.main_window import PmxRkoMainWindow

app = QApplication(["PmxRkoTrading"])
win = PmxRkoMainWindow()
win.show()
app.exec()
