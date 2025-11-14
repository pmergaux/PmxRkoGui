# main.py
import json
import os
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import PmxRkoMainWindow

# ===================== application ===================
app = QApplication(sys.argv)
window = PmxRkoMainWindow()
window.show()
sys.exit(app.exec_())
