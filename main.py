# main.py
import multiprocessing as mp
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import PmxRkoMainWindow

def create_app():
    """Factory to create the application and main window."""
    app = QApplication(sys.argv if sys.argv else ["PmxRkoTrading"])
    window = PmxRkoMainWindow()
    return app, window

if __name__ == "__main__":
    mp.freeze_support()
    # mp.set_start_method('spawn', force=True) 
    mp.set_start_method('forkserver', force=True)

    app, win = create_app()
    win.show()
    sys.exit(app.exec())
