# main.py
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QApplication
from gui.config_tab import ConfigTab
from gui.live_chart import LiveChart
from gui.optimization_tab import OptimizationTab
from simulation.simulator import RenkoSimulator
from utils.utils import load_config


class GrokTrader(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main_window.ui", self)
        self.config_file = "config.json"
        self.cfg = load_config(self, self.config_file)

        self.config_tab = ConfigTab(self, self.cfg)
        self.live_monitor = LiveChart(self, self.cfg)
        self.opt_tab = OptimizationTab(self, self.cfg)

        self.main_tabs.addTab(self.config_tab, "Configuration")
        self.main_tabs.addTab(self.live_monitor, "Trading Live")
        self.main_tabs.addTab(self.opt_tab, "Optimisation")

        # --- c'est livechart qui lance le live
        #self.simulator = RenkoSimulator(self.live_monitor.add_signal)
        #self.simulator.start()

    def get_config(self):
        return self.cfg

    def closeEvent(self, event):
        self.simulator.stop()
        super().closeEvent(event)

app = QApplication([])
win = GrokTrader()
win.show()
app.exec()
