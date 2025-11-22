# gui/main_window.py
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6 import uic

from gui.config_tab import ConfigTab
from gui.execution_tab import ExecutionTab
from gui.optimization_tab import OptimizationTab
from utils.utils import load_config

class PmxRkoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main_window.ui", self)
        self.live_config = load_config(self, "config_live.json", required=True)
        self.optim_config = load_config(self,"config_optim.json", required=True)

        self.config_tab = ConfigTab(self)
        self.exec_tab = ExecutionTab(self)
        self.opt_tab = OptimizationTab(self)

        self.main_tabs.addTab(self.config_tab, "Configuration")
        self.main_tabs.addTab(self.exec_tab, "Trading live")
        self.main_tabs.addTab(self.opt_tab, "Optimisation")

    def start_execution(self, simulator, period):
        print("simulator timer", period)
        simulator.timer.start(period)  # ← Le timer est DANS le simulateur

    def get_config_live(self):
        return self.live_config

    def get_config_optim(self):
        return self.optim_config


