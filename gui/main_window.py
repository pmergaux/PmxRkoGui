# gui/main_window.py
from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.config_tab import ConfigTab
from gui.execution_tab import ExecutionTab
from gui.optimization_tab import OptimizationTab
from utils.utils import load_config
from simulation.simulator import RenkoSimulator

class PmxRkoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PmxRkoTrader - ÉTAPE 1")
        self.setGeometry(100, 100, 1400, 900)

        self.live_config = load_config(self, "config_live.json", required=True)
        self.optim_config = load_config(self,"config_optim.json", required=True)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # === ONGLET 1 : CONFIGURATION ===
        self.config_tab = ConfigTab(self, self.live_config)
        tabs.addTab(self.config_tab, "Configuration")
        # === ONGLET 2 : EXÉCUTION ===
        self.execution_tab = ExecutionTab(self)
        tabs.addTab(self.execution_tab, "Exécution")
        # === ONGLET 3 : OPTIMISATION ===
        self.optimization_tab = OptimizationTab(self, self.optim_config)
        tabs.addTab(self.optimization_tab, "Optimisation")

        # === SIMULATEUR ===
        self.simulator = None

    def start_execution(self):
        cfg = self.config_tab.get_config()
        self.simulator = RenkoSimulator(self.exec_tab, cfg)  # ← CORRIGÉ : 3 args OK
        self.simulator.timer.start(100)  # ← Le timer est DANS le simulateur

    def stop_execution(self):
        if self.simulator:
            self.simulator.timer.stop()

