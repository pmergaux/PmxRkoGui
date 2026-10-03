# gui/main_window.py
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6 import uic

from gui.config_tab import ConfigTab
from gui.execution_tab import ExecutionTab
from gui.optimization_tab import OptimizationTab
from utils.config_utils import load_config

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

    def set_config_live(self, cfg):
        self.live_config = cfg

    def set_config_optim(self, cfg):
        self.optim_config = cfg

    """
    # voir + pour charger model, scaler, config
    def launch_best_in_live(self):
        from core.model_manager import ModelManager
        model, scaler, config, results = ModelManager().load_best()
    
        if model is None:
            self.log("Aucun modèle trouvé")
            return
    
        self.live_trader.load_model(model, scaler, config)
        self.live_trader.start()
        self.log(f"LIVE DÉMARRÉ → {results['hash']} | Sharpe {results['sharpe']:.3f}")
    """