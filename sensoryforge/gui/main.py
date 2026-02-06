"""PyQt5 entry point assembling the tactile simulation GUI tabs."""

import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets

# Ensure repository root on sys.path for package imports when run as a script
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from GUIs.protocol_execution_controller import (  # noqa: E402
    ProtocolExecutionController,
)
from GUIs.tabs import (  # noqa: E402
    MechanoreceptorTab,
    StimulusDesignerTab,
    SpikingNeuronTab,
    ProtocolSuiteTab,
    AnalyticalInversionTab,
)
from utils.project_registry import ProjectRegistry  # noqa: E402


class TactileSimulationWindow(QtWidgets.QMainWindow):
    """Main application window hosting modular tactile simulation tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bio-inspired Tactile Simulation")
        self.setMinimumSize(1024, 680)
        self.resize(1200, 780)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        registry_root = Path.cwd() / "project_registry"
        self.project_registry = ProjectRegistry(registry_root)

        self.mechanoreceptor_tab = MechanoreceptorTab()
        tabs.addTab(
            self.mechanoreceptor_tab,
            "Mechanoreceptors & Innervation",
        )

        self.stimulus_tab = StimulusDesignerTab(self.mechanoreceptor_tab)
        tabs.addTab(self.stimulus_tab, "Stimulus Designer")

        self.spiking_tab = SpikingNeuronTab(
            self.mechanoreceptor_tab,
            self.stimulus_tab,
        )
        tabs.addTab(self.spiking_tab, "Spiking Neurons")

        self.protocol_tab = ProtocolSuiteTab(
            self.mechanoreceptor_tab,
            self.stimulus_tab,
            self.spiking_tab,
            project_registry=self.project_registry,
        )
        tabs.addTab(self.protocol_tab, "Protocol Suite")

        self.analytical_inversion_tab = AnalyticalInversionTab(
            self.mechanoreceptor_tab,
            self.stimulus_tab,
        )
        tabs.addTab(self.analytical_inversion_tab, "Analytical Inversion")

        self.execution_controller = ProtocolExecutionController(
            self.mechanoreceptor_tab,
            self.spiking_tab,
            self.protocol_tab,
            self.project_registry,
            parent=self,
        )
        self.protocol_tab.run_requested.connect(self.execution_controller.start_batch)
        # self.protocol_tab.load_run_requested.connect(
        #     self.execution_controller.load_run_into_analysis
        # )
        self.execution_controller.batch_finished.connect(
            lambda: self.protocol_tab.set_running(False)
        )

        existing_runs = self.execution_controller.bootstrap_analysis_runs()
        # self.sta_analysis_tab.load_runs(existing_runs)


def main() -> None:
    """Launch the tactile simulation application."""
    app = QtWidgets.QApplication(sys.argv)
    window = TactileSimulationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
