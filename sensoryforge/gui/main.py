"""PyQt5 entry point assembling the tactile simulation GUI tabs."""

import os
import sys
from pathlib import Path
import json

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Ensure repository root on sys.path for package imports when run as a script
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensoryforge.gui.protocol_execution_controller import (  # noqa: E402
    ProtocolExecutionController,
)
from sensoryforge.gui.tabs import (  # noqa: E402
    MechanoreceptorTab,
    StimulusDesignerTab,
    SpikingNeuronTab,
    ProtocolSuiteTab,
)
from sensoryforge.utils.project_registry import ProjectRegistry  # noqa: E402


class TactileSimulationWindow(QtWidgets.QMainWindow):
    """Main application window hosting modular tactile simulation tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SensoryForge – Sensory Encoding")
        self.setMinimumSize(1024, 680)
        self.resize(1200, 780)

        # Create menu bar
        self._create_menu_bar()

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

        # NOTE: AnalyticalInversion tab excluded from v0.1.0 (requires decoding).
        # Will be added in v0.2.0+ after Papers 2-3 publication.

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

    def _create_menu_bar(self) -> None:
        """Create menu bar with config load/save options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Load config action
        load_action = QtWidgets.QAction('&Load Config (YAML)...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load configuration from YAML file')
        load_action.triggered.connect(self._load_config)
        file_menu.addAction(load_action)
        
        # Save config action
        save_action = QtWidgets.QAction('&Save Config (YAML)...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current configuration to YAML file')
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QtWidgets.QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About action
        about_action = QtWidgets.QAction('&About', self)
        about_action.setStatusTip('About SensoryForge')
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        # Phase 2 info action
        phase2_action = QtWidgets.QAction('Phase &2 Features', self)
        phase2_action.setStatusTip('Information about Phase 2 features')
        phase2_action.triggered.connect(self._show_phase2_info)
        help_menu.addAction(phase2_action)

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Configuration',
            '',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        
        if filename:
            try:
                import yaml
                with open(filename, 'r') as f:
                    config = yaml.safe_load(f)
                
                QMessageBox.information(
                    self,
                    'Config Loaded',
                    f'Configuration loaded from:\n{filename}\n\n'
                    'Note: Full YAML config integration with GUI coming soon.\n'
                    'Currently, configs can be used with CLI:\n'
                    f'sensoryforge run {filename}'
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    'Load Error',
                    f'Failed to load configuration:\n{str(e)}'
                )

    def _save_config(self) -> None:
        """Save current configuration to YAML file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Configuration',
            '',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        
        if filename:
            try:
                # Create a basic config from current GUI state
                # This is a minimal implementation - full integration coming soon
                config = {
                    'pipeline': {
                        'device': 'cpu',
                        'seed': 42
                    },
                    'gui': {
                        'note': 'Generated from GUI - full integration coming soon'
                    }
                }
                
                import yaml
                with open(filename, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                QMessageBox.information(
                    self,
                    'Config Saved',
                    f'Basic configuration template saved to:\n{filename}\n\n'
                    'Note: Full YAML config export from GUI coming soon.\n'
                    'Edit the YAML file and use with CLI:\n'
                    f'sensoryforge run {filename}'
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    'Save Error',
                    f'Failed to save configuration:\n{str(e)}'
                )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            'About SensoryForge',
            '<h2>SensoryForge v0.2.0</h2>'
            '<p>Modular, extensible framework for simulating sensory '
            'encoding across modalities.</p>'
            '<p><b>Features:</b></p>'
            '<ul>'
            '<li>Multi-population grids (CompositeGrid)</li>'
            '<li>Equation DSL for custom neuron models</li>'
            '<li>Extended stimuli (texture, moving)</li>'
            '<li>Adaptive ODE solvers</li>'
            '<li>YAML configuration</li>'
            '<li>Command-line interface</li>'
            '</ul>'
            '<p><b>Usage:</b></p>'
            '<ul>'
            '<li>GUI: Interactive simulation design</li>'
            '<li>CLI: <tt>sensoryforge run config.yml</tt></li>'
            '<li>Python API: <tt>from sensoryforge.core import ...</tt></li>'
            '</ul>'
        )

    def _show_phase2_info(self) -> None:
        """Show Phase 2 features information."""
        QMessageBox.information(
            self,
            'Phase 2 Features',
            '<h3>Phase 2 Integration Features</h3>'
            '<p><b>CompositeGrid:</b> Multi-population receptor mosaics<br>'
            'Example: SA1, RA1, SA2 with different densities and arrangements</p>'
            '<p><b>Equation DSL:</b> Define neuron models via equations<br>'
            'Example: Custom Izhikevich variants without coding</p>'
            '<p><b>Extended Stimuli:</b> Texture and moving stimuli<br>'
            'Example: Gabor patches, edge gratings, linear/circular motion</p>'
            '<p><b>Adaptive Solvers:</b> High-precision ODE integration<br>'
            'Example: Dormand-Prince (RK45) for stiff systems</p>'
            '<p><b>YAML Configuration:</b> Declarative pipeline setup<br>'
            'Use: <tt>sensoryforge validate config.yml</tt></p>'
            '<p><b>CLI:</b> Command-line simulation execution<br>'
            'Commands: run, validate, visualize, list-components</p>'
            '<hr>'
            '<p><i>GUI integration for Phase 2 features coming soon!</i><br>'
            'Currently use YAML configs with CLI for full access.</p>'
        )


def main() -> None:
    """Launch the tactile simulation application."""
    app = QtWidgets.QApplication(sys.argv)
    window = TactileSimulationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
