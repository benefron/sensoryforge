"""PyQt5 entry point assembling the SensoryForge GUI.

The GUI is the primary experimentation tool — an interactive workbench for
designing sensory encoding experiments, tuning parameters, and observing
population responses in real time.  Once a configuration is validated here,
it can be exported to YAML and scaled via the CLI for batch runs.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Ensure repository root on sys.path for package imports when run as a script
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensoryforge.gui.tabs import (  # noqa: E402
    MechanoreceptorTab,
    StimulusDesignerTab,
    SpikingNeuronTab,
)
from sensoryforge.utils.project_registry import ProjectRegistry  # noqa: E402


class SensoryForgeWindow(QtWidgets.QMainWindow):
    """Main application window for interactive sensory encoding experiments.

    The GUI provides three core tabs:

    1. **Mechanoreceptors & Innervation** — spatial grid, receptor populations,
       receptive field visualization.
    2. **Stimulus Designer** — interactive stimulus creation and preview.
    3. **Spiking Neurons** — neuron model configuration, simulation, and
       spike-train visualization.
    Configurations designed in the GUI can be exported to YAML for batch
    execution via the CLI (``sensoryforge run config.yml``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SensoryForge – Sensory Encoding Workbench")
        self.setMinimumSize(1024, 680)
        self.resize(1200, 780)

        # Create menu bar
        self._create_menu_bar()

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

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

        # Create project registry (used for config save/load and exports)
        registry_root = Path.cwd() / "project_registry"
        self._registry = ProjectRegistry(registry_root)

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
        
        # Advanced features action
        advanced_action = QtWidgets.QAction('&Advanced Features', self)
        advanced_action.setStatusTip('Information about advanced features')
        advanced_action.triggered.connect(self._show_phase2_info)
        help_menu.addAction(advanced_action)
        
        # CLI Guide action
        cli_guide_action = QtWidgets.QAction('&CLI Guide', self)
        cli_guide_action.setStatusTip('How to use the command-line interface')
        cli_guide_action.triggered.connect(self._show_cli_guide)
        help_menu.addAction(cli_guide_action)

    def _load_config(self) -> None:
        """Load YAML configuration and populate all GUI tabs.

        Calls each tab's ``set_config()`` in dependency order:
        mechanoreceptor → stimulus → spiking.  The config structure
        mirrors what ``_save_config()`` produces, enabling full
        round-trip fidelity.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load YAML Configuration',
            '',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("YAML did not produce a dict")
        except Exception as e:
            QMessageBox.critical(
                self, 'Load Error',
                f'Failed to parse YAML:\n{e}'
            )
            return

        errors: list = []

        # 1) Mechanoreceptor tab (grid + populations)
        mechano_cfg = {}
        if "grid" in config or "populations" in config:
            mechano_cfg["grid"] = config.get("grid", {})
            mechano_cfg["populations"] = config.get("populations", [])
        try:
            if mechano_cfg:
                self.mechanoreceptor_tab.set_config(mechano_cfg)
        except Exception as e:
            errors.append(f"Grid/Populations: {e}")

        # 2) Stimulus tab
        try:
            stim_cfg = config.get("stimulus", {})
            if stim_cfg:
                self.stimulus_tab.set_config(stim_cfg)
        except Exception as e:
            errors.append(f"Stimulus: {e}")

        # 3) Spiking tab (must come after mechano so population names exist)
        try:
            sim_cfg = config.get("simulation", {})
            if sim_cfg:
                self.spiking_tab.set_config(sim_cfg)
        except Exception as e:
            errors.append(f"Simulation: {e}")

        # Report result
        if errors:
            QMessageBox.warning(
                self, 'Load Warnings',
                f'Configuration loaded with warnings:\n\n'
                + '\n'.join(errors)
            )
        else:
            QMessageBox.information(
                self, 'Config Loaded',
                f'Configuration loaded successfully from:\n{filename}'
            )

    def _save_config(self) -> None:
        """Save current GUI state to a YAML configuration file.

        Collects config from each tab via ``get_config()`` and writes a
        single YAML that can be loaded back with ``_load_config()``,
        ensuring full round-trip fidelity.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Configuration',
            'sensoryforge_config.yml',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        if not filename:
            return

        try:
            mechano = self.mechanoreceptor_tab.get_config()
            stimulus = self.stimulus_tab.get_config()
            simulation = self.spiking_tab.get_config()

            config: dict = {
                'metadata': {
                    'version': '0.3.0',
                    'created': datetime.utcnow().isoformat() + 'Z',
                    'source': 'SensoryForge GUI',
                },
                'grid': mechano.get('grid', {}),
                'populations': mechano.get('populations', []),
                'stimulus': stimulus,
                'simulation': simulation,
            }

            with open(filename, 'w') as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            QMessageBox.information(
                self,
                'Config Saved',
                f'Configuration saved to:\n{filename}\n\n'
                f'Load back with File → Load Config, or run via CLI:\n'
                f'sensoryforge run {filename}'
            )
        except Exception as e:
            QMessageBox.critical(
                self, 'Save Error',
                f'Failed to save configuration:\n{e}'
            )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            'About SensoryForge',
            '<h2>SensoryForge v0.2.0</h2>'
            '<p>An extensible playground for generating population activity '
            'in response to multiple stimuli and modalities.</p>'
            '<p><b>Core Features:</b></p>'
            '<ul>'
            '<li>Multiple spiking neuron models (Izhikevich, AdEx, MQIF, FA, SA)</li>'
            '<li>SA/RA dual-pathway temporal filtering</li>'
            '<li>Multi-population grids (CompositeGrid)</li>'
            '<li>Equation DSL for custom neuron models</li>'
            '<li>Extended stimuli (Gaussian, texture, moving)</li>'
            '<li>Adaptive ODE solvers (Euler, Dormand-Prince)</li>'
            '<li>YAML configuration &amp; CLI for scalable batch runs</li>'
            '</ul>'
            '<p><b>Workflow:</b></p>'
            '<ul>'
            '<li><b>GUI:</b> Design &amp; test experiments interactively</li>'
            '<li><b>CLI:</b> <tt>sensoryforge run config.yml</tt> for batch runs</li>'
            '<li><b>Python API:</b> <tt>from sensoryforge.core import ...</tt></li>'
            '</ul>'
        )

    def _show_phase2_info(self) -> None:
        """Show advanced features information."""
        QMessageBox.information(
            self,
            'Advanced Features',
            '<h3>Advanced Features</h3>'
            '<p><b>CompositeGrid:</b> Multi-population receptor mosaics<br>'
            'SA1, RA1, SA2 populations with different densities and arrangements</p>'
            '<p><b>Equation DSL:</b> Define neuron models via equations<br>'
            'Custom Izhikevich variants, AdEx, or your own dynamics</p>'
            '<p><b>Extended Stimuli:</b> Texture and moving stimuli<br>'
            'Gabor patches, edge gratings, linear/circular motion</p>'
            '<p><b>Adaptive Solvers:</b> High-precision ODE integration<br>'
            'Dormand-Prince (RK45) for stiff systems</p>'
            '<p><b>YAML Configuration:</b> Declarative pipeline setup<br>'
            '<tt>sensoryforge validate config.yml</tt></p>'
            '<p><b>CLI:</b> Scalable batch simulation execution<br>'
            'Commands: run, validate, visualize, list-components</p>'
            '<hr>'
            '<p>Design experiments in the GUI, then export to YAML '
            'for large-scale batch runs via the CLI.</p>'
        )

    def _show_cli_guide(self) -> None:
        """Show CLI usage guide."""
        QMessageBox.information(
            self,
            'SensoryForge CLI Guide',
            '<h3>Command-Line Interface</h3>'
            '<p><b>Run a simulation:</b></p>'
            '<pre>sensoryforge run config.yml --duration 1000 --output results.pt</pre>'
            '<p><b>Validate configuration:</b></p>'
            '<pre>sensoryforge validate config.yml</pre>'
            '<p><b>List available components:</b></p>'
            '<pre>sensoryforge list-components</pre>'
            '<p><b>Visualize pipeline:</b></p>'
            '<pre>sensoryforge visualize config.yml</pre>'
            '<hr>'
            '<h4>Example YAML Configuration:</h4>'
            '<pre>'
            'grid:\n'
            '  type: composite\n'
            '  populations:\n'
            '    sa1: {density: 10.0, arrangement: poisson}\n'
            '    ra1: {density: 5.0, arrangement: hex}\n\n'
            'neurons:\n'
            '  type: dsl\n'
            '  equations: "dv/dt = 0.04*v**2 + 5*v + 140 - u + I"\n'
            '  threshold: "v >= 30"\n'
            '  reset: "v = -65"\n\n'
            'stimuli:\n'
            '  - type: texture\n'
            '    config: {pattern: gabor, wavelength: 2.0}\n\n'
            'solver:\n'
            '  type: adaptive\n'
            '  config: {method: dopri5}\n'
            '</pre>'
            '<p>See docs/user_guide/cli.md for complete documentation.</p>'
        )


def main() -> None:
    """Launch the SensoryForge GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    window = SensoryForgeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
