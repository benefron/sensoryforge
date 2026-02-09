"""PyQt5 entry point assembling the tactile simulation GUI tabs."""

import os
import sys
from pathlib import Path

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
        phase2_action = QtWidgets.QAction('Phase &2 Features (CLI)', self)
        phase2_action.setStatusTip('Information about Phase 2 features (use CLI for full access)')
        phase2_action.triggered.connect(self._show_phase2_info)
        help_menu.addAction(phase2_action)
        
        # CLI Guide action
        cli_guide_action = QtWidgets.QAction('&CLI Guide', self)
        cli_guide_action.setStatusTip('How to use the command-line interface')
        cli_guide_action.triggered.connect(self._show_cli_guide)
        help_menu.addAction(cli_guide_action)

    def _load_config(self) -> None:
        """Load and visualize YAML configuration."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load YAML Configuration',
            '',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        
        if filename:
            try:
                import yaml
                from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
                
                with open(filename, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Try to instantiate pipeline to validate
                try:
                    pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
                    info = pipeline.get_pipeline_info()
                    
                    # Show config summary
                    msg = f'<h3>Configuration Loaded</h3>'
                    msg += f'<p><b>File:</b> {filename}</p>'
                    msg += f'<p><b>Device:</b> {info["config"]["pipeline"]["device"]}</p>'
                    msg += f'<p><b>Grid:</b> {info["grid_properties"]["size"]}</p>'
                    msg += f'<p><b>Neurons:</b> SA={info["neuron_counts"]["sa_neurons"]}, '
                    msg += f'RA={info["neuron_counts"]["ra_neurons"]}</p>'
                    
                    # Check for Phase 2 features
                    if pipeline.composite_grid:
                        msg += '<p><b>✓ CompositeGrid detected</b></p>'
                    if config.get('neurons', {}).get('type') == 'dsl':
                        msg += '<p><b>✓ DSL neuron model detected</b></p>'
                    if config.get('solver', {}).get('type') == 'adaptive':
                        msg += '<p><b>✓ Adaptive solver detected</b></p>'
                    
                    msg += '<hr><p><b>To run simulation:</b></p>'
                    msg += f'<pre>sensoryforge run {filename} --duration 1000</pre>'
                    
                    QMessageBox.information(
                        self,
                        'Config Validated',
                        msg
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        'Config Warning',
                        f'Configuration loaded but validation failed:\n{str(e)}\n\n'
                        f'You can still try running with CLI:\n'
                        f'sensoryforge run {filename}'
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    'Load Error',
                    f'Failed to load configuration:\n{str(e)}'
                )

    def _save_config(self) -> None:
        """Generate and save a YAML configuration template."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Configuration Template',
            'sensoryforge_config.yml',
            'YAML Files (*.yml *.yaml);;All Files (*)'
        )
        
        if filename:
            try:
                # Generate a comprehensive config template with Phase 2 features
                config = {
                    'metadata': {
                        'name': 'SensoryForge Configuration',
                        'version': '0.2.0',
                        'created': 'From GUI',
                        'note': 'Edit this file and use with: sensoryforge run config.yml'
                    },
                    'pipeline': {
                        'device': 'cpu',
                        'seed': 42,
                        'grid_size': 80,
                        'spacing': 0.15,
                        'center': [0.0, 0.0]
                    },
                    '# Uncomment to use CompositeGrid': None,
                    'grid_example_composite': {
                        'type': 'composite',
                        'populations': {
                            'sa1': {'density': 10.0, 'arrangement': 'poisson'},
                            'ra1': {'density': 5.0, 'arrangement': 'hex'},
                            'sa2': {'density': 3.0, 'arrangement': 'poisson'}
                        }
                    },
                    'neurons': {
                        'sa_neurons': 10,
                        'ra_neurons': 14,
                        'sa2_neurons': 5,
                        'dt': 0.5
                    },
                    '# Uncomment to use Equation DSL': None,
                    'neurons_example_dsl': {
                        'type': 'dsl',
                        'equations': 'dv/dt = 0.04*v**2 + 5*v + 140 - u + I\\ndu/dt = a*(b*v - u)',
                        'threshold': 'v >= 30',
                        'reset': 'v = c; u = u + d',
                        'parameters': {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
                    },
                    'stimuli': [
                        {'type': 'gaussian', 'config': {'amplitude': 30.0, 'sigma': 1.0}},
                        '# Uncomment for texture stimulus',
                        {'type_example': 'texture', 'config': {'pattern': 'gabor', 'wavelength': 2.0}},
                        '# Uncomment for moving stimulus',
                        {'type_example': 'moving', 'config': {'motion_type': 'linear', 'start': [-2, 0], 'end': [2, 0]}}
                    ],
                    'solver': {
                        'type': 'euler',
                        '# Use adaptive for higher accuracy': None,
                        'adaptive_example': {'type': 'adaptive', 'config': {'method': 'dopri5', 'rtol': 1e-5}}
                    }
                }
                
                import yaml
                with open(filename, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                QMessageBox.information(
                    self,
                    'Template Saved',
                    f'<h3>Configuration Template Saved</h3>'
                    f'<p><b>File:</b> {filename}</p>'
                    f'<p>This template includes examples for all Phase 2 features:</p>'
                    f'<ul>'
                    f'<li>CompositeGrid (multi-population)</li>'
                    f'<li>Equation DSL (custom neurons)</li>'
                    f'<li>Extended stimuli (texture, moving)</li>'
                    f'<li>Adaptive solvers</li>'
                    f'</ul>'
                    f'<p><b>Next steps:</b></p>'
                    f'<ol>'
                    f'<li>Edit {filename} to configure your simulation</li>'
                    f'<li>Validate: <tt>sensoryforge validate {filename}</tt></li>'
                    f'<li>Run: <tt>sensoryforge run {filename}</tt></li>'
                    f'</ol>'
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
            '<p>Select Help → CLI Guide for examples.</p>'
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
    """Launch the tactile simulation application."""
    app = QtWidgets.QApplication(sys.argv)
    window = TactileSimulationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
