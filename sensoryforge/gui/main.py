"""PyQt5 entry point assembling the SensoryForge GUI.

The GUI is the primary experimentation tool — an interactive workbench for
designing sensory encoding experiments, tuning parameters, and observing
population responses in real time.  Once a configuration is validated here,
it can be exported to YAML and scaled via the CLI for batch runs.
"""

import os
import sys

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
