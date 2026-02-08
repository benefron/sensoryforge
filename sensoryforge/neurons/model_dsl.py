"""Domain-specific language for defining neuron models from equations.

This module provides the `NeuronModel` class that allows users to define neuron
models using mathematical equation strings. The equations are parsed with SymPy,
validated, and compiled into PyTorch nn.Module instances compatible with the
existing neuron interface.

Example:
    >>> from sensoryforge.neurons.model_dsl import NeuronModel
    >>> # Define Izhikevich neuron using equations
    >>> equations = '''
    ... dv/dt = 0.04*v**2 + 5*v + 140 - u + I
    ... du/dt = a*(b*v - u)
    ... '''
    >>> threshold = 'v >= 30'
    >>> reset = '''
    ... v = c
    ... u = u + d
    ... '''
    >>> model = NeuronModel(
    ...     equations=equations,
    ...     threshold=threshold,
    ...     reset=reset,
    ...     parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
    ...     state_vars={'v': -65.0, 'u': -13.0}
    ... )
    >>> neuron = model.compile(solver='euler', dt=0.05, device='cpu')
    >>> # Use like any other neuron model
    >>> v_trace, spikes = neuron(input_current)
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn

# Check for sympy availability and provide helpful error message
try:
    import sympy
    from sympy import symbols, sympify, diff, Symbol
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Provide a helpful message when sympy is needed (always define it)
_SYMPY_ERROR_MSG = (
    "SymPy is required for the equation DSL but is not installed.\n"
    "Install it with: pip install 'sensoryforge[dsl]'\n"
    "Or install SymPy directly: pip install sympy>=1.11"
)


class NeuronModel:
    """Parse neuron model equations and compile to PyTorch nn.Module.
    
    This class provides a domain-specific language (DSL) for defining neuron
    models using mathematical equations. It parses differential equations,
    threshold conditions, and reset rules, then compiles them into an efficient
    PyTorch module.
    
    Args:
        equations: String containing differential equations in the form
            'dv/dt = ...' or 'dx/dt = ...'. Multiple equations can be
            separated by newlines. The special variable 'I' represents
            input current.
        threshold: String representing the spike threshold condition,
            e.g., 'v >= 30' or 'v > threshold_value'.
        reset: String containing reset rules when threshold is crossed,
            e.g., 'v = c' or 'v = c\\nu = u + d'. Multiple rules can be
            separated by newlines.
        parameters: Dictionary of model parameters and their values,
            e.g., {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}.
        state_vars: Dictionary of state variable names and their initial
            values, e.g., {'v': -65.0, 'u': -13.0}. If not provided,
            state variables are inferred from equations with zero initial values.
    
    Raises:
        ImportError: If SymPy is not installed.
        ValueError: If equations are malformed or inconsistent.
    
    Example:
        >>> # Define a simple integrate-and-fire neuron
        >>> model = NeuronModel(
        ...     equations='dv/dt = -v + I',
        ...     threshold='v >= 1.0',
        ...     reset='v = 0.0',
        ...     state_vars={'v': 0.0}
        ... )
        >>> neuron = model.compile(dt=0.1)
    """
    
    def __init__(
        self,
        equations: str,
        threshold: str,
        reset: str,
        parameters: Optional[Dict[str, float]] = None,
        state_vars: Optional[Dict[str, float]] = None,
    ):
        """Initialize the neuron model from equation strings.
        
        Args:
            equations: Differential equations defining the model dynamics.
            threshold: Condition for spike generation.
            reset: Rules to apply when threshold is crossed.
            parameters: Model parameters (constants).
            state_vars: State variables and their initial values.
        
        Raises:
            ImportError: If SymPy is not installed.
        """
        if not SYMPY_AVAILABLE:
            raise ImportError(_SYMPY_ERROR_MSG)
        
        self.equations_str = equations.strip()
        self.threshold_str = threshold.strip()
        self.reset_str = reset.strip()
        self.parameters = parameters or {}
        self.state_vars = state_vars or {}
        
        # Parse and validate equations
        self._parse_equations()
        self._parse_threshold()
        self._parse_reset()
        self._validate_model()
    
    def _parse_equations(self) -> None:
        """Parse differential equations and extract state variables and derivatives.
        
        Parses equations in the form 'dx/dt = expression' and extracts:
        - State variable names (x)
        - Derivative expressions
        - All symbols used in expressions
        
        Raises:
            ValueError: If equation format is invalid.
        """
        self.derivatives: Dict[str, sympy.Expr] = {}
        self.state_var_list: List[str] = []
        
        # Split equations by newline and process each
        lines = [line.strip() for line in self.equations_str.split('\n') if line.strip()]
        
        for line in lines:
            # Match pattern: d<var>/dt = <expression>
            match = re.match(r'd(\w+)/dt\s*=\s*(.+)', line)
            if not match:
                raise ValueError(
                    f"Invalid equation format: '{line}'\n"
                    f"Expected format: 'd<variable>/dt = <expression>'"
                )
            
            var_name = match.group(1)
            expr_str = match.group(2)
            
            # Parse the expression using sympy
            # Use local_dict to prevent 'I' from being interpreted as imaginary unit
            try:
                # Create a local namespace where 'I' is a symbol, not imaginary unit
                I_symbol = Symbol('I', real=True)
                local_dict = {'I': I_symbol}
                expr = parse_expr(expr_str, local_dict=local_dict)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse expression '{expr_str}': {str(e)}"
                )
            
            self.derivatives[var_name] = expr
            self.state_var_list.append(var_name)
            
            # Initialize state variable if not provided
            if var_name not in self.state_vars:
                self.state_vars[var_name] = 0.0
        
        if not self.derivatives:
            raise ValueError("No valid differential equations found")
    
    def _parse_threshold(self) -> None:
        """Parse threshold condition for spike detection.
        
        Parses conditions like 'v >= 30' or 'x > threshold_val' and
        extracts the comparison operator and threshold value.
        
        Raises:
            ValueError: If threshold format is invalid.
        """
        # Match pattern: <var> <operator> <value>
        # Supported operators: >=, >, <=, <, ==
        match = re.match(r'(\w+)\s*(>=|>|<=|<|==)\s*(.+)', self.threshold_str)
        if not match:
            raise ValueError(
                f"Invalid threshold format: '{self.threshold_str}'\n"
                f"Expected format: '<variable> <operator> <value>'"
            )
        
        self.threshold_var = match.group(1)
        self.threshold_op = match.group(2)
        threshold_val_str = match.group(3)
        
        # Try to parse threshold value as expression
        try:
            # Prevent 'I' from being imaginary unit
            I_symbol = Symbol('I', real=True)
            local_dict = {'I': I_symbol}
            self.threshold_expr = parse_expr(threshold_val_str, local_dict=local_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to parse threshold value '{threshold_val_str}': {str(e)}"
            )
    
    def _parse_reset(self) -> None:
        """Parse reset rules to apply when threshold is crossed.
        
        Parses assignments like 'v = c' or 'u = u + d' and stores
        them as sympy expressions.
        
        Raises:
            ValueError: If reset rule format is invalid.
        """
        self.reset_rules: Dict[str, sympy.Expr] = {}
        
        # Split reset rules by newline
        lines = [line.strip() for line in self.reset_str.split('\n') if line.strip()]
        
        for line in lines:
            # Match pattern: <var> = <expression>
            match = re.match(r'(\w+)\s*=\s*(.+)', line)
            if not match:
                raise ValueError(
                    f"Invalid reset rule format: '{line}'\n"
                    f"Expected format: '<variable> = <expression>'"
                )
            
            var_name = match.group(1)
            expr_str = match.group(2)
            
            # Parse the expression
            try:
                # Prevent 'I' from being imaginary unit
                I_symbol = Symbol('I', real=True)
                local_dict = {'I': I_symbol}
                expr = parse_expr(expr_str, local_dict=local_dict)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse reset expression '{expr_str}': {str(e)}"
                )
            
            self.reset_rules[var_name] = expr
    
    def _validate_model(self) -> None:
        """Validate that the model is internally consistent.
        
        Checks:
        - Threshold variable is a state variable
        - Reset rules reference valid variables
        - All symbols in equations are defined (state vars or parameters)
        
        Raises:
            ValueError: If model is inconsistent.
        """
        # Check threshold variable exists
        if self.threshold_var not in self.state_var_list:
            raise ValueError(
                f"Threshold variable '{self.threshold_var}' is not a state variable. "
                f"Available state variables: {self.state_var_list}"
            )
        
        # Check reset rules reference valid variables
        for var_name in self.reset_rules.keys():
            if var_name not in self.state_var_list:
                raise ValueError(
                    f"Reset rule variable '{var_name}' is not a state variable. "
                    f"Available state variables: {self.state_var_list}"
                )
        
        # Collect all symbols used in equations
        all_symbols = set()
        for expr in self.derivatives.values():
            all_symbols.update(str(s) for s in expr.free_symbols)
        for expr in self.reset_rules.values():
            all_symbols.update(str(s) for s in expr.free_symbols)
        all_symbols.update(str(s) for s in self.threshold_expr.free_symbols)
        
        # Check all symbols are defined (excluding 'I' which is input current)
        undefined = all_symbols - set(self.state_var_list) - set(self.parameters.keys()) - {'I'}
        if undefined:
            raise ValueError(
                f"Undefined symbols in equations: {undefined}\n"
                f"Define them in 'parameters' or 'state_vars'"
            )
    
    def compile(
        self,
        solver: str = 'euler',
        dt: float = 0.05,
        device: str = 'cpu',
        noise_std: float = 0.0,
    ) -> nn.Module:
        """Compile the model to a PyTorch nn.Module.
        
        Args:
            solver: Integration method. Currently only 'euler' is supported.
            dt: Time step for integration in milliseconds.
            device: Device to place the module on ('cpu', 'cuda', or 'mps').
            noise_std: Standard deviation of additive Langevin noise (mV/sqrt(ms)).
                Applied to the first state variable during integration.
        
        Returns:
            A PyTorch nn.Module with the standard neuron interface:
            - forward(input_current) -> (v_trace, spikes)
            - input_current: [batch, steps, features]
            - v_trace: [batch, steps+1, features] 
            - spikes: [batch, steps+1, features] (bool)
        
        Raises:
            ValueError: If solver is not supported.
        
        Example:
            >>> model = NeuronModel(equations=..., threshold=..., reset=...)
            >>> neuron = model.compile(dt=0.1, device='cuda')
            >>> v_trace, spikes = neuron(input_current)
        """
        if solver != 'euler':
            raise ValueError(
                f"Unsupported solver: '{solver}'. Only 'euler' is currently supported."
            )
        
        return _CompiledNeuronModule(
            model=self,
            dt=dt,
            device=device,
            noise_std=noise_std,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model configuration to a dictionary.
        
        Returns:
            Dictionary containing all model configuration that can be
            used to reconstruct the model with from_config().
        
        Example:
            >>> model = NeuronModel(...)
            >>> config = model.to_dict()
            >>> restored_model = NeuronModel.from_config(config)
        """
        return {
            'equations': self.equations_str,
            'threshold': self.threshold_str,
            'reset': self.reset_str,
            'parameters': self.parameters.copy(),
            'state_vars': self.state_vars.copy(),
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NeuronModel':
        """Create a NeuronModel from a configuration dictionary.
        
        Args:
            config: Dictionary with keys 'equations', 'threshold', 'reset',
                and optionally 'parameters' and 'state_vars'.
        
        Returns:
            A new NeuronModel instance.
        
        Raises:
            ValueError: If required keys are missing from config.
        
        Example:
            >>> config = {
            ...     'equations': 'dv/dt = -v + I',
            ...     'threshold': 'v >= 1.0',
            ...     'reset': 'v = 0.0',
            ...     'state_vars': {'v': 0.0}
            ... }
            >>> model = NeuronModel.from_config(config)
        """
        required_keys = ['equations', 'threshold', 'reset']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(
                f"Missing required keys in config: {missing}\n"
                f"Required keys: {required_keys}"
            )
        
        return cls(
            equations=config['equations'],
            threshold=config['threshold'],
            reset=config['reset'],
            parameters=config.get('parameters'),
            state_vars=config.get('state_vars'),
        )


class _CompiledNeuronModule(nn.Module):
    """Internal compiled neuron module implementing the DSL equations.
    
    This class should not be instantiated directly. Use NeuronModel.compile()
    instead. It implements the forward pass using Forward Euler integration
    with the equations defined in the NeuronModel.
    
    Args:
        model: The NeuronModel to compile.
        dt: Integration time step in milliseconds.
        device: Device to place tensors on.
        noise_std: Standard deviation of Langevin noise.
    """
    
    def __init__(
        self,
        model: NeuronModel,
        dt: float,
        device: str,
        noise_std: float,
    ):
        """Initialize the compiled module.
        
        Args:
            model: The NeuronModel containing equations and parameters.
            dt: Time step for integration.
            device: Device for computation.
            noise_std: Noise standard deviation.
        """
        super().__init__()
        self.model = model
        self.dt = dt
        self.device_str = device
        self.noise_std = noise_std
        
        # Create lambdified functions for efficient evaluation
        self._create_lambdas()
    
    def _create_lambdas(self) -> None:
        """Create lambdified (compiled) functions from symbolic expressions.
        
        Converts sympy expressions to Python functions that can efficiently
        evaluate using PyTorch tensors.
        """
        import numpy as np  # Import once at function level
        
        # Get all state variables and parameters
        state_syms = [Symbol(name, real=True) for name in self.model.state_var_list]
        param_syms = [Symbol(name, real=True) for name in self.model.parameters.keys()]
        # Input current symbol (already correctly parsed as 'I', not imaginary unit)
        I_sym = Symbol('I', real=True)
        
        # Create ordered list of all symbols for lambdify
        all_syms = state_syms + param_syms + [I_sym]
        
        # Lambdify derivatives for each state variable
        # Use numpy for better numerical stability, then convert to torch
        self.derivative_funcs = {}
        for var_name, expr in self.model.derivatives.items():
            # Lambdify with numpy
            func_numpy = sympy.lambdify(all_syms, expr, modules='numpy')
            # Wrap to handle torch tensors
            def make_torch_func(np_func):
                def torch_func(*args):
                    # Convert torch tensors to numpy, evaluate, convert back
                    np_args = [a.cpu().numpy() if torch.is_tensor(a) else np.array(a) for a in args]
                    result = np_func(*np_args)
                    # Handle scalar vs array results
                    if np.isscalar(result):
                        return result
                    return torch.from_numpy(np.array(result, dtype=np.float32))
                return torch_func
            self.derivative_funcs[var_name] = make_torch_func(func_numpy)
        
        # Lambdify threshold expression
        threshold_numpy = sympy.lambdify(all_syms, self.model.threshold_expr, modules='numpy')
        def threshold_torch(*args):
            np_args = [a.cpu().numpy() if torch.is_tensor(a) else np.array(a) for a in args]
            result = threshold_numpy(*np_args)
            if np.isscalar(result):
                return result
            return torch.from_numpy(np.array(result, dtype=np.float32))
        self.threshold_func = threshold_torch
        
        # Lambdify reset rules
        self.reset_funcs = {}
        for var_name, expr in self.model.reset_rules.items():
            reset_numpy = sympy.lambdify(all_syms, expr, modules='numpy')
            def make_reset_torch(np_func):
                def reset_torch(*args):
                    np_args = [a.cpu().numpy() if torch.is_tensor(a) else np.array(a) for a in args]
                    result = np_func(*np_args)
                    if np.isscalar(result):
                        return result
                    return torch.from_numpy(np.array(result, dtype=np.float32))
                return reset_torch
            self.reset_funcs[var_name] = make_reset_torch(reset_numpy)
    
    def forward(
        self,
        input_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementing neuron dynamics.
        
        Args:
            input_current: Input currents with shape [batch, steps, features].
                Units are typically mA (milliamperes).
        
        Returns:
            Tuple of (v_trace, spikes) where:
            - v_trace: Membrane potential trajectory [batch, steps+1, features]
                in mV. The first state variable is used for visualization.
            - spikes: Boolean spike events [batch, steps+1, features].
        
        Example:
            >>> neuron = model.compile()
            >>> I = torch.randn(2, 100, 5)  # 2 batch, 100 steps, 5 features
            >>> v_trace, spikes = neuron(I)
            >>> v_trace.shape  # (2, 101, 5)
            >>> spikes.shape   # (2, 101, 5)
        """
        batch, steps, features = input_current.shape
        device = torch.device(self.device_str)
        dtype = input_current.dtype
        
        # Move input to correct device
        input_current = input_current.to(device=device, dtype=dtype)
        
        # Initialize state variables
        state = {}
        for var_name, init_val in self.model.state_vars.items():
            state[var_name] = torch.full(
                (batch, features), init_val, dtype=dtype, device=device
            )
        
        # Create parameter tensors (constant across batch and features)
        params = {}
        for param_name, param_val in self.model.parameters.items():
            params[param_name] = torch.tensor(
                param_val, dtype=dtype, device=device
            )
        
        # Allocate output tensors
        # Use first state variable for voltage trace
        v_var_name = self.model.state_var_list[0]
        v_trace = torch.zeros(
            (batch, steps + 1, features), dtype=dtype, device=device
        )
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        
        # Record initial state
        v_trace[:, 0, :] = state[v_var_name]
        
        # Helper function to ensure real-valued tensor output from lambdified functions
        def ensure_real(val):
            """Convert lambdified output to real tensor."""
            # Handle scalar numeric values
            if isinstance(val, (int, float)):
                return torch.tensor(val, dtype=dtype, device=device)
            # Handle complex scalars
            if isinstance(val, complex):
                if abs(val.imag) > 1e-10:
                    import warnings
                    warnings.warn(f"Discarding imaginary part {val.imag} from equation evaluation")
                return torch.tensor(val.real, dtype=dtype, device=device)
            # Handle tensors
            if not torch.is_tensor(val):
                return torch.tensor(val, dtype=dtype, device=device)
            # If complex tensor, take real part (imaginary should be negligible/zero for real equations)
            if torch.is_complex(val):
                return val.real.to(dtype=dtype, device=device)
            return val.to(dtype=dtype, device=device)
        
        # Helper to prepare evaluation arguments
        def prepare_eval_args(current_timestep):
            """Prepare arguments for lambdified function evaluation."""
            return (
                [state[name] for name in self.model.state_var_list] +
                [params[name] for name in self.model.parameters.keys()] +
                [input_current[:, current_timestep, :]]
            )
        
        # Time integration loop (Forward Euler)
        for t in range(steps):
            # Prepare arguments for lambdified functions
            args = prepare_eval_args(t)
            
            # Check threshold condition
            threshold_val = ensure_real(self.threshold_func(*args))
            
            # Handle different comparison operators
            if self.model.threshold_op == '>=':
                fired = state[self.model.threshold_var] >= threshold_val
            elif self.model.threshold_op == '>':
                fired = state[self.model.threshold_var] > threshold_val
            elif self.model.threshold_op == '<=':
                fired = state[self.model.threshold_var] <= threshold_val
            elif self.model.threshold_op == '<':
                fired = state[self.model.threshold_var] < threshold_val
            elif self.model.threshold_op == '==':
                fired = state[self.model.threshold_var] == threshold_val
            else:
                # Should not happen due to parsing validation
                fired = torch.zeros_like(state[self.model.threshold_var], dtype=torch.bool)
            
            # Record spike and visualization voltage
            spikes[:, t, :] = fired
            v_vis = state[v_var_name].clone()
            # Handle scalar or tensor threshold values correctly
            if threshold_val.numel() == 1:
                v_vis[fired] = threshold_val.item()
            else:
                v_vis[fired] = threshold_val[fired]
            v_trace[:, t, :] = v_vis
            
            # Create state copies for updates
            state_next = {}
            
            # Apply reset rules where fired
            for var_name in self.model.state_var_list:
                if var_name in self.reset_funcs:
                    # Evaluate reset expression
                    reset_val = ensure_real(self.reset_funcs[var_name](*args))
                    state_next[var_name] = torch.where(
                        fired, reset_val, state[var_name]
                    )
                else:
                    # No reset rule - keep current value
                    state_next[var_name] = state[var_name].clone()
            
            # Integrate non-fired neurons (Forward Euler)
            not_fired = ~fired
            for var_name in self.model.state_var_list:
                # Compute derivative
                dvar = ensure_real(self.derivative_funcs[var_name](*args))
                
                # Apply Euler step with optional noise (only for first state variable)
                if var_name == v_var_name and self.noise_std != 0.0:
                    # Add Langevin noise: η ~ N(0, noise_std * sqrt(dt))
                    sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                    eta = torch.randn_like(state[var_name]) * (self.noise_std * sqrt_dt)
                    integrated = state[var_name] + self.dt * dvar + eta
                else:
                    integrated = state[var_name] + self.dt * dvar
                
                # Update only non-fired neurons
                state_next[var_name] = torch.where(
                    not_fired, integrated, state_next[var_name]
                )
            
            # Update state for next iteration
            state = state_next
            
            # Record final state
            v_trace[:, t + 1, :] = state[v_var_name]
            
            # Check for spikes at the end of step (like existing neurons)
            args_final = prepare_eval_args(t)
            threshold_val_final = ensure_real(self.threshold_func(*args_final))
            if self.model.threshold_op == '>=':
                spikes[:, t + 1, :] = state[self.model.threshold_var] >= threshold_val_final
            elif self.model.threshold_op == '>':
                spikes[:, t + 1, :] = state[self.model.threshold_var] > threshold_val_final
            else:
                # For other operators, use the same logic
                spikes[:, t + 1, :] = fired
        
        return v_trace, spikes
