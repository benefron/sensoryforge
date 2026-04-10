"""Tests for M-1: SAFilterTorch and RAFilterTorch must satisfy BaseFilter.reset_state() contract.

The BaseFilter interface specifies reset_state() (singular, no args).
SAFilterTorch and RAFilterTorch previously only had reset_states() (plural, takes args),
breaking polymorphism — any code iterating over BaseFilter instances would get AttributeError.
"""

import pytest
import torch
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.filters.base import BaseFilter


def test_sa_filter_is_basefilter_subclass():
    sa = SAFilterTorch()
    assert isinstance(sa, BaseFilter)


def test_ra_filter_is_basefilter_subclass():
    ra = RAFilterTorch()
    assert isinstance(ra, BaseFilter)


def test_sa_filter_has_reset_state_no_args():
    """SAFilterTorch.reset_state() must exist and accept no arguments."""
    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=0.1)
    sa.reset_state()  # must not raise AttributeError or TypeError


def test_ra_filter_has_reset_state_no_args():
    """RAFilterTorch.reset_state() must exist and accept no arguments."""
    ra = RAFilterTorch(tau_RA=30.0, k3=2.0, dt=0.1)
    ra.reset_state()  # must not raise AttributeError or TypeError


def test_sa_filter_reset_state_clears_cached_tensors():
    """reset_state() must set x and I_SA back to None."""
    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=0.1)
    # Run forward to populate state
    x = torch.ones(1, 20, 4)
    sa(x)
    assert sa.x is not None, "x should be populated after forward pass"
    assert sa.I_SA is not None, "I_SA should be populated after forward pass"

    sa.reset_state()
    assert sa.x is None, "x should be None after reset_state()"
    assert sa.I_SA is None, "I_SA should be None after reset_state()"


def test_ra_filter_reset_state_clears_cached_tensors():
    """reset_state() must set I_RA back to None."""
    ra = RAFilterTorch(tau_RA=30.0, k3=2.0, dt=0.1)
    x = torch.ones(1, 20, 4)
    ra(x)
    assert ra.I_RA is not None, "I_RA should be populated after forward pass"

    ra.reset_state()
    assert ra.I_RA is None, "I_RA should be None after reset_state()"


def test_basefilter_polymorphism_reset_state():
    """Iterating a list of BaseFilter instances and calling reset_state() must not raise."""
    filters = [
        SAFilterTorch(dt=0.1),
        RAFilterTorch(dt=0.1),
    ]
    x = torch.randn(1, 50, 8)
    for f in filters:
        f(x)  # populate state

    # This is the pattern that was breaking with AttributeError before the fix
    for f in filters:
        f.reset_state()  # must work for all BaseFilter subclasses


def test_reset_state_allows_fresh_forward():
    """After reset_state(), filter must process a new sequence cleanly."""
    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=0.1)
    x1 = torch.ones(1, 50, 4) * 10.0
    out1 = sa(x1)

    sa.reset_state()

    x2 = torch.ones(1, 50, 4) * 10.0
    out2 = sa(x2)

    # Outputs should be identical since both started from zero state
    assert torch.allclose(out1, out2, atol=1e-5), (
        "After reset_state(), identical input should produce identical output"
    )
