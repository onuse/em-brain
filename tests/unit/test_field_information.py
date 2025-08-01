#!/usr/bin/env python3
"""Test field information calculation to debug high values."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from server.src.utils.tensor_ops import field_information, field_stats


def test_field_information_calculation():
    """Test different field states and their information values."""
    
    # Test 1: Zero field
    zero_field = torch.zeros(32, 32, 32, 64)
    info = field_information(zero_field)
    print(f"Zero field information: {info}")
    
    # Test 2: Normal activation field (typical range -1 to 1)
    normal_field = torch.tanh(torch.randn(32, 32, 32, 64))
    info = field_information(normal_field)
    print(f"Normal field information: {info}")
    
    # Test 3: Field with some large values
    large_field = torch.randn(32, 32, 32, 64) * 10
    info = field_information(large_field)
    print(f"Large field information: {info}")
    
    # Test 4: Extremely large values (like we're seeing)
    extreme_field = torch.randn(32, 32, 32, 64) * 1e6
    info = field_information(extreme_field)
    print(f"Extreme field information: {info}")
    
    # Test 5: Check a typical brain field pattern
    brain_field = torch.zeros(32, 32, 32, 64)
    brain_field[:, :, :, :32] = torch.tanh(torch.randn(32, 32, 32, 32))  # Content
    brain_field[:, :, :, 32:48] = torch.randn(32, 32, 32, 16) * 0.1  # Memory
    brain_field[:, :, :, 48:] = torch.randn(32, 32, 32, 16) * 0.01  # Evolution
    info = field_information(brain_field)
    stats = field_stats(brain_field)
    print(f"\nTypical brain field:")
    print(f"  Information: {info}")
    print(f"  Stats: {stats}")
    
    # Check for NaN or Inf
    print(f"\nChecking for numerical issues:")
    print(f"  Has NaN: {torch.isnan(brain_field).any()}")
    print(f"  Has Inf: {torch.isinf(brain_field).any()}")
    print(f"  Max value: {brain_field.max()}")
    print(f"  Min value: {brain_field.min()}")


if __name__ == "__main__":
    test_field_information_calculation()