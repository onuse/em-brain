#!/usr/bin/env python3
"""
Automatic GPU-optimal configuration for Field Brain.

Detects your GPU and sets the best parameters automatically.
"""

import torch
import numpy as np


def get_gpu_memory():
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0
    
    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1e9
    
    # Check currently allocated
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    available_gb = total_memory_gb - allocated_gb
    
    return available_gb, total_memory_gb


def calculate_memory_usage(spatial_size, channels):
    """Calculate estimated memory usage in GB."""
    # Main field tensor
    field_size = spatial_size ** 3 * channels * 4  # float32
    
    # Additional tensors needed:
    # - Field momentum (same size as field)
    # - Temporary tensors for computation (~0.5x field)
    # Total: 2.5x the field size (no training, no gradients!)
    total_bytes = field_size * 2.5
    
    return total_bytes / 1e9


def get_optimal_config(memory_limit=None):
    """
    Get optimal brain configuration for available hardware.
    Uses all available resources.
    
    Args:
        memory_limit: Override automatic detection (GB)
        
    Returns:
        Dict with optimal configuration
    """
    # Detect GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_properties(0).name
        available_gb, total_gb = get_gpu_memory()
        
        if memory_limit:
            available_gb = min(available_gb, memory_limit)
        
        # Use 90% of available memory (we're inference-only, no gradient accumulation)
        usable_gb = available_gb * 0.90
        
    else:
        device = torch.device('cpu')
        gpu_name = "CPU"
        usable_gb = 2  # Conservative for CPU
    
    # Configuration table - just sizes, no names
    configs = [
        (16, 32),
        (24, 48),
        (32, 64),
        (40, 80),
        (48, 96),
        (56, 112),
        (64, 128),
        (80, 160),
        (96, 192),
        (128, 256),
    ]
    
    # Find the largest configuration that fits in available memory
    spatial, channels = (16, 32)  # Start with minimum
    
    for s, c in reversed(configs):
        mem_usage = calculate_memory_usage(s, c)
        if mem_usage <= usable_gb:
            spatial, channels = s, c
            break
    params = spatial ** 3 * channels
    memory_usage = calculate_memory_usage(spatial, channels)
    
    # Estimate processing speed
    if device.type == 'cuda':
        # Rough estimates for RTX 3070-class GPU
        if params < 1_000_000:
            hz_estimate = 5000
        elif params < 5_000_000:
            hz_estimate = 2000
        elif params < 10_000_000:
            hz_estimate = 500
        elif params < 20_000_000:
            hz_estimate = 250
        elif params < 50_000_000:
            hz_estimate = 100
        else:
            hz_estimate = 50
    else:
        hz_estimate = max(10, 1000000 / params)  # CPU estimate
    
    config = {
        'spatial_size': spatial,
        'channels': channels,
        'parameters': params,
        'memory_gb': memory_usage,
        'device': device,
        'gpu_name': gpu_name,
        'estimated_hz': hz_estimate,
        'available_memory_gb': available_gb if torch.cuda.is_available() else None,
        'total_memory_gb': total_gb if torch.cuda.is_available() else None,
    }
    
    return config


# Test/Demo
if __name__ == "__main__":
    print("Detecting optimal configuration for your hardware...")
    
    config = get_optimal_config()
    
    print("\n" + "="*60)
    print("OPTIMAL BRAIN CONFIGURATION")
    print("="*60)
    
    print(f"\nHardware: {config['gpu_name']}")
    if config['available_memory_gb']:
        print(f"Memory: {config['available_memory_gb']:.1f}/{config['total_memory_gb']:.1f} GB available")
    
    print(f"\nConfiguration:")
    print(f"  Spatial: {config['spatial_size']}³")
    print(f"  Channels: {config['channels']}")
    print(f"  Parameters: {config['parameters']:,}")
    print(f"  Memory Usage: {config['memory_gb']:.2f} GB")
    print(f"  Estimated Speed: {config['estimated_hz']:,} Hz")
    
    print("\n" + "="*60)