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


def print_config(config):
    """Pretty print configuration."""
    print("\n" + "="*60)
    print("OPTIMAL BRAIN CONFIGURATION")
    print("="*60)
    
    print(f"\nHardware: {config['gpu_name']}")
    if config['available_memory_gb']:
        print(f"Memory: {config['available_memory_gb']:.1f}/{config['total_memory_gb']:.1f} GB available")
    
    print(f"\nTarget: {config['target']}")
    print(f"Size: {config['size_name']}")
    
    print(f"\nConfiguration:")
    print(f"  Spatial: {config['spatial_size']}³")
    print(f"  Channels: {config['channels']}")
    print(f"  Parameters: {config['parameters']:,}")
    print(f"  Memory Usage: {config['memory_gb']:.2f} GB")
    print(f"  Estimated Speed: {config['estimated_hz']:,} Hz")
    
    print(f"\nExpected Capabilities:")
    if config['parameters'] < 1_000_000:
        print("  • Basic reflexes and reactions")
        print("  • Simple pattern learning")
    elif config['parameters'] < 5_000_000:
        print("  • Complex reflexes")
        print("  • Pattern recognition")
        print("  • Short-term memory")
    elif config['parameters'] < 20_000_000:
        print("  • Behavioral sequences")
        print("  • Environmental mapping")
        print("  • Associative learning")
    elif config['parameters'] < 50_000_000:
        print("  • Abstract patterns")
        print("  • Strategic behavior")
        print("  • Long-term memory")
    else:
        print("  • Complex reasoning")
        print("  • Creative problem solving")
        print("  • Emergent consciousness?")
    
    print("\n" + "="*60)


def create_optimal_brain(target='balanced', quiet=False):
    """
    Create a brain with optimal settings for your hardware.
    
    Args:
        target: 'speed', 'balanced', or 'intelligence'
        quiet: Suppress output
        
    Returns:
        TrulyMinimalBrain instance
    """
    from .truly_minimal_brain import TrulyMinimalBrain
    
    config = get_optimal_config(target)
    
    if not quiet:
        print_config(config)
    
    brain = TrulyMinimalBrain(
        spatial_size=config['spatial_size'],
        channels=config['channels'],
        device=config['device'],
        quiet_mode=quiet
    )
    
    return brain, config


# Test/Demo
if __name__ == "__main__":
    import sys
    
    target = sys.argv[1] if len(sys.argv) > 1 else 'balanced'
    
    print(f"Detecting optimal configuration for target: {target}")
    
    config = get_optimal_config(target)
    print_config(config)
    
    print("\nTo create this brain:")
    print(f"  brain, config = create_optimal_brain('{target}')")
    
    # Show all options
    print("\n" + "="*60)
    print("ALL CONFIGURATIONS")
    print("="*60)
    
    for t in ['speed', 'balanced', 'intelligence']:
        c = get_optimal_config(t)
        print(f"\n{t.upper():12} -> {c['spatial_size']}³×{c['channels']:3} = {c['parameters']:>10,} params, "
              f"{c['memory_gb']:>5.1f} GB, {c['estimated_hz']:>5,} Hz")