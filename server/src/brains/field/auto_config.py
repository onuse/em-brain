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
    
    # Additional tensors (gradients, temporaries, etc.)
    # Conservative estimate: 3x the field size
    total_bytes = field_size * 3
    
    return total_bytes / 1e9


def get_optimal_config(target='balanced', memory_limit=None):
    """
    Get optimal brain configuration for available hardware.
    
    Args:
        target: 'speed', 'balanced', or 'intelligence'
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
        
        # Use 85% of available memory (leave headroom)
        usable_gb = available_gb * 0.85
        
    else:
        device = torch.device('cpu')
        gpu_name = "CPU"
        usable_gb = 2  # Conservative for CPU
    
    # Configuration table
    # (spatial, channels, min_memory_gb, name)
    configs = [
        (16, 32, 0.1, "tiny"),
        (24, 48, 0.5, "small"),
        (32, 64, 2.0, "medium"),
        (40, 80, 4.0, "large"),
        (48, 96, 6.0, "xlarge"),
        (56, 112, 8.0, "xxlarge"),
        (64, 128, 12.0, "huge"),
        (80, 160, 20.0, "gigantic"),
        (96, 192, 35.0, "colossal"),
        (128, 256, 80.0, "maximum"),
    ]
    
    # Select based on target
    if target == 'speed':
        # Prioritize fast processing
        # Smaller brain, higher frequency
        max_params = 5_000_000  # 5M max for speed
        selected = None
        for spatial, channels, min_mem, name in configs:
            params = spatial ** 3 * channels
            mem_usage = calculate_memory_usage(spatial, channels)
            if params <= max_params and mem_usage <= usable_gb:
                selected = (spatial, channels, name)
        
        if not selected:
            selected = (16, 32, "tiny")
            
    elif target == 'intelligence':
        # Maximum size that fits
        selected = None
        for spatial, channels, min_mem, name in reversed(configs):
            mem_usage = calculate_memory_usage(spatial, channels)
            if mem_usage <= usable_gb:
                selected = (spatial, channels, name)
                break
        
        if not selected:
            selected = (16, 32, "tiny")
            
    else:  # balanced
        # Good tradeoff between size and speed
        # Aim for ~500Hz processing
        selected = None
        target_params = 10_000_000  # 10M for balance
        
        best_diff = float('inf')
        for spatial, channels, min_mem, name in configs:
            params = spatial ** 3 * channels
            mem_usage = calculate_memory_usage(spatial, channels)
            
            if mem_usage <= usable_gb:
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    selected = (spatial, channels, name)
        
        if not selected:
            selected = (16, 32, "tiny")
    
    spatial, channels, size_name = selected
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
        'size_name': size_name,
        'target': target,
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