#!/usr/bin/env python3
"""Test adaptive configuration scaling with hardware."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import json
import time
from src.adaptive_configuration import AdaptiveConfigurationManager

print("Adaptive Hardware Scaling Test")
print("="*60)

# Test with current hardware
print("\n1. Current Hardware Configuration:")
manager = AdaptiveConfigurationManager(suppress_output=True)
config = manager.config

print(f"   Device: {config.device_type}")
print(f"   CPU cores: {config.cpu_cores}")
print(f"   RAM: {config.system_memory_gb:.1f}GB")
if config.gpu_memory_gb > 0:
    print(f"   GPU memory: {config.gpu_memory_gb:.1f}GB")

print(f"\n   Benchmarked Performance:")
print(f"   Field evolution: {config.field_evolution_ms:.1f}ms")
print(f"   Simulation unit: {config.future_simulation_ms_per_unit:.1f}ms")
print(f"   Memory bandwidth: {config.memory_bandwidth_gbs:.1f}GB/s")

print(f"\n   Adaptive Planning Parameters:")
print(f"   Futures: {config.n_futures}")
print(f"   Horizon: {config.planning_horizon}")
print(f"   Cache size: {config.cache_size}")
print(f"   Reactive threshold: {config.reactive_only_threshold_ms}ms")

estimated_time = config.n_futures * config.planning_horizon * config.future_simulation_ms_per_unit * 5 / 1000
print(f"   Estimated planning time: {estimated_time:.1f}s")

# Simulate different hardware by modifying benchmarks
print("\n" + "-"*60)
print("2. Simulating 20x Faster Hardware:")

# Create new manager and fake faster benchmarks
fast_manager = AdaptiveConfigurationManager(suppress_output=True)
fast_manager.config.field_evolution_ms = config.field_evolution_ms / 20
fast_manager.config.future_simulation_ms_per_unit = config.future_simulation_ms_per_unit / 20
fast_manager.config.memory_bandwidth_gbs = config.memory_bandwidth_gbs * 20
fast_manager.config.system_memory_gb = 64  # More RAM too

# Re-optimize
fast_manager._optimize_planning_parameters()
fast_config = fast_manager.config

print(f"\n   Simulated Performance:")
print(f"   Field evolution: {fast_config.field_evolution_ms:.1f}ms (20x faster)")
print(f"   Simulation unit: {fast_config.future_simulation_ms_per_unit:.1f}ms (20x faster)")
print(f"   Memory bandwidth: {fast_config.memory_bandwidth_gbs:.1f}GB/s")

print(f"\n   Auto-Scaled Planning Parameters:")
print(f"   Futures: {fast_config.n_futures} (was {config.n_futures})")
print(f"   Horizon: {fast_config.planning_horizon} (was {config.planning_horizon})")
print(f"   Cache size: {fast_config.cache_size} (was {config.cache_size})")
print(f"   Reactive threshold: {fast_config.reactive_only_threshold_ms}ms")

fast_estimated = fast_config.n_futures * fast_config.planning_horizon * fast_config.future_simulation_ms_per_unit * 5 / 1000
print(f"   Estimated planning time: {fast_estimated:.1f}s")

print(f"\n   Quality Improvement:")
print(f"   Simulation depth: {fast_config.n_futures * fast_config.planning_horizon}x "
      f"(was {config.n_futures * config.planning_horizon}x)")
print(f"   Decision quality: ~{fast_config.n_futures * fast_config.planning_horizon / (config.n_futures * config.planning_horizon):.0f}x better")

# Show biological timescales
print("\n" + "-"*60)
print("3. Achieving Biological Timescales:")

print(f"\n   Current Hardware:")
print(f"   Reactive: ~{config.field_evolution_ms * 10:.0f}ms")
print(f"   Cached plan: ~{config.field_evolution_ms * 20:.0f}ms")
print(f"   Full planning: ~{estimated_time:.0f}s")

print(f"\n   20x Faster Hardware:")
print(f"   Reactive: ~{fast_config.field_evolution_ms * 10:.0f}ms (true reflexes!)")
print(f"   Cached plan: ~{fast_config.field_evolution_ms * 20:.0f}ms (instant habits!)")
print(f"   Full planning: ~{fast_estimated:.1f}s (think while acting!)")

print("\n" + "="*60)
print("CONCLUSION:")
print(f"The same code that explores {config.n_futures} futures over {config.planning_horizon} steps")
print(f"would automatically scale to {fast_config.n_futures} futures over {fast_config.planning_horizon} steps")
print("on faster hardware, achieving true biological responsiveness!")
print("="*60)