#!/usr/bin/env python3
"""Debug tool to help find numpy conversion errors."""

import functools
import traceback

def debug_numpy_operations(func):
    """Decorator to catch and log numpy conversion errors with context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "can't convert" in str(e) and "numpy" in str(e):
                print(f"\n{'='*60}")
                print(f"NUMPY ERROR in {func.__module__}.{func.__name__}")
                print(f"{'='*60}")
                print(f"Error: {e}")
                print("\nFunction details:")
                print(f"  Module: {func.__module__}")
                print(f"  Function: {func.__name__}")
                if hasattr(func, '__qualname__'):
                    print(f"  Qualified name: {func.__qualname__}")
                print("\nStack trace:")
                traceback.print_exc()
                print(f"{'='*60}\n")
            raise
    return wrapper

# Monkey patch key methods that might have numpy issues
def patch_brain_for_debugging():
    """Patch the brain classes to add numpy debugging."""
    try:
        from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
        from src.brains.field.topology_region_system import TopologyRegionSystem
        from src.brains.field.evolved_field_dynamics import EvolvedFieldDynamics
        from src.brains.field.consolidation_system import ConsolidationSystem
        
        # List of methods to patch
        methods_to_patch = [
            # TopologyRegionSystem methods (runs every 5 cycles)
            (TopologyRegionSystem, 'detect_topology_regions'),
            (TopologyRegionSystem, '_create_topology_region'),
            (TopologyRegionSystem, '_compute_region_statistics'),
            
            # EvolvedFieldDynamics methods
            (EvolvedFieldDynamics, 'evolve_field'),
            (EvolvedFieldDynamics, 'compute_field_state'),
            (EvolvedFieldDynamics, 'compute_field_modulation'),
            
            # ConsolidationSystem methods (might run periodically)
            (ConsolidationSystem, 'consolidate_memories'),
            (ConsolidationSystem, '_identify_strong_patterns'),
        ]
        
        # Apply patches
        for cls, method_name in methods_to_patch:
            if hasattr(cls, method_name):
                original = getattr(cls, method_name)
                patched = debug_numpy_operations(original)
                setattr(cls, method_name, patched)
                print(f"✓ Patched {cls.__name__}.{method_name}")
        
        print("\nDebugging patches applied successfully!")
        
    except Exception as e:
        print(f"Error applying patches: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
    
    patch_brain_for_debugging()
    
    # Now run a simple test
    print("\nRunning test with debug patches...")
    from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    # Run enough cycles to trigger the error
    for i in range(20):
        sensory_input = [0.5] * 24
        try:
            motor_output, brain_state = brain.process_robot_cycle(sensory_input)
            if i % 5 == 0:
                print(f"Cycle {i}: OK")
        except Exception as e:
            print(f"Cycle {i}: Error - {e}")
            if "can't convert" in str(e) and "numpy" in str(e):
                break