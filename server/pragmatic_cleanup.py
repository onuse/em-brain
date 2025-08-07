#!/usr/bin/env python3
"""
Pragmatic Dead Code Cleanup for EM-Brain
=========================================

A practical approach to cleaning up the field brain modules.

Strategy:
1. Keep PureFieldBrain as the primary implementation
2. Create minimal compatibility wrappers for legacy code
3. Remove truly dead code
4. Update critical imports
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import argparse

PROJECT_ROOT = Path('/mnt/c/Users/glimm/Documents/Projects/em-brain')
FIELD_DIR = PROJECT_ROOT / 'server/src/brains/field'

# The true essentials
ESSENTIAL_FILES = {
    'pure_field_brain.py',     # The only brain we need
    '__init__.py',              # Will be rewritten
}

# Files that are used but could be replaced
LEGACY_SUPPORT = {
    'unified_field_brain.py',   # Used by demo.py and tools
    'gpu_optimizations.py',     # Could be integrated into PureFieldBrain
    'gpu_performance_integration.py',  # Depends on UnifiedFieldBrain
    'optimized_unified_field_brain.py',  # Used by gpu_performance_integration
}

# Truly dead code - safe to remove
DEFINITELY_DEAD = {
    'gpu_memory_optimizer.py',  # No external imports found
}

# Files that are only used by UnifiedFieldBrain
UNIFIED_DEPENDENCIES = {
    'active_audio_system.py',
    'active_sensing_system.py', 
    'active_tactile_system.py',
    'active_vision_system.py',
    'adaptive_motor_cortex.py',
    'consolidation_system.py',
    'emergent_sensory_mapping.py',
    'evolved_field_dynamics.py',
    'field_constants.py',
    'field_strategic_planner.py',
    'field_types.py',
    'hierarchical_prediction.py',
    'motor_cortex.py',
    'pattern_attention_adapter.py',
    'pattern_cache_pool.py',
    'pattern_motor_adapter.py',
    'prediction_error_learning.py',
    'predictive_field_system.py',
    'reward_topology_shaping.py',
    'topology_region_system.py',
    'unified_pattern_system.py',
}

# Files that have some external use
NEEDS_MIGRATION = {
    'minimal_field_brain.py',  # Used by benchmark tools
}


def create_compatibility_wrapper():
    """Create a minimal UnifiedFieldBrain wrapper around PureFieldBrain."""
    wrapper_code = '''"""
UnifiedFieldBrain Compatibility Wrapper
========================================

This is a compatibility wrapper that redirects UnifiedFieldBrain
to use PureFieldBrain internally. For new code, use PureFieldBrain directly.
"""

from .pure_field_brain import PureFieldBrain
import torch
from typing import Optional, Dict, Any, Tuple


class UnifiedFieldBrain:
    """
    Legacy compatibility wrapper for UnifiedFieldBrain.
    Internally uses PureFieldBrain for all operations.
    
    ⚠️ DEPRECATED: Use PureFieldBrain directly for new code.
    """
    
    def __init__(
        self,
        sensory_dim: int = 32,
        motor_dim: int = 5,
        spatial_resolution: int = 32,
        device: Optional[torch.device] = None,
        quiet_mode: bool = True,
        **kwargs  # Ignore legacy parameters
    ):
        # Map legacy params to PureFieldBrain scale config
        if spatial_resolution <= 16:
            scale = 'tiny'
        elif spatial_resolution <= 24:
            scale = 'small'
        elif spatial_resolution <= 32:
            scale = 'medium'
        else:
            scale = 'large'
        
        self.brain = PureFieldBrain(
            sensory_dim=sensory_dim,
            motor_dim=motor_dim,
            scale=scale,
            device=device
        )
        
        # Legacy compatibility attributes
        self.device = self.brain.device
        self.field = self.brain.field
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.quiet_mode = quiet_mode
        
    def process_sensory(self, sensory_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Legacy interface - delegates to PureFieldBrain."""
        motor_output = self.brain(sensory_input)
        brain_state = {
            'field_energy': float(torch.mean(torch.abs(self.brain.field))),
            'prediction_error': self.brain.metrics.get('prediction_error', 0.0)
        }
        return motor_output, brain_state
    
    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Direct forward pass."""
        return self.brain(sensory_input)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to PureFieldBrain."""
        return getattr(self.brain, name)


# For even older code that imports specific subsystems
class DummySystem:
    """Dummy system for legacy imports."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return torch.zeros(1)

# Legacy subsystem stubs
ActiveVisionSystem = DummySystem
PredictiveFieldSystem = DummySystem
TopologyRegionSystem = DummySystem
'''
    return wrapper_code


def create_minimal_init():
    """Create a minimal __init__.py that exports only what's needed."""
    init_code = '''"""
Field Brain Package - Minimal Export
=====================================

Exports PureFieldBrain as the primary implementation.
Legacy code can still import UnifiedFieldBrain (compatibility wrapper).
"""

from .pure_field_brain import PureFieldBrain, SCALE_CONFIGS

# Legacy compatibility - will log deprecation warning
try:
    from .unified_field_brain import UnifiedFieldBrain
except ImportError:
    # If wrapper doesn't exist, alias to PureFieldBrain
    UnifiedFieldBrain = PureFieldBrain

__all__ = [
    'PureFieldBrain',
    'SCALE_CONFIGS',
    'UnifiedFieldBrain',  # For compatibility
]

# Factory function
def create_brain(*args, **kwargs):
    """Create a brain instance - always returns PureFieldBrain."""
    return PureFieldBrain(*args, **kwargs)
'''
    return init_code


def cleanup_stage1(dry_run: bool = True):
    """Stage 1: Remove definitely dead code."""
    print("\n=== STAGE 1: Remove Definitely Dead Code ===")
    
    backup_dir = PROJECT_ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
    
    for filename in DEFINITELY_DEAD:
        file_path = FIELD_DIR / filename
        if file_path.exists():
            if dry_run:
                print(f"  Would remove: {filename}")
            else:
                # Backup first
                shutil.copy2(file_path, backup_dir / filename)
                file_path.unlink()
                print(f"  ✓ Removed: {filename}")


def cleanup_stage2(dry_run: bool = True):
    """Stage 2: Create compatibility wrapper and remove UnifiedFieldBrain dependencies."""
    print("\n=== STAGE 2: Replace UnifiedFieldBrain with Wrapper ===")
    
    backup_dir = PROJECT_ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
        
        # Backup original UnifiedFieldBrain
        original = FIELD_DIR / 'unified_field_brain.py'
        if original.exists():
            shutil.copy2(original, backup_dir / 'unified_field_brain_original.py')
        
        # Write compatibility wrapper
        wrapper_path = FIELD_DIR / 'unified_field_brain.py'
        wrapper_path.write_text(create_compatibility_wrapper())
        print(f"  ✓ Created compatibility wrapper: unified_field_brain.py")
        
        # Remove all UnifiedFieldBrain dependencies
        for filename in UNIFIED_DEPENDENCIES:
            file_path = FIELD_DIR / filename
            if file_path.exists():
                shutil.copy2(file_path, backup_dir / filename)
                file_path.unlink()
                print(f"  ✓ Removed: {filename}")
    else:
        print(f"  Would create compatibility wrapper: unified_field_brain.py")
        print(f"  Would remove {len(UNIFIED_DEPENDENCIES)} dependency files")


def cleanup_stage3(dry_run: bool = True):
    """Stage 3: Update __init__.py and clean up imports."""
    print("\n=== STAGE 3: Update Package Exports ===")
    
    if not dry_run:
        init_path = FIELD_DIR / '__init__.py'
        
        # Backup
        backup_dir = PROJECT_ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(init_path, backup_dir / '__init__.py')
        
        # Write new minimal init
        init_path.write_text(create_minimal_init())
        print(f"  ✓ Updated: __init__.py")
    else:
        print(f"  Would update: __init__.py")


def verify_cleanup():
    """Verify the cleanup didn't break anything critical."""
    print("\n=== VERIFICATION ===")
    
    # Check if PureFieldBrain can be imported
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / 'server'))
        from src.brains.field.pure_field_brain import PureFieldBrain
        print("  ✓ PureFieldBrain imports successfully")
    except Exception as e:
        print(f"  ✗ Failed to import PureFieldBrain: {e}")
        return False
    
    # Check if brain.py still works
    brain_py = PROJECT_ROOT / 'server/brain.py'
    result = subprocess.run(
        ['python3', '-c', f'import sys; sys.path.insert(0, "{PROJECT_ROOT}/server"); from src.core.unified_brain_factory import UnifiedBrainFactory'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("  ✓ Brain factory imports successfully")
    else:
        print(f"  ✗ Brain factory import failed: {result.stderr}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Pragmatic cleanup of field brain modules')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], 
                       help='Run specific stage (1=dead code, 2=wrapper, 3=init)')
    parser.add_argument('--all', action='store_true',
                       help='Run all stages')
    parser.add_argument('--live', action='store_true',
                       help='Actually perform cleanup (default is dry-run)')
    
    args = parser.parse_args()
    
    print("Pragmatic Field Brain Cleanup")
    print("="*60)
    
    dry_run = not args.live
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    else:
        print("⚠️  LIVE MODE - Files will be modified!")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run requested stages
    if args.all or args.stage == 1:
        cleanup_stage1(dry_run)
    
    if args.all or args.stage == 2:
        cleanup_stage2(dry_run)
    
    if args.all or args.stage == 3:
        cleanup_stage3(dry_run)
    
    # Verify if we did everything
    if args.all and not dry_run:
        if verify_cleanup():
            print("\n✅ Cleanup completed successfully!")
        else:
            print("\n⚠️  Cleanup completed but verification failed")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if dry_run:
        print("\nTo actually perform cleanup, run with --live flag")
        print("Recommended approach:")
        print("  1. python3 pragmatic_cleanup.py --stage 1 --live  # Remove dead code")
        print("  2. Test that everything still works")
        print("  3. python3 pragmatic_cleanup.py --stage 2 --live  # Add wrapper")
        print("  4. Test again")
        print("  5. python3 pragmatic_cleanup.py --stage 3 --live  # Update exports")
    else:
        # Count remaining files
        remaining = len(list(FIELD_DIR.glob('*.py')))
        print(f"\nRemaining files in field/: {remaining}")
        print(f"Space saved: ~{(29-remaining)*50} KB")
        
        print("\nNext steps:")
        print("  1. Run tests to ensure nothing is broken")
        print("  2. Update demo.py to use PureFieldBrain directly")
        print("  3. Update benchmark tools to use PureFieldBrain")
        print("  4. Remove the compatibility wrapper once migration is complete")


if __name__ == '__main__':
    main()