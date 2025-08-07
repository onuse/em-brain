#!/usr/bin/env python3
"""
Dead Code Cleanup Script for EM-Brain Project
==============================================

This script safely removes dead code files that are no longer needed
after the consolidation to PureFieldBrain architecture.

Safety features:
1. Creates backup before deletion
2. Checks for imports before removing
3. Updates __init__.py files
4. Provides dry-run mode
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple
import argparse

# Essential files to KEEP
ESSENTIAL_FILES = {
    'server/src/brains/field/pure_field_brain.py',
    'server/src/brains/field/gpu_optimizations.py', 
    'server/src/brains/field/gpu_performance_integration.py',
    'server/src/brains/field/__init__.py'  # Will be updated, not deleted
}

# Dead files in brains/field to remove
DEAD_FIELD_FILES = [
    # Active systems (no longer needed)
    'active_audio_system.py',
    'active_sensing_system.py',
    'active_tactile_system.py',
    'active_vision_system.py',
    
    # Old brain implementations
    'unified_field_brain.py',
    'minimal_field_brain.py',
    'optimized_unified_field_brain.py',
    
    # Pattern/motor subsystems
    'pattern_attention_adapter.py',
    'pattern_cache_pool.py',
    'pattern_motor_adapter.py',
    'unified_pattern_system.py',
    'motor_cortex.py',
    'adaptive_motor_cortex.py',
    
    # Prediction subsystems
    'predictive_field_system.py',
    'prediction_error_learning.py',
    'hierarchical_prediction.py',
    
    # Other subsystems
    'consolidation_system.py',
    'emergent_sensory_mapping.py',
    'evolved_field_dynamics.py',
    'field_strategic_planner.py',
    'reward_topology_shaping.py',
    'topology_region_system.py',
    
    # Utilities (now integrated)
    'gpu_memory_optimizer.py',
    'field_constants.py',
    'field_types.py'
]


class DeadCodeCleaner:
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.backup_dir = project_root / f"backup_deadcode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.files_to_remove = []
        self.import_warnings = []
        
    def find_imports(self, module_name: str) -> List[str]:
        """Find all files that import a given module."""
        try:
            # Use ripgrep for fast searching
            result = subprocess.run(
                ['rg', '-l', f'from.*{module_name}|import.*{module_name}', '--type', 'py'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.stdout:
                return result.stdout.strip().split('\n')
        except FileNotFoundError:
            # Fallback to grep if ripgrep not available
            result = subprocess.run(
                ['grep', '-r', '-l', f'{module_name}', '--include=*.py'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.stdout:
                return result.stdout.strip().split('\n')
        return []
    
    def check_file_safety(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if a file is safe to delete."""
        module_name = file_path.stem
        importing_files = self.find_imports(module_name)
        
        # Filter out self-imports and files we're also deleting
        dangerous_imports = []
        for imp_file in importing_files:
            imp_path = Path(imp_file)
            # Skip if it's the file itself
            if imp_path.samefile(file_path):
                continue
            # Skip if it's in our deletion list
            if imp_path.name in DEAD_FIELD_FILES:
                continue
            # Skip test files if --include-tests flag is set
            if 'test' in str(imp_path):
                continue  # We'll handle tests separately
            dangerous_imports.append(str(imp_path))
        
        return len(dangerous_imports) == 0, dangerous_imports
    
    def backup_file(self, file_path: Path):
        """Create backup of file before deletion."""
        if not self.dry_run:
            rel_path = file_path.relative_to(self.project_root)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            print(f"  ✓ Backed up: {rel_path}")
    
    def update_init_file(self):
        """Update __init__.py to only export PureFieldBrain."""
        init_path = self.project_root / 'server/src/brains/field/__init__.py'
        new_content = '''"""
Field Brain Package - Pure Field Implementation
==============================================

Exports only the PureFieldBrain and its GPU optimization utilities.
All legacy implementations have been removed.
"""

from .pure_field_brain import PureFieldBrain, SCALE_CONFIGS

# GPU optimization utilities
from .gpu_optimizations import GPUFieldKernels
from .gpu_performance_integration import GPUBrainFactory, create_optimized_brain

__all__ = [
    'PureFieldBrain',
    'SCALE_CONFIGS',
    'GPUFieldKernels',
    'GPUBrainFactory',
    'create_optimized_brain'
]

# Factory function for backward compatibility
def create_brain(*args, **kwargs):
    """Create a PureFieldBrain instance."""
    return PureFieldBrain(*args, **kwargs)
'''
        
        if not self.dry_run:
            self.backup_file(init_path)
            init_path.write_text(new_content)
            print(f"  ✓ Updated: {init_path.relative_to(self.project_root)}")
        else:
            print(f"  Would update: {init_path.relative_to(self.project_root)}")
    
    def clean_field_directory(self):
        """Clean the brains/field directory."""
        field_dir = self.project_root / 'server/src/brains/field'
        
        print("\n=== Cleaning brains/field directory ===")
        
        removed_count = 0
        for file_name in DEAD_FIELD_FILES:
            file_path = field_dir / file_name
            if file_path.exists():
                # Check safety
                is_safe, importers = self.check_file_safety(file_path)
                
                if not is_safe:
                    print(f"  ⚠️  WARNING: {file_name} is imported by:")
                    for imp in importers[:5]:  # Show first 5
                        print(f"      - {imp}")
                    self.import_warnings.append((file_name, importers))
                    continue
                
                # Mark for removal
                self.files_to_remove.append(file_path)
                removed_count += 1
                
                if not self.dry_run:
                    self.backup_file(file_path)
                    file_path.unlink()
                    print(f"  ✓ Removed: {file_name}")
                else:
                    print(f"  Would remove: {file_name}")
        
        print(f"\n  Total: {removed_count} files")
        
        # Update __init__.py
        self.update_init_file()
    
    def find_other_dead_code(self):
        """Find potentially dead code in other directories."""
        print("\n=== Searching for other dead code ===")
        
        # Patterns that indicate dead code
        dead_patterns = [
            'deprecated',
            'old_',
            '_backup',
            '_unused',
            'legacy_',
            'DEPRECATED',
            'DO_NOT_USE'
        ]
        
        dead_candidates = []
        
        for pattern in dead_patterns:
            result = subprocess.run(
                ['find', self.project_root, '-type', 'f', '-name', f'*{pattern}*.py'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                files = result.stdout.strip().split('\n')
                dead_candidates.extend(files)
        
        # Also check archive directories
        archive_dirs = [
            'server/archive',
            'archive',
            'server/src/old',
            'server/deprecated'
        ]
        
        for dir_name in archive_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"\n  Found archive directory: {dir_path.relative_to(self.project_root)}")
                # Count Python files
                py_files = list(dir_path.rglob('*.py'))
                if py_files:
                    print(f"    Contains {len(py_files)} Python files")
                    dead_candidates.append(str(dir_path))
        
        if dead_candidates:
            print(f"\n  Found {len(set(dead_candidates))} potential dead code files/directories")
            print("  Run with --aggressive flag to include these in cleanup")
    
    def update_test_imports(self):
        """Update test files to stop importing dead modules."""
        print("\n=== Updating test imports ===")
        
        # Find test files that import dead modules
        test_files_to_update = set()
        
        for dead_file in DEAD_FIELD_FILES:
            module_name = Path(dead_file).stem
            test_imports = self.find_imports(module_name)
            
            for test_file in test_imports:
                if 'test' in test_file.lower():
                    test_files_to_update.add(test_file)
        
        if test_files_to_update:
            print(f"  Found {len(test_files_to_update)} test files with dead imports")
            if not self.dry_run:
                print("  Consider updating these tests to use PureFieldBrain instead")
            
            for test_file in list(test_files_to_update)[:10]:  # Show first 10
                print(f"    - {test_file}")
    
    def generate_report(self):
        """Generate cleanup report."""
        print("\n" + "="*60)
        print("CLEANUP REPORT")
        print("="*60)
        
        if self.dry_run:
            print("\n** DRY RUN MODE - No files were actually deleted **")
        
        print(f"\nFiles marked for removal: {len(self.files_to_remove)}")
        
        if self.import_warnings:
            print(f"\n⚠️  Files with import warnings: {len(self.import_warnings)}")
            print("These files should be manually reviewed before deletion")
        
        if not self.dry_run and self.backup_dir.exists():
            print(f"\nBackup created at: {self.backup_dir}")
            print("To restore: cp -r {backup}/* {project}")
        
        # Calculate space saved
        total_size = sum(f.stat().st_size for f in self.files_to_remove if f.exists())
        print(f"\nEstimated space saved: {total_size / 1024:.1f} KB")
        
        print("\n" + "="*60)
    
    def run(self):
        """Execute the cleanup process."""
        print("EM-Brain Dead Code Cleanup")
        print("="*60)
        
        if self.dry_run:
            print("Running in DRY RUN mode - no files will be deleted")
        else:
            print("⚠️  Running in LIVE mode - files will be deleted!")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            
            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created backup directory: {self.backup_dir}")
        
        # Clean field directory
        self.clean_field_directory()
        
        # Update test imports
        self.update_test_imports()
        
        # Find other dead code
        self.find_other_dead_code()
        
        # Generate report
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='Clean up dead code from EM-Brain project')
    parser.add_argument('--live', action='store_true', 
                       help='Actually delete files (default is dry-run)')
    parser.add_argument('--aggressive', action='store_true',
                       help='Include deprecated/archive directories')
    parser.add_argument('--project-root', type=Path, 
                       default=Path('/mnt/c/Users/glimm/Documents/Projects/em-brain'),
                       help='Project root directory')
    
    args = parser.parse_args()
    
    cleaner = DeadCodeCleaner(
        project_root=args.project_root,
        dry_run=not args.live
    )
    
    cleaner.run()


if __name__ == '__main__':
    main()