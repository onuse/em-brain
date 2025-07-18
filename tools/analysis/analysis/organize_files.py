#!/usr/bin/env python3
"""
File Organization Script

This script moves files from the root folder to their proper locations
according to the CLAUDE.md project structure guidelines.
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """Organize files according to project structure."""
    root = Path(".")
    
    # Files that should stay in root
    allowed_root_files = {
        'CLAUDE.md',
        'README.md',
        'demo.py',
        'demo_runner.py',
        'test_runner.py',
        'validation_runner.py',
        '.gitignore'
    }
    
    # Allowed root directories
    allowed_root_dirs = {
        'demos',
        'docs',
        'server',
        'validation',
        'client_picarx',
        'logs',
        'robot_memory',
        'archive',
        'micro_experiments'
    }
    
    # Create analysis and experiments directories if they don't exist
    analysis_dir = root / 'server' / 'tools' / 'analysis'
    experiments_dir = root / 'server' / 'tools' / 'experiments'
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Categorize files for movement
    moves = []
    
    # Get all files in root
    for item in root.iterdir():
        if item.is_file() and item.name not in allowed_root_files:
            # Determine where to move the file
            if 'profiler' in item.name or 'performance' in item.name or 'bottleneck' in item.name:
                # Performance analysis files
                target = analysis_dir / item.name
                moves.append((item, target))
            elif 'test_' in item.name:
                # Test files
                target = root / 'server' / 'tests' / item.name
                moves.append((item, target))
            elif 'statistics' in item.name:
                # Statistics files
                target = root / 'server' / 'tools' / 'analysis' / item.name
                moves.append((item, target))
            elif 'debug' in item.name or 'trace' in item.name:
                # Debug files
                target = analysis_dir / item.name
                moves.append((item, target))
            elif 'experiment' in item.name or 'quick_' in item.name:
                # Experiment files
                target = experiments_dir / item.name
                moves.append((item, target))
            else:
                # Other files go to analysis
                target = analysis_dir / item.name
                moves.append((item, target))
        
        elif item.is_dir() and item.name not in allowed_root_dirs:
            # Move directories to appropriate locations
            if item.name == 'verification_results':
                target = root / 'validation' / item.name
                moves.append((item, target))
            else:
                # Other directories go to tools
                target = root / 'server' / 'tools' / item.name
                moves.append((item, target))
    
    # Show what will be moved
    print("📁 FILE ORGANIZATION PLAN")
    print("=" * 30)
    
    if not moves:
        print("✅ Root folder is already organized!")
        return
    
    print(f"Found {len(moves)} items to move:")
    for source, target in moves:
        print(f"  {source.name} → {target.relative_to(root)}")
    
    # Ask for confirmation
    response = input(f"\nProceed with moving {len(moves)} items? (y/N): ").lower()
    if response != 'y':
        print("❌ Organization cancelled")
        return
    
    # Perform the moves
    print("\n🚀 MOVING FILES...")
    success_count = 0
    
    for source, target in moves:
        try:
            # Create parent directory if it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file/directory
            shutil.move(str(source), str(target))
            print(f"✅ Moved {source.name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to move {source.name}: {e}")
    
    print(f"\n📊 ORGANIZATION COMPLETE")
    print(f"✅ Successfully moved {success_count}/{len(moves)} items")
    
    # Show final root structure
    print(f"\n📁 FINAL ROOT STRUCTURE:")
    for item in sorted(root.iterdir()):
        if item.is_file():
            print(f"  📄 {item.name}")
        else:
            print(f"  📁 {item.name}/")

if __name__ == "__main__":
    organize_files()