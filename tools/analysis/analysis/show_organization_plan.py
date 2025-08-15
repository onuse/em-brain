#!/usr/bin/env python3
"""
Show File Organization Plan

This script shows what files should be moved to organize the root folder
according to the CLAUDE.md project structure guidelines.
"""

import os
from pathlib import Path

def show_organization_plan():
    """Show organization plan for root folder."""
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
    
    # Categorize files for movement
    analysis_files = []
    test_files = []
    experiment_files = []
    config_files = []
    other_files = []
    directories = []
    
    # Get all files in root
    for item in root.iterdir():
        if item.is_file() and item.name not in allowed_root_files:
            # Determine where to move the file
            if 'profiler' in item.name or 'performance' in item.name or 'bottleneck' in item.name:
                analysis_files.append(item.name)
            elif 'test_' in item.name:
                test_files.append(item.name)
            elif 'statistics' in item.name:
                config_files.append(item.name)
            elif 'debug' in item.name or 'trace' in item.name:
                analysis_files.append(item.name)
            elif 'experiment' in item.name or 'quick_' in item.name:
                experiment_files.append(item.name)
            else:
                other_files.append(item.name)
        
        elif item.is_dir() and item.name not in allowed_root_dirs:
            directories.append(item.name)
    
    # Show organization plan
    print("📁 ROOT FOLDER ORGANIZATION PLAN")
    print("=" * 40)
    
    print(f"\n✅ SHOULD STAY IN ROOT:")
    for item in sorted(root.iterdir()):
        if item.is_file() and item.name in allowed_root_files:
            print(f"  📄 {item.name}")
        elif item.is_dir() and item.name in allowed_root_dirs:
            print(f"  📁 {item.name}/")
    
    total_to_move = len(analysis_files) + len(test_files) + len(experiment_files) + len(config_files) + len(other_files) + len(directories)
    
    if total_to_move == 0:
        print(f"\n🎉 ROOT FOLDER IS ALREADY ORGANIZED!")
        return
    
    print(f"\n📋 FILES TO MOVE ({total_to_move} total):")
    
    if analysis_files:
        print(f"\n  📊 Analysis files → server/tools/analysis/")
        for file in sorted(analysis_files):
            print(f"    • {file}")
    
    if test_files:
        print(f"\n  🧪 Test files → server/tests/")
        for file in sorted(test_files):
            print(f"    • {file}")
    
    if experiment_files:
        print(f"\n  🔬 Experiment files → server/tools/experiments/")
        for file in sorted(experiment_files):
            print(f"    • {file}")
    
    if config_files:
        print(f"\n  ⚙️  Configuration files → server/tools/analysis/")
        for file in sorted(config_files):
            print(f"    • {file}")
    
    if other_files:
        print(f"\n  📝 Other files → server/tools/analysis/")
        for file in sorted(other_files):
            print(f"    • {file}")
    
    if directories:
        print(f"\n  📁 Directories:")
        for dir_name in sorted(directories):
            if dir_name == 'verification_results':
                print(f"    • {dir_name}/ → validation/")
            else:
                print(f"    • {dir_name}/ → server/tools/")
    
    print(f"\n📝 ORGANIZATION COMMANDS:")
    print("=" * 25)
    
    print("# Create target directories")
    print("mkdir -p server/tools/analysis")
    print("mkdir -p server/tools/experiments")
    print("")
    
    # Generate move commands
    if analysis_files:
        print("# Move analysis files")
        for file in sorted(analysis_files):
            print(f"mv {file} server/tools/analysis/")
    
    if test_files:
        print("\n# Move test files")
        for file in sorted(test_files):
            print(f"mv {file} server/tests/")
    
    if experiment_files:
        print("\n# Move experiment files")
        for file in sorted(experiment_files):
            print(f"mv {file} server/tools/experiments/")
    
    if config_files:
        print("\n# Move configuration files")
        for file in sorted(config_files):
            print(f"mv {file} server/tools/analysis/")
    
    if other_files:
        print("\n# Move other files")
        for file in sorted(other_files):
            print(f"mv {file} server/tools/analysis/")
    
    if directories:
        print("\n# Move directories")
        for dir_name in sorted(directories):
            if dir_name == 'verification_results':
                print(f"mv {dir_name} validation/")
            else:
                print(f"mv {dir_name} server/tools/")

if __name__ == "__main__":
    show_organization_plan()