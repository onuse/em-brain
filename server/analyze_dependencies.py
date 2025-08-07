#!/usr/bin/env python3
"""
Dependency Analysis for Field Brain Modules
============================================

Analyzes which field brain modules are actually in use and which are safe to remove.
"""

import subprocess
from pathlib import Path
from typing import Dict, Set, List
import json

PROJECT_ROOT = Path('/mnt/c/Users/glimm/Documents/Projects/em-brain')

# Core files we want to keep
CORE_FILES = {
    'pure_field_brain.py',
    'gpu_optimizations.py', 
    'gpu_performance_integration.py',
    '__init__.py'
}

# All field brain files
FIELD_DIR = PROJECT_ROOT / 'server/src/brains/field'
ALL_FIELD_FILES = set(f.name for f in FIELD_DIR.glob('*.py'))

def find_importers(module_name: str, exclude_tests: bool = False) -> Set[str]:
    """Find all files that import a module."""
    importers = set()
    
    # Search for imports
    patterns = [
        f"from.*{module_name} import",
        f"from.*\\.{module_name} import",
        f"import.*{module_name}"
    ]
    
    for pattern in patterns:
        # Try ripgrep first, fall back to grep
        try:
            result = subprocess.run(
                ['rg', '-l', pattern, '--type', 'py'],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            # Use grep as fallback
            result = subprocess.run(
                ['grep', '-r', '-l', f'{module_name}', '--include=*.py', '.'],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    # Convert to relative path
                    path = Path(line)
                    rel_path = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
                    
                    # Skip test files if requested
                    if exclude_tests and 'test' in str(rel_path).lower():
                        continue
                    
                    # Skip the module itself
                    if path.name == f"{module_name}.py":
                        continue
                        
                    # Skip cleanup/analysis scripts
                    if 'cleanup' in str(rel_path) or 'analyze' in str(rel_path):
                        continue
                    
                    importers.add(str(rel_path))
    
    return importers

def analyze_dependencies():
    """Analyze dependencies for all field brain modules."""
    
    dependencies = {}
    
    for file_path in FIELD_DIR.glob('*.py'):
        if file_path.name == '__init__.py':
            continue
            
        module_name = file_path.stem
        importers = find_importers(module_name, exclude_tests=True)
        
        # Filter out field brain internal imports
        external_importers = []
        field_importers = []
        
        for imp in importers:
            if 'server/src/brains/field/' in imp:
                # Internal field brain import
                field_importers.append(imp)
            else:
                # External import
                external_importers.append(imp)
        
        dependencies[file_path.name] = {
            'module': module_name,
            'external_importers': external_importers,
            'field_importers': field_importers,
            'total_imports': len(external_importers) + len(field_importers),
            'is_core': file_path.name in CORE_FILES
        }
    
    return dependencies

def build_dependency_graph(dependencies: Dict) -> Dict[str, Set[str]]:
    """Build a dependency graph within field brain modules."""
    graph = {}
    
    for file_name, info in dependencies.items():
        graph[file_name] = set()
        
        for imp in info['field_importers']:
            # Extract the imported file name
            imp_file = Path(imp).name
            if imp_file in dependencies:
                graph[file_name].add(imp_file)
    
    return graph

def find_required_files(graph: Dict[str, Set[str]], start_files: Set[str]) -> Set[str]:
    """Find all files required by the starting set (transitive closure)."""
    required = set(start_files)
    to_check = list(start_files)
    
    while to_check:
        current = to_check.pop()
        
        # Find what this file imports
        for file_name, imports in graph.items():
            if current in imports and file_name not in required:
                required.add(file_name)
                to_check.append(file_name)
    
    return required

def main():
    print("Analyzing Field Brain Dependencies")
    print("="*60)
    
    # Analyze all dependencies
    deps = analyze_dependencies()
    
    # Build internal dependency graph
    graph = build_dependency_graph(deps)
    
    # Find what's required for core files
    required_for_pure = find_required_files(graph, CORE_FILES)
    
    # Categorize files
    categories = {
        'core': [],
        'required_by_core': [],
        'used_externally': [],
        'dead': []
    }
    
    for file_name, info in sorted(deps.items()):
        if info['is_core']:
            categories['core'].append(file_name)
        elif file_name in required_for_pure:
            categories['required_by_core'].append(file_name)
        elif info['external_importers']:
            categories['used_externally'].append(file_name)
        else:
            categories['dead'].append(file_name)
    
    # Print results
    print("\n🔵 CORE FILES (Keep):")
    for f in categories['core']:
        print(f"  ✓ {f}")
    
    print(f"\n🟡 REQUIRED BY CORE ({len(categories['required_by_core'])} files):")
    for f in categories['required_by_core']:
        info = deps[f]
        print(f"  • {f}")
        if info['field_importers']:
            for imp in info['field_importers'][:3]:
                print(f"    ← {Path(imp).name}")
    
    print(f"\n🟠 USED EXTERNALLY ({len(categories['used_externally'])} files):")
    for f in categories['used_externally']:
        info = deps[f]
        print(f"  • {f} ({len(info['external_importers'])} external imports)")
        for imp in info['external_importers'][:3]:
            print(f"    ← {imp}")
        if len(info['external_importers']) > 3:
            print(f"    ... and {len(info['external_importers'])-3} more")
    
    print(f"\n🔴 DEAD CODE ({len(categories['dead'])} files - safe to remove):")
    for f in categories['dead']:
        print(f"  ✗ {f}")
    
    # Special cases that need attention
    print("\n⚠️  SPECIAL CASES:")
    
    # Check if UnifiedFieldBrain is used
    if 'unified_field_brain.py' in deps:
        ufi_info = deps['unified_field_brain.py']
        if ufi_info['external_importers']:
            print(f"\n  UnifiedFieldBrain is still used by:")
            for imp in ufi_info['external_importers'][:5]:
                print(f"    - {imp}")
            print("\n  Options:")
            print("    1. Update these files to use PureFieldBrain")
            print("    2. Keep UnifiedFieldBrain as legacy support")
            print("    3. Create a compatibility wrapper")
    
    # Check GPU integration files
    gpu_files = ['gpu_performance_integration.py', 'gpu_optimizations.py']
    for gf in gpu_files:
        if gf in deps:
            info = deps[gf]
            if info['field_importers']:
                print(f"\n  {gf} imports these field modules:")
                for imp in info['field_importers']:
                    print(f"    - {Path(imp).name}")
    
    # Save analysis to file
    output_file = PROJECT_ROOT / 'server/field_dependency_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'categories': categories,
            'dependencies': {k: {**v, 'external_importers': v['external_importers'][:10]} 
                           for k, v in deps.items()},
            'recommendation': {
                'safe_to_remove': categories['dead'],
                'need_migration': [f for f in categories['used_externally'] 
                                 if f not in CORE_FILES],
                'keep': list(CORE_FILES) + categories['required_by_core']
            }
        }, f, indent=2)
    
    print(f"\n📄 Full analysis saved to: {output_file}")
    
    # Summary
    total = len(ALL_FIELD_FILES)
    dead = len(categories['dead'])
    print(f"\n" + "="*60)
    print(f"SUMMARY: {dead}/{total} files are dead code ({dead*100//total}%)")
    print(f"Keep: {len(categories['core']) + len(categories['required_by_core'])} files")
    print(f"Need migration: {len(categories['used_externally'])} files")

if __name__ == '__main__':
    main()