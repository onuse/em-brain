#!/usr/bin/env python3
"""
Inventory of server/src files to identify what's in use vs orphaned.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import ast


def find_all_py_files(src_dir):
    """Find all Python files in server/src."""
    py_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files


def extract_imports(file_path):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                for alias in node.names:
                    if node.module:
                        imports.add(f"{node.module}.{alias.name}")
    except:
        # If AST parsing fails, try regex
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            # Find import statements
            import_pattern = r'(?:from\\s+([\\w\\.]+)\\s+import|import\\s+([\\w\\.]+))'
            for match in re.finditer(import_pattern, content):
                if match.group(1):
                    imports.add(match.group(1))
                if match.group(2):
                    imports.add(match.group(2))
        except:
            pass
    
    return imports


def analyze_usage(base_dir):
    """Analyze which files are imported and by whom."""
    src_dir = os.path.join(base_dir, 'server', 'src')
    
    # Get all Python files
    all_files = find_all_py_files(src_dir)
    
    # Create mapping of file paths to module names
    file_to_module = {}
    for file_path in all_files:
        # Convert file path to module name
        rel_path = os.path.relpath(file_path, base_dir)
        module_name = rel_path.replace('/', '.').replace('\\', '.')[:-3]  # Remove .py
        module_name = module_name.replace('server.', '')  # Remove server prefix
        file_to_module[file_path] = module_name
    
    # Track imports
    imported_by = defaultdict(set)  # module -> set of files that import it
    imports_from = defaultdict(set)  # file -> set of modules it imports
    
    # Check all Python files in the project
    project_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip certain directories
        skip_dirs = ['__pycache__', '.git', 'logs', 'venv', 'env', '.pytest_cache']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                project_files.append(os.path.join(root, file))
    
    # Analyze imports
    for file_path in project_files:
        imports = extract_imports(file_path)
        imports_from[file_path] = imports
        
        for imp in imports:
            # Check if this import refers to any of our src modules
            for src_file, module_name in file_to_module.items():
                if imp.startswith(module_name) or imp.startswith('src.' + module_name):
                    imported_by[src_file].add(file_path)
    
    return all_files, imported_by, file_to_module


def categorize_files(all_files, imported_by, base_dir):
    """Categorize files by usage."""
    categories = {
        'actively_used': [],
        'test_only': [],
        'orphaned': [],
        'entry_points': [],
        'internal_only': []
    }
    
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, base_dir)
        importers = imported_by.get(file_path, set())
        
        # Special cases
        if '__init__.py' in file_path:
            if importers:
                categories['actively_used'].append((rel_path, len(importers)))
            continue
        
        if not importers:
            categories['orphaned'].append((rel_path, 0))
        else:
            # Check who imports this
            non_test_importers = [f for f in importers if 'test' not in f and 'archive' not in f and 'tools/analysis' not in f]
            test_importers = [f for f in importers if 'test' in f]
            
            if non_test_importers:
                if 'brain.py' in str(importers):
                    categories['entry_points'].append((rel_path, len(importers)))
                else:
                    categories['actively_used'].append((rel_path, len(importers)))
            elif test_importers:
                categories['test_only'].append((rel_path, len(test_importers)))
            else:
                # Only imported by archived or analysis tools
                categories['orphaned'].append((rel_path, 0))
    
    return categories


def analyze_file_purpose(file_path):
    """Try to determine what a file does from its docstring and imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Get module docstring
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        
        # Get key imports to understand dependencies
        imports = extract_imports(file_path)
        
        return docstring, imports
    except:
        return None, set()


def main():
    """Run the inventory analysis."""
    base_dir = Path(__file__).resolve().parents[2]  # robot-project/brain
    
    print("🔍 Analyzing server/src file usage...")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    
    all_files, imported_by, file_to_module = analyze_usage(base_dir)
    categories = categorize_files(all_files, imported_by, base_dir)
    
    # Print results
    print(f"\n📊 Total files in server/src: {len(all_files)}")
    print("=" * 80)
    
    print(f"\n✅ ACTIVELY USED ({len(categories['actively_used'])} files)")
    print("-" * 80)
    for file_path, import_count in sorted(categories['actively_used'], key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {file_path} (imported by {import_count} files)")
    
    print(f"\n🚀 ENTRY POINTS ({len(categories['entry_points'])} files)")
    print("-" * 80)
    for file_path, import_count in categories['entry_points']:
        print(f"  {file_path}")
    
    print(f"\n🧪 TEST ONLY ({len(categories['test_only'])} files)")
    print("-" * 80)
    for file_path, import_count in categories['test_only'][:10]:
        print(f"  {file_path}")
    
    print(f"\n❌ ORPHANED ({len(categories['orphaned'])} files)")
    print("-" * 80)
    
    # Group orphaned files by directory
    orphaned_by_dir = defaultdict(list)
    for file_path, _ in categories['orphaned']:
        dir_path = os.path.dirname(file_path)
        orphaned_by_dir[dir_path].append(os.path.basename(file_path))
    
    for dir_path, files in sorted(orphaned_by_dir.items()):
        print(f"\n  {dir_path}/")
        for file_name in sorted(files):
            # Get file purpose
            full_path = os.path.join(base_dir, dir_path, file_name)
            docstring, imports = analyze_file_purpose(full_path)
            if docstring:
                # First line of docstring
                first_line = docstring.split('\n')[0][:60]
                print(f"    {file_name:<40} # {first_line}")
            else:
                print(f"    {file_name}")
    
    # Summary recommendations
    print("\n\n💡 RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. Orphaned Subsystems (possibly valuable lost functionality):")
    valuable_orphans = [
        ("attention/", "Cross-modal attention system"),
        ("memory/pattern_memory.py", "Universal pattern memory"),
        ("persistence/", "State persistence and recovery"),
        ("brains/shared/", "Shared constraint and attention systems"),
        ("parameters/", "Parameter management systems")
    ]
    
    for path, desc in valuable_orphans:
        matching = [f for f, _ in categories['orphaned'] if path in f]
        if matching:
            print(f"   - {path}: {desc}")
            print(f"     Files: {', '.join([os.path.basename(f) for f in matching[:5]])}")
    
    print("\n2. Legacy/Outdated Code (safe to remove):")
    legacy_patterns = [
        ("gradient_fix.py", "Temporary bug fixes"),
        ("field_dimension_fix.py", "Temporary dimension fixes"),
        ("tcp_adapter.py", "Old TCP adapter (if exists)"),
        ("statistics_control.py", "Unused statistics")
    ]
    
    for pattern, desc in legacy_patterns:
        matching = [f for f, _ in categories['orphaned'] if pattern in f]
        if matching:
            print(f"   - {pattern}: {desc}")


if __name__ == "__main__":
    main()