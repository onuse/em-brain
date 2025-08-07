#!/usr/bin/env python3
"""
Project-Wide Dead Code Analysis
================================
Find and report all potential dead code across the entire project.
Focus on practical cleanup opportunities.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json

# Performance metrics
ANALYSIS_STATS = {
    'files_analyzed': 0,
    'dead_imports': 0,
    'unused_functions': 0,
    'duplicate_implementations': 0,
    'bytes_wasted': 0
}


def analyze_imports(project_dir: Path) -> Dict[str, Set[str]]:
    """Analyze import patterns to find dead dependencies"""
    
    imports_by_module = defaultdict(set)
    imported_from = defaultdict(set)
    
    print("🔍 Analyzing import patterns...")
    
    for py_file in project_dir.rglob("*.py"):
        # Skip archive and test files
        if 'archive' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        ANALYSIS_STATS['files_analyzed'] += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all imports
            import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
            for match in re.finditer(import_pattern, content):
                module = match.group(1) or ''
                imports = match.group(2)
                
                if module:
                    imports_by_module[str(py_file)].add(module)
                    imported_from[module].add(str(py_file))
                    
        except Exception as e:
            pass  # Skip files we can't read
    
    return dict(imported_from)


def find_unused_files(project_dir: Path, imported_modules: Dict[str, Set[str]]) -> List[Path]:
    """Find Python files that are never imported"""
    
    unused = []
    
    print("🔍 Finding unused files...")
    
    for py_file in project_dir.rglob("*.py"):
        # Skip special files and directories
        if any(skip in str(py_file) for skip in [
            'archive', '__pycache__', 'test', '__init__.py', 
            'setup.py', 'cleanup', 'analyze'
        ]):
            continue
        
        # Convert file path to module path
        rel_path = py_file.relative_to(project_dir)
        module_path = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        
        # Check if this module is ever imported
        is_imported = False
        for imported_module in imported_modules.keys():
            if module_path in imported_module or imported_module in module_path:
                is_imported = True
                break
        
        # Also check if it's a main script
        is_main = py_file.name in ['demo.py', 'brain.py', '__main__.py']
        
        if not is_imported and not is_main:
            unused.append(py_file)
            ANALYSIS_STATS['bytes_wasted'] += py_file.stat().st_size
    
    return unused


def find_duplicate_implementations(project_dir: Path) -> Dict[str, List[Path]]:
    """Find files with similar names that might be duplicates"""
    
    duplicates = defaultdict(list)
    
    print("🔍 Finding duplicate implementations...")
    
    # Common patterns that indicate duplication
    base_names = defaultdict(list)
    
    for py_file in project_dir.rglob("*.py"):
        if 'archive' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        # Extract base name without version numbers or suffixes
        name = py_file.stem
        base = re.sub(r'(_v\d+|_old|_new|_backup|_copy|_\d+)$', '', name)
        
        if base != name:  # It has a suffix
            base_names[base].append(py_file)
            ANALYSIS_STATS['duplicate_implementations'] += 1
    
    # Filter to only show actual duplicates
    for base, files in base_names.items():
        if len(files) > 1:
            duplicates[base] = files
    
    return dict(duplicates)


def analyze_brain_implementations(project_dir: Path) -> Dict[str, List[Path]]:
    """Specifically analyze brain-related files"""
    
    brain_files = defaultdict(list)
    
    print("🔍 Analyzing brain implementations...")
    
    brain_patterns = ['brain', 'Brain', 'cortex', 'Cortex', 'neural', 'Neural']
    
    for pattern in brain_patterns:
        for py_file in project_dir.rglob(f"*{pattern}*.py"):
            if 'archive' not in str(py_file) and '__pycache__' not in str(py_file):
                category = 'brain' if 'brain' in pattern.lower() else pattern.lower()
                brain_files[category].append(py_file)
    
    return dict(brain_files)


def analyze_test_coverage(project_dir: Path) -> Dict[str, int]:
    """Analyze test file distribution"""
    
    test_stats = {
        'unit_tests': 0,
        'integration_tests': 0,
        'performance_tests': 0,
        'behavioral_tests': 0,
        'other_tests': 0
    }
    
    print("🔍 Analyzing test coverage...")
    
    for test_file in project_dir.rglob("test*.py"):
        if '__pycache__' in str(test_file):
            continue
            
        if 'unit' in str(test_file):
            test_stats['unit_tests'] += 1
        elif 'integration' in str(test_file):
            test_stats['integration_tests'] += 1
        elif 'performance' in str(test_file) or 'perf' in str(test_file):
            test_stats['performance_tests'] += 1
        elif 'behavioral' in str(test_file):
            test_stats['behavioral_tests'] += 1
        else:
            test_stats['other_tests'] += 1
    
    return test_stats


def generate_cleanup_recommendations(analysis_results: Dict) -> List[str]:
    """Generate specific cleanup recommendations"""
    
    recommendations = []
    
    # Check unused files
    if analysis_results['unused_files']:
        total_size = sum(f.stat().st_size for f in analysis_results['unused_files'])
        mb = total_size / (1024 * 1024)
        recommendations.append(
            f"Remove {len(analysis_results['unused_files'])} unused files "
            f"(saves {mb:.2f} MB)"
        )
    
    # Check duplicates
    if analysis_results['duplicates']:
        recommendations.append(
            f"Consolidate {len(analysis_results['duplicates'])} sets of duplicate implementations"
        )
    
    # Check brain implementations
    brain_count = sum(len(files) for files in analysis_results['brain_files'].values())
    if brain_count > 10:
        recommendations.append(
            f"Review {brain_count} brain-related files (only PureFieldBrain is needed)"
        )
    
    # Check test distribution
    test_total = sum(analysis_results['test_coverage'].values())
    if test_total > 100:
        recommendations.append(
            f"Review {test_total} test files for consolidation opportunities"
        )
    
    return recommendations


def print_analysis_report(analysis_results: Dict):
    """Print comprehensive analysis report"""
    
    print("\n" + "="*70)
    print("📊 PROJECT-WIDE DEAD CODE ANALYSIS REPORT")
    print("="*70)
    
    # Unused files
    unused = analysis_results['unused_files']
    if unused:
        print(f"\n❌ UNUSED FILES ({len(unused)} files):")
        for f in unused[:10]:  # Show first 10
            rel_path = f.relative_to(analysis_results['project_dir'])
            size_kb = f.stat().st_size / 1024
            print(f"  • {rel_path} ({size_kb:.1f} KB)")
        if len(unused) > 10:
            print(f"  ... and {len(unused)-10} more")
    
    # Duplicates
    duplicates = analysis_results['duplicates']
    if duplicates:
        print(f"\n🔄 DUPLICATE IMPLEMENTATIONS ({len(duplicates)} groups):")
        for base, files in list(duplicates.items())[:5]:
            print(f"  {base}:")
            for f in files:
                rel_path = f.relative_to(analysis_results['project_dir'])
                print(f"    • {rel_path}")
    
    # Brain files
    brain_files = analysis_results['brain_files']
    if brain_files:
        print(f"\n🧠 BRAIN-RELATED FILES:")
        for category, files in brain_files.items():
            if files:
                print(f"  {category}: {len(files)} files")
    
    # Test coverage
    test_coverage = analysis_results['test_coverage']
    print(f"\n🧪 TEST DISTRIBUTION:")
    for test_type, count in test_coverage.items():
        if count > 0:
            print(f"  • {test_type}: {count}")
    
    # Recommendations
    recommendations = analysis_results['recommendations']
    if recommendations:
        print(f"\n💡 CLEANUP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Summary stats
    print(f"\n📈 ANALYSIS SUMMARY:")
    print(f"  • Files analyzed: {ANALYSIS_STATS['files_analyzed']}")
    print(f"  • Potential dead code: {ANALYSIS_STATS['bytes_wasted'] / (1024*1024):.2f} MB")
    print(f"  • Duplicate implementations: {ANALYSIS_STATS['duplicate_implementations']}")
    
    # Performance impact
    print(f"\n⚡ POTENTIAL PERFORMANCE IMPACT:")
    print(f"  • Import time reduction: ~{len(unused) * 50}ms")
    print(f"  • Memory footprint reduction: ~{ANALYSIS_STATS['bytes_wasted'] / (1024*1024):.1f} MB")
    print(f"  • Maintenance complexity: -{len(unused) + len(duplicates)} files")


def export_results_json(analysis_results: Dict, output_file: Path):
    """Export results to JSON for further processing"""
    
    # Convert Path objects to strings for JSON serialization
    json_safe = {
        'unused_files': [str(f) for f in analysis_results['unused_files']],
        'duplicates': {
            base: [str(f) for f in files] 
            for base, files in analysis_results['duplicates'].items()
        },
        'brain_files': {
            cat: [str(f) for f in files]
            for cat, files in analysis_results['brain_files'].items()
        },
        'test_coverage': analysis_results['test_coverage'],
        'recommendations': analysis_results['recommendations'],
        'stats': ANALYSIS_STATS
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_safe, f, indent=2)
    
    print(f"\n📄 Results exported to: {output_file}")


def main():
    """Run comprehensive dead code analysis"""
    
    project_dir = Path(__file__).parent
    
    print("🔍 EM-Brain Dead Code Analyzer")
    print("="*70)
    print(f"Analyzing: {project_dir}")
    
    # Run analyses
    imported_modules = analyze_imports(project_dir)
    unused_files = find_unused_files(project_dir, imported_modules)
    duplicates = find_duplicate_implementations(project_dir)
    brain_files = analyze_brain_implementations(project_dir)
    test_coverage = analyze_test_coverage(project_dir)
    
    # Compile results
    analysis_results = {
        'project_dir': project_dir,
        'unused_files': unused_files,
        'duplicates': duplicates,
        'brain_files': brain_files,
        'test_coverage': test_coverage,
        'imported_modules': imported_modules,
        'recommendations': []
    }
    
    # Generate recommendations
    analysis_results['recommendations'] = generate_cleanup_recommendations(analysis_results)
    
    # Print report
    print_analysis_report(analysis_results)
    
    # Export to JSON
    output_file = project_dir / 'dead_code_analysis.json'
    export_results_json(analysis_results, output_file)
    
    print("\n✅ Analysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())