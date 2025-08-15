#!/usr/bin/env python3
"""
Detailed analysis of each orphaned file to determine if it was
intentionally abandoned or accidentally disconnected.
"""

import os
import ast
from pathlib import Path
from datetime import datetime


def analyze_file_in_detail(file_path):
    """Extract detailed information about a file's purpose and contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        
        # Get file stats
        stats = os.stat(file_path)
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d')
        
        # Extract key features
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                classes.append((node.name, class_doc))
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                func_doc = ast.get_docstring(node)
                functions.append((node.name, func_doc))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Look for TODO/FIXME comments
        todos = []
        for line in content.split('\n'):
            if 'TODO' in line or 'FIXME' in line:
                todos.append(line.strip())
        
        # Check for test code
        has_tests = 'if __name__' in content
        
        return {
            'docstring': docstring,
            'modified': mod_time,
            'classes': classes,
            'functions': functions,
            'imports': imports,
            'todos': todos,
            'has_tests': has_tests,
            'size': len(content.split('\n'))
        }
    except Exception as e:
        return {'error': str(e)}


def categorize_orphaned_files():
    """Categorize orphaned files by their apparent purpose."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    # These are the orphaned files from our inventory
    orphaned_files = {
        "Field Dynamics Evolution": [
            "server/src/brains/field/dynamics/analog_field_dynamics.py",
            "server/src/brains/field/dynamics/multiscale_field_dynamics.py",
            "server/src/brains/field/dynamics/temporal_field_dynamics.py",
            "server/src/brains/field/dynamics/constraint_field_dynamics.py",
        ],
        
        "Enhanced Field Processing": [
            "server/src/brains/field/enhanced_dynamics.py",
            "server/src/brains/field/hierarchical_processing.py",
            "server/src/brains/field/attention_guided.py",
            "server/src/brains/field/attention_super_resolution.py",
            "server/src/brains/field/adaptive_field_impl.py",
        ],
        
        "Field Brain Core Files": [
            "server/src/brains/field/field_types.py",
            "server/src/brains/field/robot_interface.py",
            "server/src/brains/field/optimized_gradients.py",
            "server/src/brains/field/dynamic_unified_brain.py",
            "server/src/brains/field/blended_reality.py",
        ],
        
        "Temporary Fixes": [
            "server/src/brains/field/gradient_fix.py",
            "server/src/brains/field/field_dimension_fix.py",
        ],
        
        "Shared Brain Infrastructure": [
            "server/src/brains/shared/constraint_propagation_system.py",
            "server/src/brains/shared/emergent_attention_allocation.py",
            "server/src/brains/shared/adaptive_constraint_thresholds.py",
            "server/src/brains/shared/constraint_pattern_inhibition.py",
            "server/src/brains/shared/shared_brain_state.py",
            "server/src/brains/shared/stream_types.py",
        ],
        
        "Configuration and Parameters": [
            "server/src/parameters/cognitive_constants.py",
            "server/src/config/gpu_memory_manager.py",
        ],
        
        "Utilities and Logging": [
            "server/src/utils/persistent_memory.py",
            "server/src/utils/brain_logger.py",
            "server/src/utils/async_logger.py",
            "server/src/utils/loggable_objects.py",
        ],
        
        "Core Infrastructure": [
            "server/src/core/simple_field_brain.py",
            "server/src/core/error_codes.py",
            "server/src/core/logging_service.py",
        ],
        
        "Communication": [
            "server/src/communication/protocol.py",
        ],
        
        "Miscellaneous": [
            "server/src/statistics_control.py",
            "server/src/brains/brain_maintenance_interface.py",
        ]
    }
    
    return orphaned_files, base_dir


def analyze_category(category_name, files, base_dir):
    """Analyze a category of orphaned files."""
    print(f"\n{'='*80}")
    print(f"📁 {category_name}")
    print(f"{'='*80}")
    
    for file_path in files:
        full_path = base_dir / file_path
        if not full_path.exists():
            print(f"\n❌ {file_path} - FILE NOT FOUND")
            continue
            
        info = analyze_file_in_detail(full_path)
        
        print(f"\n📄 {os.path.basename(file_path)}")
        print(f"   Modified: {info.get('modified', 'Unknown')}")
        print(f"   Size: {info.get('size', 0)} lines")
        
        if info.get('docstring'):
            # First few lines of docstring
            doc_lines = info['docstring'].split('\n')[:3]
            for line in doc_lines:
                if line.strip():
                    print(f"   Purpose: {line.strip()}")
                    break
        
        if info.get('classes'):
            print(f"   Classes: {len(info['classes'])}")
            for class_name, class_doc in info['classes'][:3]:
                doc_preview = class_doc.split('\n')[0] if class_doc else "No docstring"
                print(f"      - {class_name}: {doc_preview[:60]}")
        
        if info.get('functions'):
            print(f"   Functions: {len(info['functions'])}")
            for func_name, func_doc in info['functions'][:3]:
                doc_preview = func_doc.split('\n')[0] if func_doc else "No docstring"
                print(f"      - {func_name}(): {doc_preview[:50]}")
        
        if info.get('imports'):
            # Check for interesting imports
            field_imports = [imp for imp in info['imports'] if 'field' in imp or 'brain' in imp]
            if field_imports:
                print(f"   Key imports: {', '.join(field_imports[:3])}")
        
        if info.get('todos'):
            print(f"   TODOs: {len(info['todos'])}")
            print(f"      {info['todos'][0][:70]}")
        
        if info.get('has_tests'):
            print(f"   ⚠️  Has test code (__main__)")


def analyze_connections():
    """Look for clues about how these files were meant to connect."""
    print("\n\n🔍 LOOKING FOR CONNECTION CLUES")
    print("="*80)
    
    base_dir = Path(__file__).resolve().parents[2]
    
    # Check for references in key files
    key_files_to_check = [
        "server/src/brains/field/core_brain.py",
        "server/src/brains/field/dynamic_unified_brain_full.py",
        "server/src/core/dynamic_brain_factory.py"
    ]
    
    # Keywords to look for
    keywords = {
        "phases": ["Phase A", "Phase B", "Phase C", "analog_field", "multiscale"],
        "attention": ["attention_guided", "attention_allocation", "super_resolution"],
        "shared": ["shared_brain_state", "stream_types", "constraint_propagation"],
        "gpu": ["gpu_memory", "memory_manager"],
        "persistence": ["persistent_memory", "persistence"],
    }
    
    for file_path in key_files_to_check:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"\n📍 Checking {file_path}:")
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                found_refs = []
                for category, terms in keywords.items():
                    for term in terms:
                        if term in content:
                            # Find the line
                            for i, line in enumerate(content.split('\n')):
                                if term in line:
                                    found_refs.append((category, term, i+1, line.strip()))
                                    break
                
                if found_refs:
                    for cat, term, line_no, line in found_refs[:5]:
                        print(f"   Line {line_no} [{cat}]: {line[:80]}")
                else:
                    print(f"   No references found")
                    
            except Exception as e:
                print(f"   Error: {e}")


def generate_evaluation():
    """Generate evaluation of likely intentional vs accidental orphaning."""
    print("\n\n🎯 EVALUATION: INTENTIONAL VS ACCIDENTAL ORPHANING")
    print("="*80)
    
    evaluations = [
        {
            "category": "Field Dynamics Evolution (Phase A)",
            "files": ["analog_field_dynamics.py", "multiscale_field_dynamics.py", "temporal_field_dynamics.py"],
            "verdict": "LIKELY INTENTIONAL",
            "evidence": [
                "These are marked as 'Phase A1', 'Phase A2', 'Phase A3' - early prototypes",
                "Superseded by current N-dimensional implementation",
                "constraint_field_nd.py is the current version"
            ],
            "recommendation": "ARCHIVE - Historical value, shows evolution of design"
        },
        
        {
            "category": "Enhanced Field Processing",
            "files": ["enhanced_dynamics.py", "hierarchical_processing.py", "attention_guided.py"],
            "verdict": "POSSIBLY ACCIDENTAL",
            "evidence": [
                "These add sophisticated features not in current brain",
                "attention_guided.py could enhance sensor processing",
                "hierarchical_processing.py adds multi-scale features"
            ],
            "recommendation": "EVALUATE - May contain valuable enhancements"
        },
        
        {
            "category": "Shared Brain Infrastructure",
            "files": ["constraint_propagation_system.py", "emergent_attention_allocation.py"],
            "verdict": "LIKELY ACCIDENTAL",
            "evidence": [
                "Sophisticated constraint and attention systems",
                "shared_brain_state.py suggests multi-brain coordination",
                "No obvious replacement in current architecture"
            ],
            "recommendation": "INVESTIGATE - Advanced features that may have been lost"
        },
        
        {
            "category": "Blended Reality",
            "files": ["blended_reality.py"],
            "verdict": "FALSE POSITIVE",
            "evidence": [
                "Actually IS used by dynamic_unified_brain_full.py",
                "Our analysis tool missed the integration pattern"
            ],
            "recommendation": "KEEP - Currently active!"
        },
        
        {
            "category": "Optimized Gradients",
            "files": ["optimized_gradients.py"],
            "verdict": "FALSE POSITIVE",
            "evidence": [
                "Used by both core_brain.py and dynamic_unified_brain_full.py",
                "Critical for gradient calculations"
            ],
            "recommendation": "KEEP - Currently active!"
        },
        
        {
            "category": "Simple Field Brain",
            "files": ["simple_field_brain.py"],
            "verdict": "INTENTIONAL",
            "evidence": [
                "Alternative simple implementation",
                "Used when use_simple_brain=True in config",
                "Kept for testing/comparison"
            ],
            "recommendation": "KEEP - Useful for testing"
        },
        
        {
            "category": "GPU Memory Manager",
            "files": ["gpu_memory_manager.py"],
            "verdict": "POSSIBLY ACCIDENTAL",
            "evidence": [
                "Sophisticated GPU memory management",
                "Could help with MPS issues on Mac",
                "No replacement in current code"
            ],
            "recommendation": "EVALUATE - May solve current GPU problems"
        },
        
        {
            "category": "Cognitive Constants",
            "files": ["cognitive_constants.py"],
            "verdict": "LIKELY ACCIDENTAL",
            "evidence": [
                "Contains 'Cognitive DNA' - fundamental constants",
                "No obvious replacement",
                "Parameters scattered in current implementation"
            ],
            "recommendation": "INVESTIGATE - May contain important tuning values"
        }
    ]
    
    for eval_item in evaluations:
        print(f"\n📋 {eval_item['category']}")
        print(f"   Files: {', '.join(eval_item['files'])}")
        print(f"   Verdict: {eval_item['verdict']}")
        print(f"   Evidence:")
        for evidence in eval_item['evidence']:
            print(f"      • {evidence}")
        print(f"   → {eval_item['recommendation']}")


def main():
    """Run detailed orphaned file analysis."""
    orphaned_files, base_dir = categorize_orphaned_files()
    
    print("🔬 DETAILED ORPHANED FILE ANALYSIS")
    print("="*80)
    print("Let's examine each orphaned file to determine if it was")
    print("intentionally abandoned or accidentally disconnected.")
    
    # Analyze each category
    for category, files in orphaned_files.items():
        analyze_category(category, files, base_dir)
    
    # Look for connection clues
    analyze_connections()
    
    # Generate evaluation
    generate_evaluation()
    
    print("\n\n📊 SUMMARY")
    print("="*80)
    print("• Several 'orphaned' files are actually still in use (false positives)")
    print("• Phase A files (analog, multiscale) are likely intentional - early prototypes")
    print("• Shared brain infrastructure looks accidentally disconnected")
    print("• Enhanced processing files may contain valuable features")
    print("• Need to carefully evaluate before deleting anything")


if __name__ == "__main__":
    main()