#!/usr/bin/env python3
"""
Detailed analysis of orphaned code to identify lost functionality.
"""

import os
import ast
from pathlib import Path


def analyze_orphaned_subsystems():
    """Analyze key orphaned subsystems in detail."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    subsystems = {
        "Persistence System": {
            "path": "server/src/persistence",
            "files": [
                "persistence_manager.py",
                "brain_serializer.py", 
                "recovery_manager.py",
                "consolidation_engine.py",
                "incremental_engine.py",
                "storage_backend.py",
                "persistence_config.py"
            ],
            "description": "Complete brain state persistence and recovery"
        },
        
        "Attention System": {
            "path": "server/src/attention",
            "files": [
                "signal_attention.py",
                "object_attention.py"
            ],
            "description": "Cross-modal attention and object tracking"
        },
        
        "Shared Brain Infrastructure": {
            "path": "server/src/brains/shared",
            "files": [
                "constraint_propagation_system.py",
                "emergent_attention_allocation.py",
                "adaptive_constraint_thresholds.py",
                "constraint_pattern_inhibition.py",
                "shared_brain_state.py",
                "stream_types.py"
            ],
            "description": "Shared constraint and attention systems"
        },
        
        "Enhanced Field Dynamics": {
            "path": "server/src/brains/field",
            "files": [
                "enhanced_dynamics.py",
                "hierarchical_processing.py",
                "attention_guided.py",
                "attention_super_resolution.py",
                "adaptive_field_impl.py"
            ],
            "description": "Advanced field processing capabilities"
        },
        
        "Memory Systems": {
            "path": "server/src/memory",
            "files": [
                "pattern_memory.py"
            ],
            "description": "Universal pattern memory system"
        },
        
        "Utils and Logging": {
            "path": "server/src/utils",
            "files": [
                "persistent_memory.py",
                "brain_logger.py",
                "async_logger.py",
                "loggable_objects.py"
            ],
            "description": "Logging and persistent memory utilities"
        },
        
        "Robot Integration": {
            "path": "server/src/robot_integration",
            "files": [
                "picarx_brainstem.py"
            ],
            "description": "Low-level robot integration layer"
        }
    }
    
    print("🔬 DETAILED ANALYSIS OF ORPHANED SUBSYSTEMS")
    print("=" * 80)
    
    for system_name, info in subsystems.items():
        print(f"\n📦 {system_name}")
        print("-" * 80)
        print(f"Description: {info['description']}")
        print(f"Location: {info['path']}/")
        print("\nFiles and functionality:")
        
        for file_name in info['files']:
            file_path = base_dir / info['path'] / file_name
            if file_path.exists():
                docstring, key_classes, key_functions = analyze_file(file_path)
                print(f"\n  📄 {file_name}")
                if docstring:
                    first_line = docstring.split('\n')[0]
                    print(f"     {first_line}")
                if key_classes:
                    print(f"     Classes: {', '.join(key_classes[:3])}")
                if key_functions:
                    print(f"     Functions: {', '.join(key_functions[:3])}")


def analyze_file(file_path):
    """Extract key information from a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        
        # Extract class and function names
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions.append(node.name)
        
        return docstring, classes, functions
    except:
        return None, [], []


def check_integration_points():
    """Check where orphaned systems were meant to integrate."""
    print("\n\n🔗 INTEGRATION POINTS")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parents[2]
    
    # Check main brain files for commented imports or references
    key_files = [
        "server/brain.py",
        "server/src/core/dynamic_brain_factory.py",
        "server/src/brains/field/dynamic_unified_brain_full.py",
        "server/src/brains/field/core_brain.py"
    ]
    
    orphaned_refs = {
        "persistence": ["PersistenceManager", "brain_serializer", "save_state", "load_state"],
        "attention": ["AttentionSystem", "CrossModalAttention", "signal_attention"],
        "shared": ["constraint_propagation", "emergent_attention", "shared_brain_state"],
        "memory": ["pattern_memory", "UniversalMemory", "PatternMemory"],
        "robot_integration": ["brainstem", "PiCarXBrainstem"]
    }
    
    for file_path in key_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"\n📍 {file_path}")
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for references to orphaned systems
                found_refs = []
                for system, keywords in orphaned_refs.items():
                    for keyword in keywords:
                        if keyword in content:
                            # Check if it's commented out
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if keyword in line and ('#' in line or line.strip().startswith('#')):
                                    found_refs.append(f"   Line {i+1}: {line.strip()}")
                
                if found_refs:
                    for ref in found_refs[:5]:  # Show first 5
                        print(ref)
            except Exception as e:
                print(f"   Error reading file: {e}")


def generate_recommendations():
    """Generate specific recommendations for each orphaned system."""
    print("\n\n🎯 SPECIFIC RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = [
        {
            "system": "Persistence System",
            "status": "RESTORE",
            "reason": "Critical for cross-session learning and state recovery",
            "action": "Integrate PersistenceManager into DynamicUnifiedFieldBrain",
            "priority": "HIGH"
        },
        {
            "system": "Attention System", 
            "status": "EVALUATE",
            "reason": "Cross-modal attention could enhance field processing",
            "action": "Test if signal_attention.py can improve sensor processing",
            "priority": "MEDIUM"
        },
        {
            "system": "Shared Constraints",
            "status": "PARTIAL RESTORE",
            "reason": "Constraint propagation valuable but complex",
            "action": "Extract core constraint concepts into current ConstraintFieldND",
            "priority": "MEDIUM"
        },
        {
            "system": "Memory Pattern System",
            "status": "EVALUATE",
            "reason": "May duplicate field topology memory",
            "action": "Compare with current memory formation in field",
            "priority": "LOW"
        },
        {
            "system": "Enhanced Field Dynamics",
            "status": "ARCHIVE",
            "reason": "Superseded by current implementation",
            "action": "Move to archive/ for reference",
            "priority": "LOW"
        },
        {
            "system": "Robot Integration",
            "status": "RESTORE",
            "reason": "Brainstem layer needed for hardware deployment",
            "action": "Update picarx_brainstem.py for current architecture",
            "priority": "HIGH"
        },
        {
            "system": "Fix Files",
            "status": "DELETE",
            "reason": "Temporary bug fixes no longer needed",
            "action": "Delete gradient_fix.py, field_dimension_fix.py",
            "priority": "LOW"
        }
    ]
    
    for rec in recommendations:
        print(f"\n{'🟢' if rec['status'] == 'RESTORE' else '🟡' if rec['status'] == 'EVALUATE' else '🔴'} {rec['system']}")
        print(f"   Status: {rec['status']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Action: {rec['action']}")
        print(f"   Priority: {rec['priority']}")


def main():
    """Run the detailed orphaned code analysis."""
    analyze_orphaned_subsystems()
    check_integration_points()
    generate_recommendations()
    
    print("\n\n📊 SUMMARY")
    print("=" * 80)
    print("• 44 orphaned files containing potentially valuable functionality")
    print("• Key lost systems: Persistence, Attention, Brainstem integration")
    print("• Recommendation: Restore persistence and brainstem as priority")
    print("• Many files can be archived or deleted after evaluation")


if __name__ == "__main__":
    main()