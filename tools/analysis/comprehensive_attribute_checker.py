#!/usr/bin/env python3
"""
Comprehensive Attribute Checker

Analyzes all potential attribute access patterns across the codebase
to predict runtime errors before they happen.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import re
import ast
from pathlib import Path
from typing import Dict, Set, List, Tuple

def extract_all_object_attribute_accesses(file_path: Path) -> Dict[str, Set[str]]:
    """Extract all object.attribute accesses from a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find object.attribute patterns
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, content)
        
        accesses = {}
        for obj, attr in matches:
            if obj not in accesses:
                accesses[obj] = set()
            accesses[obj].add(attr)
        
        return accesses
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def extract_class_attributes(file_path: Path, class_name: str) -> Set[str]:
    """Extract attributes defined in a specific class."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        attributes = set()
        methods = set()
        
        class AttributeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_target_class = False
                
            def visit_ClassDef(self, node):
                if node.name == class_name:
                    self.in_target_class = True
                    self.generic_visit(node)
                    self.in_target_class = False
                else:
                    self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if self.in_target_class:
                    methods.add(node.name)
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                if self.in_target_class:
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            attributes.add(target.attr)
                self.generic_visit(node)
        
        visitor = AttributeVisitor()
        visitor.visit(tree)
        
        return attributes, methods
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return set(), set()

def scan_all_files(directory: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Scan all Python files for object.attribute accesses."""
    all_accesses = {}
    
    for py_file in directory.rglob("*.py"):
        if py_file.name.startswith('.'):
            continue
            
        relative_path = str(py_file.relative_to(directory))
        accesses = extract_all_object_attribute_accesses(py_file)
        
        if accesses:
            all_accesses[relative_path] = accesses
    
    return all_accesses

def check_brain_ecosystem():
    """Comprehensive check of the entire brain ecosystem."""
    
    server_root = Path(__file__).parent.parent.parent / 'server' / 'src'
    
    print("🔍 Comprehensive Brain Ecosystem Analysis")
    print("=" * 60)
    print()
    
    # Scan all files for attribute accesses
    print("📂 Scanning all Python files for attribute accesses...")
    all_accesses = scan_all_files(server_root)
    
    # Focus on brain-related objects
    brain_related_objects = ['brain', 'self', 'vector_brain', 'emergent_confidence', 'hardware_adaptation']
    
    potential_issues = []
    
    # Check MinimalBrain class
    brain_file = server_root / 'brain.py'
    brain_attributes, brain_methods = extract_class_attributes(brain_file, 'MinimalBrain')
    
    print(f"🧠 MinimalBrain Analysis:")
    print(f"   Attributes: {len(brain_attributes)}")
    print(f"   Methods: {len(brain_methods)}")
    print()
    
    # Check SparseGoldilocksBrain class
    goldilocks_file = server_root / 'vector_stream' / 'sparse_goldilocks_brain.py'
    if goldilocks_file.exists():
        goldilocks_attributes, goldilocks_methods = extract_class_attributes(goldilocks_file, 'SparseGoldilocksBrain')
        print(f"🌟 SparseGoldilocksBrain Analysis:")
        print(f"   Attributes: {len(goldilocks_attributes)}")
        print(f"   Methods: {len(goldilocks_methods)}")
        print()
    
    # Check EmergentConfidenceSystem class
    confidence_file = server_root / 'vector_stream' / 'emergent_confidence_system.py'
    if confidence_file.exists():
        confidence_attributes, confidence_methods = extract_class_attributes(confidence_file, 'EmergentConfidenceSystem')
        print(f"🎯 EmergentConfidenceSystem Analysis:")
        print(f"   Attributes: {len(confidence_attributes)}")
        print(f"   Methods: {len(confidence_methods)}")
        print()
    
    # Check HardwareAdaptation class
    hardware_file = server_root / 'utils' / 'hardware_adaptation.py'
    if hardware_file.exists():
        hardware_attributes, hardware_methods = extract_class_attributes(hardware_file, 'HardwareAdaptation')
        print(f"🔧 HardwareAdaptation Analysis:")
        print(f"   Attributes: {len(hardware_attributes)}")
        print(f"   Methods: {len(hardware_methods)}")
        print()
    
    # Look for potential issues
    print("🚨 Potential Runtime Issues:")
    print()
    
    issue_count = 0
    for file_path, accesses in all_accesses.items():
        for obj_name, attributes in accesses.items():
            if obj_name == 'brain':
                for attr in attributes:
                    if attr not in brain_attributes and attr not in brain_methods:
                        potential_issues.append(f"{file_path}: brain.{attr}")
                        issue_count += 1
            elif obj_name == 'hardware_adaptation':
                for attr in attributes:
                    if attr not in hardware_attributes and attr not in hardware_methods:
                        potential_issues.append(f"{file_path}: hardware_adaptation.{attr}")
                        issue_count += 1
            elif obj_name == 'emergent_confidence':
                for attr in attributes:
                    if attr not in confidence_attributes and attr not in confidence_methods:
                        potential_issues.append(f"{file_path}: emergent_confidence.{attr}")
                        issue_count += 1
    
    if potential_issues:
        print(f"❌ Found {len(potential_issues)} potential attribute access issues:")
        for issue in sorted(potential_issues):
            print(f"   - {issue}")
        print()
    else:
        print("✅ No obvious attribute access issues detected!")
        print()
    
    # Check for common method calls that might fail
    print("🔧 Method Call Analysis:")
    common_method_patterns = ['get_', 'compute_', 'update_', 'process_', '_estimate_']
    
    method_issues = []
    for file_path, accesses in all_accesses.items():
        for obj_name, attributes in accesses.items():
            if obj_name == 'brain':
                for attr in attributes:
                    if any(attr.startswith(pattern) for pattern in common_method_patterns):
                        if attr not in brain_methods:
                            method_issues.append(f"{file_path}: brain.{attr}() method missing")
    
    if method_issues:
        print(f"⚠️  Found {len(method_issues)} potential method call issues:")
        for issue in sorted(method_issues):
            print(f"   - {issue}")
        print()
    else:
        print("✅ All expected methods appear to be implemented!")
        print()
    
    # Summary
    total_issues = len(potential_issues) + len(method_issues)
    print("📊 Summary:")
    print(f"   Files scanned: {len(all_accesses)}")
    print(f"   Potential attribute issues: {len(potential_issues)}")
    print(f"   Potential method issues: {len(method_issues)}")
    print(f"   Total potential issues: {total_issues}")
    print()
    
    if total_issues == 0:
        print("🎉 Ecosystem looks healthy! Low risk of missing attribute errors.")
    elif total_issues < 5:
        print("⚠️  Minor issues detected. Low-medium risk of runtime errors.")
    else:
        print("🚨 Multiple issues detected. High risk of runtime errors.")
    
    return total_issues == 0

if __name__ == "__main__":
    healthy = check_brain_ecosystem()
    exit(0 if healthy else 1)