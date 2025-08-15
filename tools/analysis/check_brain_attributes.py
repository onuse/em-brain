#!/usr/bin/env python3
"""
Check Brain Attributes Compatibility

Analyzes what attributes BrainLogger expects vs what MinimalBrain provides
to prevent missing attribute errors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import re
import ast
from pathlib import Path

def extract_brain_attribute_accesses(file_path):
    """Extract all brain.attribute accesses from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find brain.attribute patterns
    pattern = r'brain\.([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, content)
    
    return set(matches)

def extract_brain_class_attributes(file_path):
    """Extract attributes defined in MinimalBrain class."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse the AST to find self.attribute assignments
    tree = ast.parse(content)
    
    attributes = set()
    
    class AttributeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_minimal_brain = False
            
        def visit_ClassDef(self, node):
            if node.name == 'MinimalBrain':
                self.in_minimal_brain = True
                self.generic_visit(node)
                self.in_minimal_brain = False
            else:
                self.generic_visit(node)
        
        def visit_Assign(self, node):
            if self.in_minimal_brain:
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes.add(target.attr)
            self.generic_visit(node)
    
    visitor = AttributeVisitor()
    visitor.visit(tree)
    
    return attributes

def check_brain_compatibility():
    """Check compatibility between BrainLogger expectations and MinimalBrain implementation."""
    
    server_root = Path(__file__).parent.parent.parent / 'server' / 'src'
    
    brain_logger_path = server_root / 'utils' / 'brain_logger.py'
    brain_path = server_root / 'brain.py'
    
    print("🔍 Checking Brain Attributes Compatibility...")
    print(f"   BrainLogger: {brain_logger_path}")
    print(f"   MinimalBrain: {brain_path}")
    print()
    
    # Extract what BrainLogger expects
    expected_attributes = extract_brain_attribute_accesses(brain_logger_path)
    print(f"📋 Attributes BrainLogger expects ({len(expected_attributes)}):")
    for attr in sorted(expected_attributes):
        print(f"   - brain.{attr}")
    print()
    
    # Extract what MinimalBrain provides
    provided_attributes = extract_brain_class_attributes(brain_path)
    print(f"🧠 Attributes MinimalBrain provides ({len(provided_attributes)}):")
    for attr in sorted(provided_attributes):
        print(f"   - self.{attr}")
    print()
    
    # Check compatibility
    missing_attributes = expected_attributes - provided_attributes
    extra_attributes = provided_attributes - expected_attributes
    
    print("🔍 Compatibility Analysis:")
    print(f"   Expected: {len(expected_attributes)} attributes")
    print(f"   Provided: {len(provided_attributes)} attributes")
    print(f"   Missing:  {len(missing_attributes)} attributes")
    print(f"   Extra:    {len(extra_attributes)} attributes")
    print()
    
    if missing_attributes:
        print("❌ Missing Attributes (will cause runtime errors):")
        for attr in sorted(missing_attributes):
            print(f"   - {attr}")
        print()
    else:
        print("✅ No missing attributes!")
        print()
    
    if extra_attributes:
        print("ℹ️  Extra Attributes (unused by logger):")
        for attr in sorted(extra_attributes):
            print(f"   - {attr}")
        print()
    
    # Generate fixes if needed
    if missing_attributes:
        print("🔧 Suggested fixes for MinimalBrain.__init__():")
        for attr in sorted(missing_attributes):
            print(f"   self.{attr} = []  # or appropriate default value")
        print()
    
    return len(missing_attributes) == 0

def check_method_compatibility():
    """Check for missing methods that might be called on brain object."""
    
    server_root = Path(__file__).parent.parent.parent / 'server' / 'src'
    brain_logger_path = server_root / 'utils' / 'brain_logger.py'
    
    with open(brain_logger_path, 'r') as f:
        content = f.read()
    
    # Find brain.method() calls
    method_pattern = r'brain\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    methods = re.findall(method_pattern, content)
    
    expected_methods = set(methods)
    
    print("🔧 Methods BrainLogger expects:")
    for method in sorted(expected_methods):
        print(f"   - brain.{method}()")
    print()
    
    return expected_methods

if __name__ == "__main__":
    print("🧠 Brain Attributes Compatibility Checker")
    print("=" * 50)
    print()
    
    # Check attributes
    attributes_ok = check_brain_compatibility()
    
    # Check methods
    expected_methods = check_method_compatibility()
    
    # Summary
    print("📊 Summary:")
    if attributes_ok:
        print("   ✅ All required attributes are present")
    else:
        print("   ❌ Missing attributes detected - add them to prevent runtime errors")
    
    print(f"   ℹ️  Expected methods: {', '.join(sorted(expected_methods))}")
    print()
    print("🎯 Result: " + ("Ready to run!" if attributes_ok else "Fix missing attributes first"))