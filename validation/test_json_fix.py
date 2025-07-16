#!/usr/bin/env python3
"""
Test JSON serialization fix
"""

import json
import numpy as np

# Test the conversion function
def convert_to_json_serializable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# Test data with problematic types
test_data = {
    'bool_val': np.bool_(True),
    'float_val': np.float64(3.14),
    'int_val': np.int64(42),
    'array_val': np.array([1, 2, 3]),
    'nested': {
        'another_bool': np.bool_(False),
        'list_with_numpy': [np.float64(1.5), np.int64(10)]
    }
}

print("Testing JSON serialization fix...")
try:
    # Convert to serializable format
    serializable = convert_to_json_serializable(test_data)
    
    # Try to serialize
    json_str = json.dumps(serializable, indent=2)
    print("✅ JSON serialization successful!")
    print("Sample output:")
    print(json_str[:200] + "...")
    
except Exception as e:
    print(f"❌ JSON serialization failed: {e}")