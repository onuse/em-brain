#!/usr/bin/env python3
"""
Test that PureFieldBrain works with telemetry system
"""

import sys
import torch
sys.path.append('.')

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS
from src.core.brain_telemetry import BrainTelemetryAdapter

def test_telemetry_compatibility():
    """Test that PureFieldBrain provides all expected telemetry attributes"""
    print("Testing PureFieldBrain telemetry compatibility...")
    
    # Create brain
    brain = PureFieldBrain(
        input_dim=24,
        output_dim=4,
        scale_config=SCALE_CONFIGS['hardware_constrained'],
        aggressive=True
    )
    
    print(f"✅ Brain created")
    
    # Check critical attributes
    required_attributes = [
        'brain_cycles',
        'cycle_count', 
        'experience_count',
        'field',
        'unified_field',
        'tensor_shape',
        'total_dimensions',
        'working_memory',
        'memory_regions',
        'experiences',
        '_current_prediction_confidence',
        '_last_prediction_error',
        '_prediction_confidence_history'
    ]
    
    missing = []
    for attr in required_attributes:
        if not hasattr(brain, attr):
            missing.append(attr)
            print(f"❌ Missing: {attr}")
        else:
            value = getattr(brain, attr)
            print(f"✅ Has {attr}: {type(value).__name__}")
    
    # Test methods
    required_methods = [
        'forward',
        'process',  # Compatibility alias
        'get_brain_state',
        '_create_brain_state'
    ]
    
    for method in required_methods:
        if not hasattr(brain, method):
            missing.append(method)
            print(f"❌ Missing method: {method}")
        else:
            print(f"✅ Has method: {method}")
    
    # Test telemetry adapter
    try:
        adapter = BrainTelemetryAdapter(brain)
        telemetry = adapter.get_telemetry()
        print(f"\n✅ Telemetry extraction successful!")
        print(f"   Brain cycles: {telemetry.brain_cycles}")
        print(f"   Field information: {telemetry.field_information:.3f}")
        print(f"   Prediction confidence: {telemetry.prediction_confidence:.3f}")
    except Exception as e:
        print(f"\n❌ Telemetry extraction failed: {e}")
        return False
    
    # Test forward pass and cycle counting
    for i in range(5):
        brain.forward(torch.randn(24))
    
    print(f"\n✅ After 5 cycles:")
    print(f"   brain.brain_cycles = {brain.brain_cycles}")
    print(f"   brain.cycle_count = {brain.cycle_count}")
    print(f"   brain.experience_count = {brain.experience_count}")
    
    # Test brain state
    state = brain.get_brain_state()
    print(f"\n✅ Brain state keys: {list(state.keys())}")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("="*60)
    print("PureFieldBrain Telemetry Compatibility Test")
    print("="*60)
    
    if test_telemetry_compatibility():
        print("\n✅ All telemetry compatibility requirements met!")
    else:
        print("\n❌ Some compatibility issues remain")
    print("="*60)