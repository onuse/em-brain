#!/usr/bin/env python3
"""
Quick Field Persistence Test

Fast validation of field persistence methods.
"""

import sys
import os
import time
import tempfile
import torch

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

try:
    from field_native_brain import create_unified_field_brain
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def quick_persistence_test():
    """Quick test of field persistence methods."""
    print("🧠 QUICK FIELD PERSISTENCE TEST")
    
    # Create small field brain
    brain1 = create_unified_field_brain(spatial_resolution=4, quiet_mode=False)
    
    # Add some field activity
    for i in range(3):
        sensory_input = [0.5 + 0.3 * i] + [0.5] * 23
        brain1.process_robot_cycle(sensory_input)
    
    original_stats = brain1.get_field_memory_stats()
    original_field = brain1.unified_field.clone()
    
    print(f"\n💾 Testing field save...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.field.gz') as tmp_file:
        field_file = tmp_file.name
    
    save_success = brain1.save_field_state(field_file, compress=True)
    
    print(f"\n📂 Testing field load...")
    brain2 = create_unified_field_brain(spatial_resolution=4, quiet_mode=False)
    load_success = brain2.load_field_state(field_file)
    
    if load_success:
        field_match = torch.allclose(original_field, brain2.unified_field, atol=1e-6)
        print(f"   Field integrity: {'✅ PASS' if field_match else '❌ FAIL'}")
    
    print(f"\n🔄 Testing delta compression...")
    # Make small change
    brain1.unified_field[0, 0, 0, 0, 0, 0] += 0.1
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.delta.gz') as tmp_file:
        delta_file = tmp_file.name
    
    delta_save_success = brain1.save_field_delta(delta_file, original_field)
    
    print(f"\n🌙 Testing consolidation...")
    consolidated = brain1.consolidate_field(0.1)
    print(f"   Consolidated regions: {consolidated}")
    
    # Cleanup
    try:
        os.unlink(field_file)
        os.unlink(delta_file)
    except:
        pass
    
    success = save_success and load_success and field_match and delta_save_success
    print(f"\n✅ Overall persistence test: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = quick_persistence_test()
    print(f"\n🌊 Field persistence {'✅ WORKING' if success else '❌ NEEDS WORK'}")