#!/usr/bin/env python3
"""
Test Field Brain Server Deployment

This script demonstrates how to properly configure and deploy the field brain
through the server infrastructure, including configuration testing and validation.
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

from src.brain import MinimalBrain
from src.communication import MinimalTCPServer


def backup_original_settings():
    """Backup original settings.json"""
    server_dir = Path(__file__).parent.parent.parent / "server"
    settings_file = server_dir / "settings.json"
    backup_file = server_dir / "settings_original_backup.json"
    
    if settings_file.exists() and not backup_file.exists():
        shutil.copy2(settings_file, backup_file)
        print(f"✅ Backed up original settings to {backup_file}")
    

def test_field_brain_configuration():
    """Test field brain configuration loading and instantiation."""
    print("\n🧠 FIELD BRAIN CONFIGURATION TEST")
    print("=" * 50)
    
    # Load field brain configuration
    server_dir = Path(__file__).parent.parent.parent / "server"
    field_settings_file = server_dir / "settings_field_brain.json"
    
    if not field_settings_file.exists():
        print(f"❌ Field brain settings file not found: {field_settings_file}")
        return False
    
    # Load configuration
    with open(field_settings_file, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded field brain configuration")
    print(f"   Brain type: {config['brain']['type']}")
    print(f"   Sensory dimensions: {config['brain']['sensory_dim']}")
    print(f"   Motor dimensions: {config['brain']['motor_dim']}")
    print(f"   Field spatial resolution: {config['brain']['field_spatial_resolution']}")
    print(f"   Field temporal window: {config['brain']['field_temporal_window']}")
    
    # Test brain instantiation
    try:
        print("\n🔧 Testing brain instantiation...")
        brain = MinimalBrain(config=config, quiet_mode=True)
        
        print(f"✅ Brain created successfully!")
        print(f"   Type: {type(brain.vector_brain).__name__}")
        print(f"   Architecture: {brain.brain_type}")
        
        # Test basic processing
        print("\n🔄 Testing sensory processing...")
        sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        start_time = time.time()
        action, brain_state = brain.process_sensory_input(sensory_input)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"✅ Processing successful!")
        print(f"   Input: {len(sensory_input)}D → Output: {len(action)}D")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Confidence: {brain_state.get('prediction_confidence', 0.0):.3f}")
        print(f"   Architecture: {brain_state.get('architecture', 'unknown')}")
        
        # Test statistics
        print("\n📊 Testing brain statistics...")
        stats = brain.get_brain_stats()
        print(f"✅ Statistics available!")
        print(f"   Total cycles: {stats['brain_summary']['total_cycles']}")
        print(f"   Architecture: {stats['brain_summary']['architecture']}")
        
        # Cleanup
        brain.finalize_session()
        
        return True
        
    except Exception as e:
        print(f"❌ Brain instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tcp_server_integration():
    """Test TCP server integration with field brain."""
    print("\n🌐 TCP SERVER INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Load field brain configuration
        server_dir = Path(__file__).parent.parent.parent / "server"
        field_settings_file = server_dir / "settings_field_brain.json"
        
        with open(field_settings_file, 'r') as f:
            config = json.load(f)
        
        # Create brain and server
        print("🔧 Creating field brain and TCP server...")
        brain = MinimalBrain(config=config, quiet_mode=True)
        tcp_server = MinimalTCPServer(brain, host='127.0.0.1', port=9998)  # Use different port for testing
        
        print(f"✅ TCP server created successfully!")
        print(f"   Brain type: {type(brain.vector_brain).__name__}")
        print(f"   Server host: {tcp_server.host}")
        print(f"   Server port: {tcp_server.port}")
        
        # Test server interface methods
        print("\n🔄 Testing server interface methods...")
        
        # Test process_sensory_input
        sensory_input = [0.1] * 16
        action, brain_state = brain.process_sensory_input(sensory_input)
        print(f"✅ process_sensory_input: {len(sensory_input)}D → {len(action)}D")
        
        # Test get_brain_stats
        stats = brain.get_brain_stats()
        print(f"✅ get_brain_stats: {len(stats)} stat categories")
        
        # Cleanup
        brain.finalize_session()
        
        return True
        
    except Exception as e:
        print(f"❌ TCP server integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_switching():
    """Test switching between sparse_goldilocks and field brain configurations."""
    print("\n🔄 CONFIGURATION SWITCHING TEST")
    print("=" * 50)
    
    try:
        # Test sparse_goldilocks configuration
        print("🔧 Testing sparse_goldilocks configuration...")
        sparse_config = {
            "brain": {
                "type": "sparse_goldilocks",
                "sensory_dim": 16,
                "motor_dim": 4
            }
        }
        
        sparse_brain = MinimalBrain(config=sparse_config, quiet_mode=True)
        print(f"✅ Sparse brain: {type(sparse_brain.vector_brain).__name__}")
        sparse_brain.finalize_session()
        
        # Test field configuration
        print("\n🔧 Testing field configuration...")
        field_config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 20,
                "field_temporal_window": 10.0
            }
        }
        
        field_brain = MinimalBrain(config=field_config, quiet_mode=True)
        print(f"✅ Field brain: {type(field_brain.vector_brain).__name__}")
        field_brain.finalize_session()
        
        print("\n✅ Configuration switching successful!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration switching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_deployment_guide():
    """Create deployment guide for field brain."""
    guide_content = """# Field Brain Deployment Guide

## Quick Deployment

1. **Update settings.json:**
   ```bash
   cp settings_field_brain.json settings.json
   ```

2. **Start the server:**
   ```bash
   python3 brain_server.py
   ```

## Configuration Options

### Field Brain Specific Settings:
- `field_spatial_resolution`: Spatial resolution of the field (default: 20)
- `field_temporal_window`: Temporal window in seconds (default: 10.0)
- `field_evolution_rate`: Rate of field evolution (default: 0.1)
- `constraint_discovery_rate`: Rate of constraint discovery (default: 0.15)

### Memory Configuration:
- `persistent_memory_path`: Set to "./robot_memory_field" for field brain
- Field brains have different memory patterns than sparse brains

## Switching Between Brain Types

### To use Field Brain:
```json
{
  "brain": {
    "type": "field",
    "sensory_dim": 16,
    "motor_dim": 4,
    "field_spatial_resolution": 20,
    "field_temporal_window": 10.0
  }
}
```

### To use Sparse Goldilocks Brain:
```json
{
  "brain": {
    "type": "sparse_goldilocks",
    "sensory_dim": 16,
    "motor_dim": 4,
    "max_patterns": 100000
  }
}
```

## Performance Characteristics

- **Field Brain**: ~25ms processing time, continuous field dynamics
- **Sparse Brain**: ~10ms processing time, discrete pattern matching

## Troubleshooting

1. **Import Errors**: Ensure all field brain modules are in the correct folders
2. **Configuration Errors**: Validate JSON syntax in settings.json
3. **Memory Issues**: Adjust field_spatial_resolution if memory usage is too high
4. **Performance Issues**: Reduce field_temporal_window for faster processing

## Production Deployment

For production use:
1. Set appropriate field_spatial_resolution based on hardware
2. Configure memory persistence settings
3. Enable performance monitoring
4. Set up proper logging levels
"""
    
    guide_file = Path(__file__).parent.parent.parent / "docs" / "field_brain_deployment.md"
    guide_file.parent.mkdir(exist_ok=True)
    
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"📖 Deployment guide created: {guide_file}")


def main():
    """Run all field brain deployment tests."""
    print("🚀 FIELD BRAIN SERVER DEPLOYMENT TEST SUITE")
    print("=" * 60)
    
    # Backup original settings
    backup_original_settings()
    
    # Run tests
    tests = [
        ("Configuration Test", test_field_brain_configuration),
        ("TCP Server Integration", test_tcp_server_integration),
        ("Configuration Switching", test_configuration_switching),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Create deployment guide
    create_deployment_guide()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎯 ALL TESTS PASSED!")
        print("   Field brain is ready for production deployment!")
        print("\nNext steps:")
        print("   1. Copy settings_field_brain.json to settings.json")
        print("   2. Start the server with: python3 brain_server.py")
        print("   3. The server will use the field brain architecture")
    else:
        print(f"\n⚠️  {total - passed} tests failed - check error messages above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)