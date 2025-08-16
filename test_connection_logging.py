#!/usr/bin/env python3
"""
Test Connection Logging

This script demonstrates the enhanced connection logging by attempting to connect
to a brain server that may or may not be running. It shows all the detailed
logging that's now available for debugging connection issues.
"""

import sys
import time
import logging
from pathlib import Path

# Add client path
sys.path.append(str(Path(__file__).parent / "client_picarx" / "src"))

from brainstem.brain_client import BrainClient, BrainServerConfig

def test_connection_with_logging():
    """Test connection with detailed logging enabled."""
    
    # Enable debug logging for all components
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 60)
    print("🔍 CONNECTION LOGGING TEST")
    print("=" * 60)
    print("This test demonstrates the enhanced connection logging.")
    print("Watch for detailed debug messages about:")
    print("  - Socket creation and configuration")
    print("  - TCP connection attempts")
    print("  - Handshake message exchange")
    print("  - Timeout handling")
    print("  - Error conditions")
    print("=" * 60)
    
    # Test configuration
    config = BrainServerConfig(
        host="localhost",
        port=9999,
        timeout=5.0,  # 5 second timeout for testing
        sensory_dimensions=100,
        action_dimensions=6
    )
    
    print(f"\n📊 Test Configuration:")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Timeout: {config.timeout}s")
    print(f"   Sensory dimensions: {config.sensory_dimensions}")
    print(f"   Action dimensions: {config.action_dimensions}")
    
    # Create client
    print(f"\n🤖 Creating BrainClient...")
    client = BrainClient(config)
    
    # Attempt connection
    print(f"\n🔌 Attempting connection...")
    print("=" * 40)
    
    success = client.connect()
    
    print("=" * 40)
    if success:
        print("✅ Connection successful!")
        print("🧪 Testing sensor data exchange...")
        
        # Test sending some sensor data
        test_sensors = [0.5] * config.sensory_dimensions
        response = client.process_sensors(test_sensors)
        
        if response:
            print(f"✅ Got motor response: {len(response['motor_commands'])} commands")
        else:
            print("⚠️  No motor response (may be normal)")
        
        print("🔌 Disconnecting...")
        client.disconnect()
        
    else:
        print("❌ Connection failed!")
        print("💡 This is expected if no brain server is running.")
        print("   Start the brain server and run this test again to see successful connection logs.")
    
    print("\n" + "=" * 60)
    print("🔍 LOGGING ANALYSIS")
    print("=" * 60)
    print("Review the detailed logs above to understand:")
    print("1. Socket creation and configuration steps")
    print("2. TCP connection timing and results") 
    print("3. Handshake message encoding/decoding")
    print("4. Specific error conditions and timeouts")
    print("5. Cleanup and disconnect procedures")
    print("\nThese logs will help debug connection issues between")
    print("the brainstem and brain server.")

if __name__ == "__main__":
    test_connection_with_logging()