#!/usr/bin/env python3
"""
Simple Field Injection Test - Verify basic concept works.
"""

import sys
import time
import torch
import threading
import socket
import struct

# Add server src to path
sys.path.insert(0, 'server/src')

from streams.field_injection_threads import BatteryFieldInjector


def send_battery_updates():
    """Send some battery updates via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(10):
        voltage = 7.4 - (i * 0.1)  # Simulate discharge
        packet = struct.pack('!f', voltage)
        sock.sendto(packet, ('localhost', 10004))
        print(f"  Sent: {voltage:.2f}V")
        time.sleep(0.1)
    sock.close()


def main():
    print("SIMPLE FIELD INJECTION TEST")
    print("=" * 40)
    
    # Create a simple field tensor
    print("\n1. Creating field tensor...")
    field = torch.zeros(8, 8, 8, 16)  # Small field
    print(f"   Field shape: {field.shape}")
    print(f"   Initial value at [0,0,0,-1]: {field[0,0,0,-1].item():.3f}")
    
    # Create battery injector
    print("\n2. Creating battery injector...")
    injector = BatteryFieldInjector(field, port=10004)
    
    # Start injection thread
    print("\n3. Starting injection thread...")
    injector.start()
    time.sleep(0.5)  # Let it initialize
    
    # Send battery updates in background
    print("\n4. Sending battery updates...")
    sender = threading.Thread(target=send_battery_updates, daemon=True)
    sender.start()
    
    # Watch field change
    print("\n5. Monitoring field changes...")
    for i in range(15):
        value = field[0,0,0,-1].item()
        print(f"   t={i*0.2:.1f}s: field[0,0,0,-1] = {value:.4f}")
        time.sleep(0.2)
    
    # Stop injector
    print("\n6. Stopping injector...")
    injector.stop()
    
    # Check results
    print("\n" + "=" * 40)
    print("RESULTS:")
    final_value = field[0,0,0,-1].item()
    print(f"  Final field value: {final_value:.4f}")
    print(f"  Total injections: {injector.injection_count}")
    
    if final_value != 0 and injector.injection_count > 0:
        print("\n✅ SUCCESS! Field injection works!")
        print("  Battery sensor successfully modified field in parallel thread")
    else:
        print("\n❌ FAILED - No field modification detected")


if __name__ == "__main__":
    main()