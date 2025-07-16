#!/usr/bin/env python3
"""Stress test to trigger error 5.0 by overwhelming the brain server."""

import sys
import time
import threading
import numpy as np
from pathlib import Path

brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication.client import MinimalBrainClient

def stress_client(client_id, num_requests=50):
    """Stress test with rapid concurrent requests."""
    client = MinimalBrainClient()
    
    if not client.connect():
        print(f"Client {client_id}: Failed to connect")
        return
    
    errors = []
    successes = 0
    
    try:
        for i in range(num_requests):
            try:
                # Generate random sensory input
                sensory_input = [np.random.random() for _ in range(16)]
                
                # Very short timeout to stress the system
                action = client.get_action(sensory_input, timeout=0.01)
                successes += 1
                
            except Exception as e:
                errors.append(str(e))
                if "5.0" in str(e):
                    print(f"Client {client_id}: 🎯 Error 5.0 at request {i}: {e}")
                elif len(errors) % 10 == 0:
                    print(f"Client {client_id}: Error {len(errors)}: {e}")
                    
    finally:
        client.disconnect()
        print(f"Client {client_id}: {successes} successes, {len(errors)} errors")
        
    return errors

def main():
    print("🔥 Stress testing brain server to trigger error 5.0")
    print("=" * 60)
    
    # Start multiple concurrent clients
    threads = []
    for i in range(5):  # 5 concurrent clients
        thread = threading.Thread(target=stress_client, args=(i, 100))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    print("\n✅ Stress test complete")
    
    # Check for error logs
    print("\n📋 Checking error logs...")
    import os
    os.system("python3 server/tools/view_errors.py --limit 10")

if __name__ == "__main__":
    main()