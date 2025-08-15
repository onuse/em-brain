#!/usr/bin/env python3
"""
Quick test to verify thread safety fixes work.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import socket
import json
import threading
from server.src.communication.client import MinimalBrainClient

def test_client(client_id, num_requests=10):
    """Test concurrent brain requests."""
    try:
        client = MinimalBrainClient(server_host='localhost', server_port=9999)
        
        for i in range(num_requests):
            # Simple sensory input (16D as expected)
            sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            
            try:
                action = client.get_action(sensory_input)
                print(f"Client {client_id} request {i}: SUCCESS")
                time.sleep(0.01)  # Small delay
            except Exception as e:
                print(f"Client {client_id} request {i}: ERROR - {e}")
                
    except Exception as e:
        print(f"Client {client_id}: Failed to connect - {e}")

def main():
    print("🧪 Testing thread safety fixes...")
    
    # Start multiple concurrent clients
    threads = []
    num_clients = 10
    
    for i in range(num_clients):
        thread = threading.Thread(target=test_client, args=(i, 10))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("✅ Thread safety test completed")
    
    # Check for new errors in log
    try:
        with open('server/logs/brain_errors.jsonl', 'r') as f:
            lines = f.readlines()
            
        print(f"📊 Total errors in log: {len(lines)}")
        
        # Look for recent dictionary errors
        recent_dict_errors = 0
        for line in lines[-20:]:  # Check last 20 entries
            if 'dictionary changed size during iteration' in line:
                recent_dict_errors += 1
                
        if recent_dict_errors == 0:
            print("✅ No recent dictionary iteration errors found!")
        else:
            print(f"❌ Found {recent_dict_errors} recent dictionary errors")
            
    except FileNotFoundError:
        print("ℹ️  No error log found")

if __name__ == "__main__":
    main()