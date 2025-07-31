#!/usr/bin/env python3
"""
Debug Monitor - Simple tool to check what sessions are available
"""

import socket
import json
import sys


def send_json(sock, data):
    """Send JSON data to server"""
    message = json.dumps(data).encode('utf-8')
    sock.sendall(message + b'\n')


def receive_json(sock):
    """Receive JSON data from server"""
    data = b''
    while not data.endswith(b'\n'):
        chunk = sock.recv(4096)
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode('utf-8').strip())


def main():
    """Debug monitoring connection"""
    print("🔍 Debug Monitor")
    print("=" * 50)
    
    # Connect to monitoring server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(('localhost', 9998))
        
        # Receive welcome
        welcome = receive_json(sock)
        print(f"\n✅ Connected: {welcome.get('server', 'Unknown')}")
        print(f"Available commands: {welcome.get('commands', [])}")
        
        # Test different commands
        commands_to_test = [
            'session_info',
            'active_brains', 
            'telemetry',
            'connection_stats',
            'performance_metrics'
        ]
        
        for cmd in commands_to_test:
            print(f"\n📋 Testing command: {cmd}")
            print("-" * 40)
            
            send_json(sock, {'command': cmd})
            response = receive_json(sock)
            
            if response:
                print(f"Status: {response.get('status')}")
                data = response.get('data', {})
                
                if cmd == 'telemetry' and isinstance(data, dict):
                    print(f"Sessions with telemetry: {list(data.keys())}")
                    for session_id, telemetry in data.items():
                        print(f"  {session_id}: {telemetry.get('behavior_state', 'unknown')} state")
                elif cmd == 'session_info' and isinstance(data, list):
                    print(f"Active sessions: {len(data)}")
                    for session in data:
                        print(f"  - {session.get('session_id')} ({session.get('robot_type')})")
                elif cmd == 'active_brains' and isinstance(data, list):
                    print(f"Brain pool entries: {len(data)}")
                    for brain in data:
                        print(f"  - Profile: {brain.get('profile')}")
                else:
                    print(f"Data: {json.dumps(data, indent=2)[:500]}")
            else:
                print("No response")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())