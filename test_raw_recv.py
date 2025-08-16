#!/usr/bin/env python3
"""
Test what's ACTUALLY being received on the socket.
No protocol, just raw bytes.
"""

import socket
import sys
import time

def raw_server(port=9999):
    """Server that sends a simple test message."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    
    print(f"Listening on port {port}")
    
    while True:
        client, addr = server.accept()
        print(f"Client connected from {addr}")
        
        # Wait for any data from client
        print("Waiting for client data...")
        data = client.recv(100)
        print(f"Received: {data.hex() if data else 'nothing'}")
        
        # Send test response
        response = b"HELLO_FROM_SERVER_123"
        print(f"Sending {len(response)} bytes: {response}")
        sent = client.send(response)
        print(f"Actually sent {sent} bytes")
        
        # Keep connection open
        time.sleep(2)
        client.close()
        print("Connection closed\n")

def raw_client(host, port=9999):
    """Client that receives ANY data available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    print(f"Connecting to {host}:{port}")
    sock.connect((host, port))
    print("Connected!")
    
    # Send something
    sock.send(b"CLIENT_TEST")
    print("Sent CLIENT_TEST")
    
    # Try multiple ways to receive
    print("\n=== Receiving Methods ===")
    
    # Method 1: Blocking recv with timeout
    print("\n1. Blocking recv(100) with 5s timeout:")
    sock.settimeout(5.0)
    try:
        data = sock.recv(100)
        print(f"   Got {len(data)} bytes: {data}")
        print(f"   Hex: {data.hex()}")
    except socket.timeout:
        print("   TIMEOUT - no data received")
    
    # Method 2: Non-blocking immediate check
    print("\n2. Non-blocking recv:")
    sock.setblocking(False)
    time.sleep(0.1)  # Give server time
    try:
        data = sock.recv(100)
        print(f"   Got {len(data)} bytes: {data}")
    except BlockingIOError:
        print("   No data available (BlockingIOError)")
    
    # Method 3: Receive exactly 1 byte
    print("\n3. Receive 1 byte at a time:")
    sock.setblocking(True)
    sock.settimeout(1.0)
    for i in range(5):
        try:
            byte = sock.recv(1)
            if byte:
                print(f"   Byte {i}: 0x{byte[0]:02X} ('{chr(byte[0]) if 32 <= byte[0] < 127 else '?'}')")
            else:
                print(f"   Connection closed at byte {i}")
                break
        except socket.timeout:
            print(f"   Timeout at byte {i}")
            break
    
    sock.close()
    print("\nDone")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python test_raw_recv.py server [port]")
        print("  Client: python test_raw_recv.py client <host> [port]")
        sys.exit(1)
    
    if sys.argv[1] == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
        raw_server(port)
    elif sys.argv[1] == "client":
        host = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
        raw_client(host, port)