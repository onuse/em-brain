#!/usr/bin/env python3
"""
Minimal TCP test to debug connection issues between Windows and Raspberry Pi.
Tests if basic TCP communication works without the complex protocol.
"""

import socket
import sys
import time
import struct

def test_server(port=9999):
    """Simple test server that sends a response to any connection."""
    print(f"Starting test server on port {port}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    
    print(f"Listening on 0.0.0.0:{port}")
    
    while True:
        client, addr = server.accept()
        print(f"\n=== Client connected from {addr} ===")
        
        try:
            # Test 1: Receive data from client
            print("Test 1: Waiting for client data...")
            client.settimeout(5.0)
            data = client.recv(1024)
            print(f"Received {len(data)} bytes: {data}")
            
            # Test 2: Send immediate response
            print("Test 2: Sending immediate response...")
            response = b"SERVER_RESPONSE_123"
            sent = client.send(response)
            print(f"Sent {sent} bytes")
            
            # Test 3: Wait and send another message
            time.sleep(0.5)
            print("Test 3: Sending delayed message...")
            response2 = b"DELAYED_MESSAGE_456"
            sent2 = client.send(response2)
            print(f"Sent {sent2} bytes")
            
            # Test 4: Send structured data (like handshake)
            print("Test 4: Sending structured data...")
            magic = 0x524F424F
            msg_data = struct.pack('!IIf', magic, 12, 3.14159)
            sent3 = client.send(msg_data)
            print(f"Sent {sent3} bytes of structured data")
            
            # Keep connection open for a bit
            print("Keeping connection open for 2 seconds...")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Closing client connection")
            client.close()

def test_client(host, port=9999):
    """Simple test client that connects and receives data."""
    print(f"Connecting to {host}:{port}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    
    try:
        # Connect
        sock.connect((host, port))
        print(f"Connected to {host}:{port}")
        
        # Test 1: Send data
        print("Test 1: Sending data to server...")
        sock.send(b"CLIENT_HELLO")
        
        # Test 2: Receive immediate response
        print("Test 2: Waiting for immediate response...")
        data = sock.recv(1024)
        print(f"Received {len(data)} bytes: {data}")
        
        # Test 3: Receive delayed message
        print("Test 3: Waiting for delayed message...")
        data2 = sock.recv(1024)
        print(f"Received {len(data2)} bytes: {data2}")
        
        # Test 4: Receive structured data
        print("Test 4: Waiting for structured data...")
        data3 = sock.recv(1024)
        if len(data3) >= 12:
            magic, value, pi = struct.unpack('!IIf', data3[:12])
            print(f"Received structured: magic=0x{magic:08X}, value={value}, pi={pi}")
        else:
            print(f"Received only {len(data3)} bytes")
        
        print("Test complete!")
        
    except socket.timeout:
        print("ERROR: Socket timeout - no data received!")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python test_tcp_connection.py server [port]")
        print("  Client: python test_tcp_connection.py client <host> [port]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
        test_server(port)
    elif mode == "client":
        if len(sys.argv) < 3:
            print("Error: client mode requires host address")
            sys.exit(1)
        host = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
        test_client(host, port)
    else:
        print(f"Unknown mode: {mode}")