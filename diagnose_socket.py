#!/usr/bin/env python3
"""
Diagnose why TCP data isn't being received.
Tests various socket configurations and sending methods.
"""

import socket
import time
import sys
import struct

def diagnostic_server(port=9998):
    """Server that tries multiple ways to send data."""
    print(f"Starting diagnostic server on port {port}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    
    print(f"Listening on 0.0.0.0:{port}")
    
    while True:
        client, addr = server.accept()
        print(f"\n{'='*60}")
        print(f"Client connected from {addr}")
        
        try:
            # Wait for client to send something first
            print("Waiting for client hello...")
            client.settimeout(5.0)
            hello = client.recv(100)
            print(f"Received: {hello}")
            
            # Test 1: Send with send()
            print("\nTest 1: Using send()")
            data1 = b"TEST1_SEND"
            sent = client.send(data1)
            print(f"  Sent {sent} bytes")
            time.sleep(0.1)
            
            # Test 2: Send with sendall()
            print("\nTest 2: Using sendall()")
            data2 = b"TEST2_SENDALL"
            client.sendall(data2)
            print(f"  Sent {len(data2)} bytes")
            time.sleep(0.1)
            
            # Test 3: Send with TCP_NODELAY
            print("\nTest 3: With TCP_NODELAY")
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            data3 = b"TEST3_NODELAY"
            client.sendall(data3)
            print(f"  Sent {len(data3)} bytes")
            time.sleep(0.1)
            
            # Test 4: Send in chunks
            print("\nTest 4: Sending in chunks")
            for i in range(3):
                chunk = f"CHUNK_{i}_".encode()
                client.send(chunk)
                print(f"  Sent chunk {i}: {len(chunk)} bytes")
                time.sleep(0.05)
            
            # Test 5: Send binary data like handshake
            print("\nTest 5: Binary protocol data")
            magic = 0x524F424F
            binary_data = struct.pack('!II', magic, 1234)
            client.sendall(binary_data)
            print(f"  Sent {len(binary_data)} bytes: {binary_data.hex()}")
            
            # Keep alive for a bit
            print("\nKeeping connection open for 3 seconds...")
            time.sleep(3)
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Closing connection")
            client.close()

def diagnostic_client(host, port=9998):
    """Client that tries to receive data."""
    print(f"Connecting to {host}:{port}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect with timeout
        sock.settimeout(10.0)
        sock.connect((host, port))
        print(f"Connected successfully")
        
        # Send hello
        print("Sending hello...")
        sock.send(b"HELLO_FROM_CLIENT")
        
        # Try different receive methods
        print("\nTrying to receive data...")
        
        # Method 1: Regular recv
        print("\nMethod 1: recv(1024)")
        sock.settimeout(2.0)
        try:
            data = sock.recv(1024)
            print(f"  Received {len(data)} bytes: {data}")
        except socket.timeout:
            print(f"  TIMEOUT - no data received")
        
        # Method 2: Non-blocking recv
        print("\nMethod 2: Non-blocking recv")
        sock.setblocking(False)
        time.sleep(0.5)  # Give server time to send
        try:
            data = sock.recv(1024)
            print(f"  Received {len(data)} bytes: {data}")
        except BlockingIOError:
            print(f"  WOULD_BLOCK - no data available")
        sock.setblocking(True)
        
        # Method 3: Peek at data
        print("\nMethod 3: MSG_PEEK")
        sock.settimeout(2.0)
        try:
            data = sock.recv(1024, socket.MSG_PEEK)
            print(f"  Peeked {len(data)} bytes: {data}")
            # Now actually read it
            data = sock.recv(1024)
            print(f"  Read {len(data)} bytes: {data}")
        except socket.timeout:
            print(f"  TIMEOUT - no data to peek")
        
        # Method 4: Read in small chunks
        print("\nMethod 4: Reading 1 byte at a time")
        sock.settimeout(1.0)
        for i in range(5):
            try:
                byte = sock.recv(1)
                if byte:
                    print(f"  Byte {i}: {byte}")
                else:
                    print(f"  Connection closed")
                    break
            except socket.timeout:
                print(f"  Timeout on byte {i}")
                break
        
        print("\nTest complete")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python diagnose_socket.py server [port]")
        print("  Client: python diagnose_socket.py client <host> [port]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9998
        diagnostic_server(port)
    elif mode == "client":
        if len(sys.argv) < 3:
            print("Error: client mode requires host address")
            sys.exit(1)
        host = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9998
        diagnostic_client(host, port)