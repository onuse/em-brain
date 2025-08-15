#!/usr/bin/env python3
"""
Debug protocol mismatch between client and server.
This will show exactly what bytes are being sent/received.
"""

import socket
import struct
import sys
import time

def debug_server_receive(port=9999):
    """Listen like the server and show what we receive."""
    print("Starting debug server on port", port)
    print("=" * 60)
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)
    
    print(f"Listening on port {port}...")
    print("Waiting for client connection...")
    
    client_sock, addr = server_sock.accept()
    print(f"\n✅ Client connected from {addr}")
    print("\nReceiving data...")
    print("-" * 40)
    
    # Read first 32 bytes to see what client is sending
    data = client_sock.recv(32)
    
    print(f"Raw bytes received ({len(data)} bytes):")
    print("Hex: " + " ".join(f"{b:02X}" for b in data))
    print("Raw: " + repr(data))
    
    if len(data) >= 4:
        # Try interpreting first 4 bytes different ways
        print("\nFirst 4 bytes interpreted as:")
        
        # As unsigned int (what magic bytes should be)
        val_uint = struct.unpack('!I', data[:4])[0]
        print(f"  Unsigned int (big-endian): 0x{val_uint:08X} ({val_uint})")
        
        if val_uint == 0xDEADBEEF:
            print("  ✅ THIS IS CORRECT MAGIC BYTES!")
            
            # Parse rest of message
            if len(data) >= 13:
                msg_len = struct.unpack('!I', data[4:8])[0]
                msg_type = data[8]
                vec_len = struct.unpack('!I', data[9:13])[0]
                print(f"\n  Message length: {msg_len}")
                print(f"  Message type: {msg_type}")
                print(f"  Vector length: {vec_len}")
                
                if vec_len == 307212:
                    print("  ✅ Vector length matches vision data!")
                elif vec_len == 5:
                    print("  ⚠️  Vector length is 5 (no vision data)")
            
        else:
            print(f"  ❌ NOT MAGIC BYTES (expected 0xDEADBEEF)")
            
            # Maybe it's old protocol (no magic bytes)?
            old_msg_len = val_uint
            if old_msg_len < 1000:  # Reasonable for old protocol
                print(f"\n  Could be OLD PROTOCOL with message length: {old_msg_len}")
                if len(data) >= 9:
                    old_msg_type = data[4]
                    old_vec_len = struct.unpack('!I', data[5:9])[0]
                    print(f"  Old message type: {old_msg_type}")
                    print(f"  Old vector length: {old_vec_len}")
        
        # As float (in case it's data)
        val_float = struct.unpack('!f', data[:4])[0]
        print(f"  Float: {val_float}")
    
    # Continue reading to see pattern
    print("\nReading more data...")
    more_data = client_sock.recv(64)
    if more_data:
        print(f"Next {len(more_data)} bytes:")
        print("Hex: " + " ".join(f"{b:02X}" for b in more_data[:32]))
    
    client_sock.close()
    server_sock.close()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    if val_uint == 0xDEADBEEF:
        print("✅ Client IS using new protocol with magic bytes")
    else:
        print("❌ Client is NOT using magic bytes - using OLD protocol")
        print("   The client needs to be updated with the new brain_client.py")
        print("\nFIX: On the robot, run:")
        print("   cd ~/em-brain")
        print("   git pull")
        print("   cd client_picarx")
        print("   python3 picarx_robot.py --brain-host <SERVER-IP>")

def debug_client_send(host='localhost', port=9999):
    """Send a test message like the client would."""
    print(f"Connecting to {host}:{port} as client...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    
    # Send WITH magic bytes (new protocol)
    print("Sending handshake WITH magic bytes (new protocol)...")
    
    capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]
    vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
    message_length = 1 + 4 + len(vector_data)
    
    # New protocol: magic + length + type + vector_length + data
    MAGIC_BYTES = 0xDEADBEEF
    MSG_HANDSHAKE = 2
    header = struct.pack('!IIBI', MAGIC_BYTES, message_length, MSG_HANDSHAKE, len(capabilities))
    message = header + vector_data
    
    print(f"Sending {len(message)} bytes")
    print("First 16 bytes (hex):", " ".join(f"{b:02X}" for b in message[:16]))
    sock.send(message)
    
    # Wait for response
    print("Waiting for response...")
    response = sock.recv(64)
    print(f"Received {len(response)} bytes")
    
    sock.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'client':
        # Act as client
        host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
        debug_client_send(host)
    else:
        # Act as server (default)
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
        debug_server_receive(port)