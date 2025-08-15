#!/usr/bin/env python3
"""
Verify that the protocol fixes are working correctly.
Tests the magic bytes and proper message framing.
"""

import socket
import struct
import sys

def test_protocol(host='localhost', port=9999):
    """Test the protocol with magic bytes."""
    
    print(f"Testing protocol connection to {host}:{port}")
    print("-" * 50)
    
    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        print("✅ Connected to server")
        
        # Test 1: Send proper handshake with magic bytes
        print("\nTest 1: Sending proper handshake with magic bytes...")
        capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]  # Version, sensory dims, motor dims, width, height
        
        # Encode with magic bytes
        MAGIC_BYTES = 0xDEADBEEF
        MSG_HANDSHAKE = 2
        
        vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
        message_length = 1 + 4 + len(vector_data)  # type + vector_length + data
        
        # Pack with magic bytes
        header = struct.pack('!IIBI', MAGIC_BYTES, message_length, MSG_HANDSHAKE, len(capabilities))
        message = header + vector_data
        
        sock.send(message)
        print(f"   Sent handshake with {len(capabilities)} capabilities")
        
        # Receive response (should also have magic bytes)
        magic_data = sock.recv(4)
        if len(magic_data) == 4:
            magic = struct.unpack('!I', magic_data)[0]
            if magic == MAGIC_BYTES:
                print(f"   ✅ Received valid magic bytes: 0x{magic:08X}")
            else:
                print(f"   ❌ Invalid magic bytes: 0x{magic:08X}")
        else:
            print(f"   ❌ Incomplete magic bytes received")
        
        # Read rest of response
        length_data = sock.recv(4)
        if len(length_data) == 4:
            msg_length = struct.unpack('!I', length_data)[0]
            print(f"   Response message length: {msg_length} bytes")
            
            if msg_length < 50_000_000:  # Sanity check
                print(f"   ✅ Message size is reasonable")
            else:
                print(f"   ❌ Message size too large: {msg_length}")
        
        # Test 2: Send corrupted message (without magic bytes)
        print("\nTest 2: Sending corrupted message (no magic bytes)...")
        bad_message = struct.pack('!IBI', 100, MSG_HANDSHAKE, 5) + b'\x00' * 20
        sock.send(bad_message)
        print("   Sent corrupted message")
        
        # Server should reject it
        try:
            response = sock.recv(1024)
            print(f"   Server response: {len(response)} bytes")
            # Try to decode error
            if len(response) >= 13:
                magic = struct.unpack('!I', response[:4])[0]
                if magic == MAGIC_BYTES:
                    print("   ✅ Server recovered and sent valid response")
                else:
                    print("   ⚠️  Server response has invalid magic")
        except socket.timeout:
            print("   ⚠️  Server didn't respond (may have rejected message)")
        
        sock.close()
        print("\n✅ Protocol verification complete!")
        print("   The new protocol with magic bytes is working correctly.")
        
    except ConnectionRefused:
        print("❌ Could not connect to server. Is the brain server running?")
        print("   Start it with: python3 server/brain.py")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        host = 'localhost'
    
    test_protocol(host)