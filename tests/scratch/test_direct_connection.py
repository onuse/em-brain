#!/usr/bin/env python3
"""
Direct test of the protocol issue - sends exact bytes we expect.
"""

import socket
import struct
import time

def test_raw_protocol():
    """Test with raw bytes to diagnose the issue."""
    
    print("Testing Raw Protocol Communication")
    print("=" * 60)
    
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
    
    try:
        sock.connect(('localhost', 9999))
        print("✅ Connected to server")
        
        # Test 1: Send OLD protocol handshake (no magic bytes)
        print("\nTest 1: OLD Protocol (no magic bytes)")
        print("-" * 40)
        
        # OLD format: [length][type][vector_length][data]
        capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]
        vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
        message_length = 1 + 4 + len(vector_data)  # type + vector_length + data
        
        old_message = struct.pack('!IBI', message_length, 2, len(capabilities)) + vector_data
        
        print(f"Sending OLD format message: {len(old_message)} bytes")
        print(f"  Message length: {message_length}")
        print(f"  Message type: 2 (handshake)")
        print(f"  Vector length: {len(capabilities)}")
        print(f"  First 16 bytes (hex): {' '.join(f'{b:02X}' for b in old_message[:16])}")
        
        sock.sendall(old_message)
        print("✅ Sent handshake")
        
        # Wait for response
        time.sleep(0.5)
        
        # Try to receive response
        try:
            response = sock.recv(1024)
            print(f"✅ Received response: {len(response)} bytes")
            
            # Check what format server sent back
            if len(response) >= 4:
                first_word = struct.unpack('!I', response[:4])[0]
                if first_word == 0xDEADBEEF:
                    print("  Server responded with NEW format (magic bytes)")
                else:
                    print(f"  Server responded with OLD format (length={first_word})")
                    
        except socket.timeout:
            print("⏰ No response received (timeout)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sock.close()
        print("\n" + "=" * 60)

def test_new_protocol():
    """Test with NEW protocol (magic bytes)."""
    
    print("\nTesting NEW Protocol (with magic bytes)")
    print("=" * 60)
    
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        sock.connect(('localhost', 9999))
        print("✅ Connected to server")
        
        # NEW format: [magic][length][type][vector_length][data]
        MAGIC_BYTES = 0xDEADBEEF
        capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]
        vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
        message_length = 1 + 4 + len(vector_data)
        
        new_message = struct.pack('!IIBI', MAGIC_BYTES, message_length, 2, len(capabilities)) + vector_data
        
        print(f"Sending NEW format message: {len(new_message)} bytes")
        print(f"  Magic bytes: 0x{MAGIC_BYTES:08X}")
        print(f"  Message length: {message_length}")
        print(f"  Message type: 2 (handshake)")
        print(f"  Vector length: {len(capabilities)}")
        print(f"  First 16 bytes (hex): {' '.join(f'{b:02X}' for b in new_message[:16])}")
        
        sock.sendall(new_message)
        print("✅ Sent handshake")
        
        # Wait for response
        time.sleep(0.5)
        
        # Try to receive response
        try:
            response = sock.recv(1024)
            print(f"✅ Received response: {len(response)} bytes")
            
            # Check format
            if len(response) >= 4:
                first_word = struct.unpack('!I', response[:4])[0]
                if first_word == 0xDEADBEEF:
                    print("  ✅ Server responded with NEW format (magic bytes)")
                else:
                    print(f"  Server responded with OLD format? (first word=0x{first_word:08X})")
                    
        except socket.timeout:
            print("⏰ No response received (timeout)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sock.close()
        print("\n" + "=" * 60)

def test_large_sensory_data():
    """Test sending actual sensory data (307,212 values)."""
    
    print("\nTesting Large Sensory Data (307,212 values)")
    print("=" * 60)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        sock.connect(('localhost', 9999))
        print("✅ Connected to server")
        
        # First send handshake
        print("\n1. Sending handshake...")
        MAGIC_BYTES = 0xDEADBEEF
        capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]
        vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
        message_length = 1 + 4 + len(vector_data)
        
        handshake = struct.pack('!IIBI', MAGIC_BYTES, message_length, 2, len(capabilities)) + vector_data
        sock.sendall(handshake)
        
        # Get response
        response = sock.recv(1024)
        print(f"   Received handshake response: {len(response)} bytes")
        
        # Now send sensory data
        print("\n2. Sending sensory data (307,212 values)...")
        sensory_data = [0.5] * 307212  # All gray pixels
        
        # This will be LARGE
        vector_data = struct.pack(f'{len(sensory_data)}f', *sensory_data)
        message_length = 1 + 4 + len(vector_data)
        
        print(f"   Total message size: {message_length + 8} bytes ({(message_length + 8) / 1_000_000:.1f} MB)")
        
        # Send header first
        header = struct.pack('!IIBI', MAGIC_BYTES, message_length, 0, len(sensory_data))
        sock.sendall(header)
        print(f"   Sent header: {len(header)} bytes")
        
        # Send data in chunks to avoid buffer issues
        chunk_size = 65536  # 64KB chunks
        sent = 0
        while sent < len(vector_data):
            chunk = vector_data[sent:sent + chunk_size]
            sock.sendall(chunk)
            sent += len(chunk)
            if sent % (1024 * 1024) == 0:  # Progress every 1MB
                print(f"   Sent {sent / 1_000_000:.1f} MB...")
        
        print(f"   ✅ Sent all {len(vector_data)} bytes of sensory data")
        
        # Get motor response
        response = sock.recv(1024)
        print(f"   ✅ Received motor response: {len(response)} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "old":
            test_raw_protocol()
        elif sys.argv[1] == "new":
            test_new_protocol()
        elif sys.argv[1] == "large":
            test_large_sensory_data()
        else:
            print("Usage: python test_direct_connection.py [old|new|large]")
    else:
        # Test all
        test_raw_protocol()
        test_new_protocol()
        test_large_sensory_data()