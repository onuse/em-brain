#!/usr/bin/env python3
"""
Test to identify the ACTUAL issue, not guess.
"""

import socket
import struct
import time

def test_large_message():
    """Send a large message and see exactly what happens."""
    
    print("TESTING LARGE MESSAGE TRANSMISSION")
    print("=" * 70)
    
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 9999))
    
    # 1. Send handshake first
    print("\n1. Sending handshake...")
    MAGIC_BYTES = 0xDEADBEEF
    capabilities = [1.0, 307212.0, 3.0, 640.0, 480.0]
    vector_data = struct.pack(f'{len(capabilities)}f', *capabilities)
    message_length = 1 + 4 + len(vector_data)
    handshake = struct.pack('!IIBI', MAGIC_BYTES, message_length, 2, len(capabilities)) + vector_data
    
    print(f"   Handshake size: {len(handshake)} bytes")
    sent = sock.send(handshake)  # Use send, not sendall
    print(f"   Actually sent: {sent} bytes")
    
    if sent < len(handshake):
        print(f"   ⚠️  PARTIAL SEND! Only sent {sent} of {len(handshake)} bytes")
    
    # 2. Receive handshake response
    print("\n2. Receiving handshake response...")
    # Read magic bytes
    magic = sock.recv(4)
    if len(magic) == 4:
        magic_val = struct.unpack('!I', magic)[0]
        print(f"   Got magic: 0x{magic_val:08X}")
        
        # Read rest of response
        length_bytes = sock.recv(4)
        msg_len = struct.unpack('!I', length_bytes)[0]
        print(f"   Response length: {msg_len}")
        
        # Read the response data
        response = sock.recv(msg_len)
        print(f"   Got {len(response)} bytes of response")
    
    # 3. Now send large sensory data
    print("\n3. Sending large sensory data...")
    sensory_values = [0.5] * 307212  # 307,212 values
    vector_data = struct.pack(f'{len(sensory_values)}f', *sensory_values)
    message_length = 1 + 4 + len(vector_data)
    
    # Build complete message
    sensory_msg = struct.pack('!IIBI', MAGIC_BYTES, message_length, 0, len(sensory_values)) + vector_data
    
    print(f"   Message size: {len(sensory_msg)} bytes ({len(sensory_msg)/1_000_000:.1f} MB)")
    print(f"   First 32 bytes (hex): {sensory_msg[:32].hex()}")
    
    # Try sending it
    print(f"   Attempting to send...")
    sent = sock.send(sensory_msg)  # Use send to see how much actually sends
    print(f"   Actually sent: {sent} bytes")
    
    if sent < len(sensory_msg):
        print(f"   ⚠️  PARTIAL SEND! Only sent {sent} of {len(sensory_msg)} bytes")
        print(f"   Percentage sent: {sent/len(sensory_msg)*100:.1f}%")
        
        # What did we actually send?
        print(f"\n   Analysis:")
        print(f"   - Sent all header: {sent >= 13}")
        if sent >= 13:
            print(f"   - Sent {(sent - 13)/4} float values")
        
        # Try to send the rest
        print(f"\n   Trying to send remaining {len(sensory_msg) - sent} bytes...")
        remaining = sensory_msg[sent:]
        sent2 = sock.send(remaining)
        print(f"   Sent additional: {sent2} bytes")
        
        if sent2 < len(remaining):
            print(f"   ⚠️  STILL INCOMPLETE! Total sent: {sent + sent2} of {len(sensory_msg)}")
    else:
        print(f"   ✅ Sent complete message!")
    
    # 4. Try to receive response
    print("\n4. Waiting for motor response...")
    time.sleep(0.5)
    try:
        response = sock.recv(1024)
        print(f"   Got response: {len(response)} bytes")
    except socket.timeout:
        print(f"   No response (timeout)")
    
    sock.close()
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("If we see PARTIAL SEND, then the issue is TCP send buffer size.")
    print("If we see complete send but server still fails, it's a different issue.")

if __name__ == "__main__":
    test_large_message()