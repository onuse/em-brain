#!/usr/bin/env python3
"""
Verify that TCP buffer size is the actual issue.
"""

import socket
import struct

print("TCP BUFFER SIZE VERIFICATION")
print("=" * 70)

# Check system defaults
sock = socket.socket()
send_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
recv_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
sock.close()

print(f"\nSystem defaults:")
print(f"  Send buffer: {send_buf:,} bytes ({send_buf/1024:.1f} KB)")
print(f"  Recv buffer: {recv_buf:,} bytes ({recv_buf/1024:.1f} KB)")

# Calculate message size
num_values = 307212
message_size = 13 + (num_values * 4)  # header + data
print(f"\nMessage size:")
print(f"  Sensory data: {message_size:,} bytes ({message_size/1_000_000:.2f} MB)")

# Check if it fits
print(f"\nAnalysis:")
if message_size > send_buf:
    print(f"  ❌ Message ({message_size:,} bytes) is {message_size/send_buf:.1f}x larger than send buffer!")
    print(f"     This WILL cause partial sends and stream desynchronization")
    
    # How many sends would it take?
    chunks = (message_size + send_buf - 1) // send_buf
    print(f"  ❌ Would need {chunks} separate send operations")
    print(f"     But protocol expects atomic message delivery!")
else:
    print(f"  ✅ Message fits in send buffer")

print(f"\nTHE FIX:")
print(f"  Set buffer to at least {message_size:,} bytes (preferably 2MB)")
print(f"  Both client (SO_SNDBUF) and server (SO_RCVBUF)")

# Test what happens with larger buffer
print(f"\nTesting with 2MB buffer:")
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
new_send_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
sock.close()

print(f"  New send buffer: {new_send_buf:,} bytes ({new_send_buf/1_000_000:.1f} MB)")
if message_size <= new_send_buf:
    print(f"  ✅ Message now fits in buffer!")
else:
    print(f"  ❌ Still too small")