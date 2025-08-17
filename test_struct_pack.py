#!/usr/bin/env python3
"""Test if struct.pack works correctly"""

import struct

# Test what the server is trying to do
response_capabilities = [1.0, 12.0, 6.0, 1.0, 15.0]

print("Testing struct.pack formats:")

# Original broken version (would crash)
try:
    bad_format = f'{len(response_capabilities)}f'
    print(f"1. f-string format: '{bad_format}'")
    data = struct.pack(bad_format, *response_capabilities)
    print(f"   ERROR: This shouldn't work! Got {len(data)} bytes")
except Exception as e:
    print(f"   ✓ Failed as expected: {e}")

# Our fix
try:
    good_format = str(len(response_capabilities)) + 'f'
    print(f"\n2. String concat format: '{good_format}'")
    data = struct.pack(good_format, *response_capabilities)
    print(f"   ✓ Success! Packed {len(data)} bytes")
    print(f"   Hex: {data.hex()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Alternative that definitely works
try:
    print(f"\n3. Explicit format: '5f'")
    data = struct.pack('5f', *response_capabilities)
    print(f"   ✓ Success! Packed {len(data)} bytes")
    print(f"   Hex: {data.hex()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# What about network byte order?
try:
    print(f"\n4. Network byte order: '!5f'")
    data = struct.pack('!5f', *response_capabilities)
    print(f"   ✓ Success! Packed {len(data)} bytes")
    print(f"   Hex: {data.hex()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Build complete message like server does
print("\n5. Complete handshake message:")
vector_data = struct.pack('5f', *response_capabilities)
message_length = 1 + 4 + len(vector_data)
response_header = struct.pack('!IIBI', 
    0x524F424F,  # Magic ('ROBO')
    message_length,  # Length
    2,  # MSG_HANDSHAKE
    len(response_capabilities)  # Vector length
)
response_msg = response_header + vector_data
print(f"   Header: {len(response_header)} bytes")
print(f"   Vector: {len(vector_data)} bytes")
print(f"   Total: {len(response_msg)} bytes")
print(f"   Hex: {response_msg.hex()}")