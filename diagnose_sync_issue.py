#!/usr/bin/env python3
"""
Diagnose why server sees 0xE1E0E03E instead of 0xDEADBEEF
"""

import struct

print("DIAGNOSIS: Server expects magic bytes but gets 0xE1E0E03E")
print("=" * 70)

# What the server saw
received = 0xE1E0E03E
print(f"Server received: 0x{received:08X}")
print(f"As float: {struct.unpack('!f', struct.pack('!I', received))[0]}")

# What client should send for sensory data
MAGIC_BYTES = 0xDEADBEEF
num_values = 307212
message_length = 1 + 4 + (num_values * 4)  # type + vec_len + data

print(f"\nClient should send:")
print(f"  [0:4]   0x{MAGIC_BYTES:08X} (magic bytes)")
print(f"  [4:8]   0x{message_length:08X} ({message_length} = message length)")
print(f"  [8:9]   0x00 (MSG_SENSORY_INPUT)")
print(f"  [9:13]  0x{num_values:08X} ({num_values} = vector length)")

# Build the header
header = struct.pack('!IIBI', MAGIC_BYTES, message_length, 0, num_values)
print(f"\nActual header bytes: {header.hex()}")

# Check different offsets
print(f"\nChecking different read offsets:")
for offset in range(0, 13):
    if offset + 4 <= len(header):
        val = struct.unpack('!I', header[offset:offset+4])[0]
        print(f"  Offset {offset}: 0x{val:08X}")
        if val == received:
            print(f"    ^^^ MATCH! Server is reading at offset {offset}")

# The value 0xE1E0E03E looks like float data
# Let's check what sensor values would produce this
test_float = 0.44  # A typical sensor value
test_bytes = struct.pack('!f', test_float)
print(f"\nFloat {test_float} as bytes: {test_bytes.hex()}")

# Maybe it's reading from the data portion?
print(f"\nTheory: Server might be reading from the data portion")
print(f"After sending handshake response, server expects next message")
print(f"But if there's buffered data or partial read, it could be offset")

# Check if it could be motor command values the server sent
motor_values = [0.0, 0.0, 0.0]  # Typical motor response
motor_bytes = struct.pack('!fff', *motor_values)
print(f"\nMotor response [0,0,0] as bytes: {motor_bytes.hex()}")

# The issue is likely:
print("\nROOT CAUSE HYPOTHESIS:")
print("-" * 40)
print("1. After handshake, server sends response")
print("2. Client sends sensory data immediately")
print("3. But server might have leftover data in buffer")
print("4. Or client might be sending data before fully reading response")
print("")
print("The value 0xE1E0E03E (as float: 0.44140625) looks like sensor data!")
print("This suggests the server is reading from the middle of the data portion")