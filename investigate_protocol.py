#!/usr/bin/env python3
"""
Investigate the exact protocol issue.
"""

import struct

print("INVESTIGATION: Why do we get 'Message too large: 2307426365'?")
print("=" * 70)

# The error number
error_val = 2307426365
print(f"\nError value: {error_val}")
print(f"As hex: 0x{error_val:08X}")
print(f"As bytes: {struct.pack('!I', error_val)}")

# What the client sends for sensory data (NEW protocol with magic bytes)
print("\n1. CLIENT SENDS (NEW protocol):")
print("-" * 40)

# For 307,212 sensory values
num_values = 307212
vector_size_bytes = num_values * 4  # Each float is 4 bytes
print(f"Sensory values: {num_values}")
print(f"Vector data size: {vector_size_bytes} bytes ({vector_size_bytes/1_000_000:.1f} MB)")

# NEW protocol message structure
MAGIC_BYTES = 0xDEADBEEF
MSG_SENSORY = 0
message_length = 1 + 4 + vector_size_bytes  # type(1) + vector_length(4) + data

print(f"\nMessage structure:")
print(f"  [0:4]   Magic bytes: 0x{MAGIC_BYTES:08X}")
print(f"  [4:8]   Message length: {message_length} (0x{message_length:08X})")
print(f"  [8:9]   Message type: {MSG_SENSORY}")
print(f"  [9:13]  Vector length: {num_values} (0x{num_values:08X})")
print(f"  [13:]   Vector data: {vector_size_bytes} bytes")

# Total size
total_size = 4 + 4 + 1 + 4 + vector_size_bytes  # magic + length + type + vec_len + data
print(f"\nTotal message size: {total_size} bytes ({total_size/1_000_000:.1f} MB)")

# Now let's see what could cause 2307426365
print("\n2. SERVER INTERPRETATION:")
print("-" * 40)

# If server reads this after handshake succeeds...
# The protocol_compat.py has self.client_uses_magic = True after handshake

# But wait - let me check what 0x898A583D could be
print(f"\nThe error value 0x{error_val:08X} could be:")

# Could it be reading part of the sensory data as length?
test_floats = [0.5] * 4  # If sensory data is 0.5 for all values
test_bytes = struct.pack('!ffff', *test_floats)
print(f"Four 0.5 floats as bytes: {test_bytes.hex()}")
test_as_int = struct.unpack('!I', test_bytes[:4])[0]
print(f"First 4 bytes as int: {test_as_int} (0x{test_as_int:08X})")

# Check if 2307426365 matches
if test_as_int == error_val:
    print("MATCH! The server is reading float data as message length!")
else:
    print(f"No match. Let's check what float would give us {error_val}...")
    
    # Reverse engineer what float value would give this
    bytes_val = struct.pack('!I', error_val)
    as_float = struct.unpack('!f', bytes_val)[0]
    print(f"0x{error_val:08X} as float: {as_float}")
    
    # This is a huge number, not a typical sensor value
    
print("\n3. ROOT CAUSE HYPOTHESIS:")
print("-" * 40)
print("The protocol_compat.py might be:")
print("1. Not persisting client_uses_magic between messages")
print("2. Each message creates a new protocol instance") 
print("3. Or there's a logic error in how it handles subsequent messages")

print("\n4. THE REAL ISSUE:")
print("-" * 40)
print("In clean_tcp_server.py, we create ONE protocol instance per client:")
print("  protocol = self.protocol_class()")
print("")
print("But BackwardCompatibleProtocol stores client_uses_magic as instance variable.")
print("This SHOULD work... unless the instance is being recreated somehow.")

# Let's check the actual message the client would send
print("\n5. ACTUAL BYTE SEQUENCE:")
print("-" * 40)
print("Client sends for sensory data:")
header = struct.pack('!IIBI', MAGIC_BYTES, message_length, MSG_SENSORY, num_values)
print(f"Header (13 bytes): {header.hex()}")
print(f"  Magic: {header[0:4].hex()}")
print(f"  Length: {header[4:8].hex()} = {struct.unpack('!I', header[4:8])[0]}")
print(f"  Type: {header[8:9].hex()} = {header[8]}")
print(f"  VecLen: {header[9:13].hex()} = {struct.unpack('!I', header[9:13])[0]}")

print("\nIf server reads at wrong offset, it might read:")
print(f"  Bytes [4:8] as first word: {struct.unpack('!I', header[4:8])[0]} (message length)")
print(f"  Bytes [8:12] as first word: {struct.unpack('!I', header[8:12])[0]} (type + part of veclen)")
print(f"  Bytes [9:13] as first word: {struct.unpack('!I', header[9:13])[0]} (vector length)")

# Check if any match our error
if struct.unpack('!I', header[8:12])[0] == error_val:
    print("\nFOUND IT! Server is reading at offset 8 (skipping magic+length)")