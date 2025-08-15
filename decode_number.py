import struct

val = 1695496766
print(f"Number: {val}")
print(f"As hex: 0x{val:08X}")

# As float
bytes_val = struct.pack('!I', val)
float_val = struct.unpack('!f', bytes_val)[0]
print(f"As float: {float_val}")

# Check if it could be part of our expected data
print(f"\nExpected magic bytes: 0x{0xDEADBEEF:08X}")
print(f"This is NOT magic bytes")

# Could it be a sensory data value count?
print(f"\nAs message length for OLD protocol:")
print(f"  Would be {val} bytes = {val/1_000_000:.1f} MB")
print(f"  For 307,212 floats, we expect: {307212 * 4 + 5} bytes = {(307212 * 4 + 5)/1_000_000:.1f} MB")

# This is way too big, so it's corrupted data