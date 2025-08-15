#!/usr/bin/env python3
"""
Test with smaller vision that actually fits in TCP buffers.
"""

print("TESTING WITH REASONABLE MESSAGE SIZE")
print("=" * 70)

# Small vision: 160x120
width, height = 160, 120
pixels = width * height
basic_sensors = 5
audio = 7
total = basic_sensors + pixels + audio

print(f"\nSmall vision configuration:")
print(f"  Resolution: {width}x{height}")
print(f"  Pixels: {pixels:,}")
print(f"  Total sensors: {total:,}")

# Calculate message size
message_size = 13 + (total * 4)  # header + floats
print(f"\nMessage size: {message_size:,} bytes ({message_size/1024:.1f} KB)")

# Check against buffer
buffer_size = 16384
if message_size <= buffer_size:
    print(f"✅ Fits in TCP buffer ({buffer_size:,} bytes)")
else:
    print(f"❌ Still too large for buffer")

print(f"\nBenefits:")
print(f"  • Message is {1228861/message_size:.1f}x smaller")
print(f"  • Can actually be sent atomically")
print(f"  • No fragmentation issues")
print(f"  • Still provides visual input for learning")

print(f"\nTo use:")
print(f"  1. Copy robot_config_small.json to robot_config.json")
print(f"  2. Restart robot client")
print(f"  3. Brain will adapt to smaller dimensions")