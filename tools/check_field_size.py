#!/usr/bin/env python3
"""Check why the field is so massive."""

# Field shape from our dimension fix: [20, 20, 20, 10, 15, 8, 8, 6, 4, 5, 3]
shape = [20, 20, 20, 10, 15, 8, 8, 6, 4, 5, 3]
shape_small = [5, 5, 5, 10, 15, 8, 8, 6, 4, 5, 3]

print("Field shape analysis:")
print(f"Full shape: {shape}")
print(f"Small shape: {shape_small}")

# Calculate total elements
total = 1
total_small = 1
for s in shape:
    total *= s
for s in shape_small:
    total_small *= s
    
print(f"\nFull field elements: {total:,}")
print(f"Small field elements: {total_small:,}")

# Memory in MB (float32 = 4 bytes)
memory_mb = (total * 4) / (1024 * 1024)
memory_mb_small = (total_small * 4) / (1024 * 1024)

print(f"\nMemory usage:")
print(f"Full field: {memory_mb:.2f} MB")
print(f"Small field: {memory_mb_small:.2f} MB")

# The problem: we expanded singleton dimensions to large sizes!
print("\n⚠️  THE PROBLEM:")
print("We replaced 32 singleton dimensions (size 1) with:")
print("- 8, 8, 6, 4, 5, 3")
print(f"This multiplied the size by {8*8*6*4*5*3:,}x !")
print("\nOriginal field was probably ~3MB, now it's >1GB!")