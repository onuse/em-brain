#!/usr/bin/env python3
"""
Test script to show what adaptive hardware limits are being calculated
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.utils.hardware_adaptation import get_adaptive_cognitive_limits, get_hardware_adaptation
import psutil

def main():
    print("🧪 Testing Hardware Adaptation Limits")
    print("=" * 50)
    
    # Show current hardware
    print(f"💻 Hardware Info:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()
    
    # Get adaptive limits
    limits = get_adaptive_cognitive_limits()
    print(f"🎯 Adaptive Cognitive Limits:")
    for key, value in limits.items():
        print(f"   {key}: {value:,}")
    print()
    
    # Calculate what this means for total experiences
    similarity_limit = limits.get('similarity_search_limit', 500)
    max_per_cycle = limits.get('max_experiences_per_cycle', 30)
    
    # This is what the storage system would calculate
    base_limit = similarity_limit * 4
    max_experiences = max(1000, base_limit)
    cleanup_threshold = max_experiences + max_per_cycle * 2
    
    print(f"📊 Storage System Calculations:")
    print(f"   Base limit: {similarity_limit} * 4 = {base_limit:,}")
    print(f"   Max experiences: max(1000, {base_limit:,}) = {max_experiences:,}")
    print(f"   Cleanup threshold: {max_experiences:,} + {max_per_cycle} * 2 = {cleanup_threshold:,}")
    print()
    
    # Show memory implications (updated realistic estimate)
    bytes_per_experience = 200  # From updated hardware adaptation
    total_memory_mb = (max_experiences * bytes_per_experience) / (1024 * 1024)
    print(f"🧠 Memory Implications:")
    print(f"   Experiences: {max_experiences:,}")
    print(f"   Memory per experience: {bytes_per_experience} bytes")
    print(f"   Total memory: {total_memory_mb:.1f} MB")
    print(f"   % of available RAM: {(total_memory_mb / 1024) / (psutil.virtual_memory().available / (1024**3)) * 100:.1f}%")
    print()
    
    # Compare to hundreds of thousands
    target_experiences = 500000  # 500k experiences
    target_memory_mb = (target_experiences * bytes_per_experience) / (1024 * 1024)
    print(f"🎯 Target Scale (500k experiences):")
    print(f"   Memory needed: {target_memory_mb:.1f} MB ({target_memory_mb/1024:.1f} GB)")
    print(f"   Current limit allows: {(max_experiences / target_experiences) * 100:.1f}% of target")
    print(f"   Scale factor needed: {target_experiences / max_experiences:.1f}x")

if __name__ == "__main__":
    main()