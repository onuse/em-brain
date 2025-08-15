#!/usr/bin/env python3
"""
Migrate JSON brain states to efficient binary format

This tool converts existing JSON brain state files to the new binary format,
reducing file sizes by ~20-50x and improving load times by ~10-100x.
"""

import sys
import os
from pathlib import Path
import time

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.persistence.binary_persistence import BinaryPersistence, migrate_json_to_binary


def main():
    """Migrate all JSON brain states to binary format"""
    print("🔄 Brain State Migration Tool")
    print("=" * 60)
    
    # Find brain memory directory
    memory_path = Path("./brain_memory")
    if not memory_path.exists():
        print("❌ No brain_memory directory found")
        return
    
    # Find JSON files
    json_files = list(memory_path.glob("brain_state_*.json"))
    if not json_files:
        print("✅ No JSON files to migrate")
        return
    
    print(f"\n📊 Found {len(json_files)} JSON brain state files")
    
    # Calculate total size
    total_json_size = sum(f.stat().st_size for f in json_files) / 1e9
    print(f"   Total size: {total_json_size:.2f} GB")
    
    # Sort by size (migrate largest first to free up space)
    json_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    
    # Show top 5 largest
    print("\n📈 Largest files:")
    for f in json_files[:5]:
        size_mb = f.stat().st_size / 1e6
        print(f"   {f.name}: {size_mb:.1f} MB")
    
    # Confirm migration
    print("\n⚠️  This will:")
    print("   1. Convert JSON files to binary format")
    print("   2. Keep original JSON files (delete manually after verification)")
    print("   3. Reduce storage by ~20-50x")
    
    response = input("\nProceed with migration? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return
    
    # Create binary persistence
    binary_persistence = BinaryPersistence(str(memory_path), use_compression=True)
    
    # Migrate each file
    success_count = 0
    total_binary_size = 0
    
    print("\n🚀 Starting migration...")
    print("-" * 60)
    
    for i, json_file in enumerate(json_files):
        print(f"\n[{i+1}/{len(json_files)}] {json_file.name}")
        
        # Get original size
        json_size = json_file.stat().st_size / 1e6
        print(f"   Original size: {json_size:.1f} MB")
        
        # Migrate
        if migrate_json_to_binary(json_file, binary_persistence):
            success_count += 1
            
            # Check new file size
            # Derive binary filename
            parts = json_file.stem.split('_')
            if len(parts) >= 4:
                session_id = f"{parts[2]}_{parts[3]}"
                cycles = parts[4] if len(parts) > 4 else "0"
            else:
                session_id = "migrated"
                cycles = "0"
            
            tensor_file = memory_path / f"brain_state_{session_id}_{cycles}_tensors.pt.gz"
            if tensor_file.exists():
                binary_size = tensor_file.stat().st_size / 1e6
                total_binary_size += binary_size / 1e3  # Convert to GB
                compression = (1 - binary_size / json_size) * 100
                print(f"   Binary size: {binary_size:.1f} MB")
                print(f"   Compression: {compression:.1f}%")
        else:
            print("   ❌ Migration failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully migrated: {success_count}/{len(json_files)} files")
    print(f"📉 Storage reduction: {total_json_size:.2f} GB → {total_binary_size:.2f} GB")
    print(f"💾 Space saved: {total_json_size - total_binary_size:.2f} GB")
    
    if success_count > 0:
        print("\n✅ Migration complete!")
        print("   Verify binary files work correctly before deleting JSON files")
        print("   To delete JSON files: rm brain_memory/brain_state_*.json")
    
    # Show storage stats
    stats = binary_persistence.get_storage_stats()
    print(f"\n📈 Storage growth history:")
    for entry in stats['growth_history'][-5:]:
        print(f"   Cycle {entry['cycles']}: {entry['size_mb']:.1f} MB")


if __name__ == "__main__":
    main()