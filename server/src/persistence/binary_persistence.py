"""
Binary Persistence for Brain State

Efficient binary format for saving/loading brain states using PyTorch's native format.
This replaces JSON serialization with binary storage, reducing file size by ~20-50x
and improving load/save times by ~10-100x.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import gzip
import shutil


class BinaryPersistence:
    """Handles efficient binary persistence of brain states"""
    
    def __init__(self, memory_path: str = "./brain_memory", use_compression: bool = True):
        """
        Initialize binary persistence.
        
        Args:
            memory_path: Directory for brain state files
            use_compression: Whether to use gzip compression (reduces size by ~3x)
        """
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression
        
    def save_brain_state(self, brain_state: Dict[str, Any], session_id: str, cycles: int) -> float:
        """
        Save brain state in efficient binary format.
        
        Returns:
            Save time in seconds
        """
        start_time = time.time()
        
        # Separate metadata from tensor data
        metadata = {}
        tensor_data = {}
        
        for key, value in brain_state.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                # Convert numpy to torch for consistent saving
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                tensor_data[key] = value
            elif key == 'unified_field' and isinstance(value, list):
                # Handle legacy list format - convert to tensor
                tensor_data[key] = torch.tensor(value, dtype=torch.float32)
            else:
                # Keep as metadata
                metadata[key] = value
        
        # Create filename
        timestamp = int(time.time())
        base_filename = f"brain_state_{session_id}_{cycles}"
        
        # Save metadata as small JSON
        metadata_file = self.memory_path / f"{base_filename}_meta.json"
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy types
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=convert_numpy)
        
        # Save tensor data in binary format
        tensor_file = self.memory_path / f"{base_filename}_tensors.pt"
        if self.use_compression:
            # Save to temporary file first
            temp_file = self.memory_path / f"{base_filename}_tensors_temp.pt"
            torch.save(tensor_data, temp_file)
            
            # Compress
            with open(temp_file, 'rb') as f_in:
                with gzip.open(f"{tensor_file}.gz", 'wb', compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove temp file
            temp_file.unlink()
            
            final_file = Path(f"{tensor_file}.gz")
        else:
            torch.save(tensor_data, tensor_file)
            final_file = tensor_file
        
        # Calculate sizes
        metadata_size = metadata_file.stat().st_size / 1e6
        tensor_size = final_file.stat().st_size / 1e6
        save_time = time.time() - start_time
        
        print(f"💾 Binary save complete: {base_filename}")
        print(f"   Metadata: {metadata_size:.1f} MB")
        print(f"   Tensors: {tensor_size:.1f} MB")
        print(f"   Time: {save_time:.1f}s")
        
        # Clean up old files
        self._cleanup_old_files()
        
        return save_time
    
    def load_brain_state(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load brain state from binary format.
        
        Args:
            session_id: Specific session to load, or None for latest
            
        Returns:
            Combined brain state dictionary or None if not found
        """
        start_time = time.time()
        
        # Find files to load
        if session_id:
            # Load specific session
            meta_files = list(self.memory_path.glob(f"brain_state_{session_id}_*_meta.json"))
        else:
            # Load latest
            meta_files = list(self.memory_path.glob("brain_state_*_meta.json"))
        
        if not meta_files:
            return None
        
        # Sort by modification time and get latest
        meta_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        meta_file = meta_files[0]
        
        # Derive tensor filename
        base_name = str(meta_file).replace('_meta.json', '')
        tensor_file = Path(f"{base_name}_tensors.pt")
        tensor_file_gz = Path(f"{base_name}_tensors.pt.gz")
        
        # Load metadata
        with open(meta_file, 'r') as f:
            brain_state = json.load(f)
        
        # Load tensor data
        if tensor_file_gz.exists():
            # Decompress and load
            with gzip.open(tensor_file_gz, 'rb') as f_in:
                tensor_data = torch.load(f_in, map_location='cpu')
        elif tensor_file.exists():
            tensor_data = torch.load(tensor_file, map_location='cpu')
        else:
            print(f"⚠️  Tensor file not found for {meta_file.name}")
            return None
        
        # Merge tensor data into brain state
        brain_state.update(tensor_data)
        
        load_time = time.time() - start_time
        
        # Get file info
        cycles = brain_state.get('brain_cycles', 0)
        tensor_size = (tensor_file_gz.stat().st_size if tensor_file_gz.exists() 
                      else tensor_file.stat().st_size) / 1e6
        
        print(f"🔄 Binary load complete: {meta_file.stem}")
        print(f"   Cycles: {cycles}")
        print(f"   Tensor size: {tensor_size:.1f} MB")  
        print(f"   Time: {load_time:.1f}s")
        
        return brain_state
    
    def _cleanup_old_files(self, keep_count: int = 10):
        """Keep only the most recent files"""
        # Clean up metadata files
        meta_files = list(self.memory_path.glob("brain_state_*_meta.json"))
        if len(meta_files) > keep_count:
            meta_files.sort(key=lambda f: f.stat().st_mtime)
            for f in meta_files[:-keep_count]:
                # Remove metadata
                f.unlink()
                
                # Remove associated tensor file
                base_name = str(f).replace('_meta.json', '')
                tensor_file = Path(f"{base_name}_tensors.pt")
                tensor_file_gz = Path(f"{base_name}_tensors.pt.gz")
                
                if tensor_file.exists():
                    tensor_file.unlink()
                if tensor_file_gz.exists():
                    tensor_file_gz.unlink()
                
                print(f"🗑️  Removed old state: {f.stem}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored brain states"""
        meta_files = list(self.memory_path.glob("brain_state_*_meta.json"))
        tensor_files = list(self.memory_path.glob("brain_state_*_tensors.pt*"))
        
        total_size = sum(f.stat().st_size for f in meta_files + tensor_files) / 1e6
        
        # Get size growth over time
        sizes_by_cycle = []
        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            base_name = str(meta_file).replace('_meta.json', '')
            tensor_file = Path(f"{base_name}_tensors.pt")
            tensor_file_gz = Path(f"{base_name}_tensors.pt.gz")
            
            size = 0
            if tensor_file_gz.exists():
                size = tensor_file_gz.stat().st_size / 1e6
            elif tensor_file.exists():
                size = tensor_file.stat().st_size / 1e6
            
            sizes_by_cycle.append({
                'cycles': meta.get('brain_cycles', 0),
                'size_mb': size,
                'timestamp': meta_file.stat().st_mtime
            })
        
        sizes_by_cycle.sort(key=lambda x: x['cycles'])
        
        return {
            'total_files': len(meta_files),
            'total_size_mb': total_size,
            'growth_history': sizes_by_cycle,
            'compression_enabled': self.use_compression
        }


def migrate_json_to_binary(json_file: Path, binary_persistence: BinaryPersistence) -> bool:
    """
    Migrate a JSON brain state file to binary format.
    
    Args:
        json_file: Path to JSON file
        binary_persistence: Binary persistence instance
        
    Returns:
        True if successful
    """
    print(f"\n🔄 Migrating {json_file.name} to binary format...")
    
    try:
        # Load JSON (this will be slow)
        start_time = time.time()
        with open(json_file, 'r') as f:
            brain_state = json.load(f)
        
        load_time = time.time() - start_time
        print(f"   JSON load time: {load_time:.1f}s")
        
        # Extract session info from filename
        # Format: brain_state_session_XXXXXXXX_NNN.json
        parts = json_file.stem.split('_')
        if len(parts) >= 4:
            session_id = f"{parts[2]}_{parts[3]}"
            cycles = int(parts[4]) if len(parts) > 4 else brain_state.get('brain_cycles', 0)
        else:
            session_id = "migrated"
            cycles = brain_state.get('brain_cycles', 0)
        
        # Save in binary format
        binary_persistence.save_brain_state(brain_state, session_id, cycles)
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False