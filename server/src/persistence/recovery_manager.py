"""
Recovery Manager

Handles robust brain state recovery during startup:
1. Load latest consolidated snapshot (base state)
2. Find and apply all incremental changes since consolidation
3. Validate final state integrity
4. Handle corruption and recovery scenarios
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .brain_serializer import BrainSerializer, SerializedBrainState
from .storage_backend import StorageBackend
from .persistence_config import PersistenceConfig


class RecoveryResult:
    """Result of brain state recovery operation."""
    
    def __init__(self):
        self.success = False
        self.brain_state: Optional[SerializedBrainState] = None
        self.recovery_method = "none"
        self.base_source = ""
        self.incremental_files_applied = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.recovery_time_ms = 0.0
        self.patterns_recovered = 0
        self.total_experiences_recovered = 0


class RecoveryManager:
    """Manages robust brain state recovery with multiple fallback strategies."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.brain_serializer = BrainSerializer()
        self.storage = StorageBackend(config)
        
        # Recovery stats
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time_ms': 0.0,
            'corruption_recoveries': 0,
            'fallback_recoveries': 0,
            'incremental_files_processed': 0
        }
        
        # Ensure directories exist
        self.storage.ensure_directories_exist()
    
    def recover_brain_state(self) -> RecoveryResult:
        """
        Recover complete brain state using consolidated + incremental approach.
        
        Recovery Strategy:
        1. Load latest consolidated snapshot (base state)
        2. Find all incremental files since consolidation
        3. Apply incremental changes in chronological order
        4. Validate final state integrity
        5. Fallback strategies if corruption detected
        """
        start_time = time.perf_counter()
        result = RecoveryResult()
        
        try:
            print("🔄 Starting brain state recovery...")
            
            # Strategy 1: Load consolidated + apply incrementals
            result = self._recover_from_consolidated_plus_incrementals()
            
            if not result.success:
                # Strategy 2: Load most recent incremental file
                print("⚠️ Consolidated recovery failed, trying latest incremental...")
                result = self._recover_from_latest_incremental()
            
            if not result.success:
                # Strategy 3: Scan for any valid brain state file
                print("⚠️ Incremental recovery failed, scanning for any valid state...")
                result = self._recover_from_any_valid_file()
            
            if not result.success:
                # Strategy 4: Create fresh state
                print("⚠️ No valid state found, starting fresh...")
                result = self._create_fresh_state()
            
            # Record recovery time
            result.recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update statistics
            self._update_recovery_stats(result)
            
            # Report results
            self._report_recovery_results(result)
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Recovery failed with exception: {e}")
            result.recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"❌ Brain state recovery failed: {e}")
            self.recovery_stats['failed_recoveries'] += 1
            
            return result
    
    def _recover_from_consolidated_plus_incrementals(self) -> RecoveryResult:
        """Primary recovery strategy: Load consolidated base + apply incrementals."""
        result = RecoveryResult()
        
        try:
            # Step 1: Load latest consolidated snapshot
            base_state, base_source = self._load_latest_consolidated_snapshot()
            
            if not base_state:
                result.warnings.append("No consolidated snapshot found")
                return result
            
            result.base_source = base_source
            result.brain_state = base_state
            
            # Step 2: Find incremental files to apply
            incremental_files = self._find_incremental_files_since_base(base_state)
            
            if not incremental_files:
                # No incrementals to apply - base state is complete
                result.success = True
                result.recovery_method = "consolidated_only"
                result.patterns_recovered = len(base_state.patterns)
                result.total_experiences_recovered = base_state.total_experiences
                return result
            
            # Step 3: Apply incremental changes
            final_state = self._apply_incremental_files(base_state, incremental_files)
            
            if not final_state:
                result.errors.append("Failed to apply incremental files")
                return result
            
            # Step 4: Validate final state
            if not self._validate_brain_state(final_state):
                result.errors.append("Final state validation failed")
                return result
            
            result.success = True
            result.recovery_method = "consolidated_plus_incrementals"
            result.brain_state = final_state
            result.incremental_files_applied = len(incremental_files)
            result.patterns_recovered = len(final_state.patterns)
            result.total_experiences_recovered = final_state.total_experiences
            
            return result
            
        except Exception as e:
            result.errors.append(f"Consolidated+incremental recovery failed: {e}")
            return result
    
    def _recover_from_latest_incremental(self) -> RecoveryResult:
        """Fallback strategy: Use most recent incremental file."""
        result = RecoveryResult()
        
        try:
            # Find all incremental files (exclude integrity files)
            all_files = self.storage.list_files(
                self.config.get_incremental_dir(),
                "delta_*.json.gz"
            )
            incremental_files = [f for f in all_files if not f['filename'].endswith('.integrity')]
            
            if not incremental_files:
                result.warnings.append("No incremental files found")
                return result
            
            # Sort by creation time (newest first)
            incremental_files.sort(key=lambda x: x['created_time'], reverse=True)
            
            # Try to load the most recent incremental file
            for file_info in incremental_files:
                try:
                    brain_dict = self.storage.load_json(file_info['filepath'])
                    if brain_dict:
                        # Remove incremental metadata
                        brain_dict.pop('incremental_metadata', None)
                        
                        brain_state = self.brain_serializer.from_dict(brain_dict)
                        
                        if self._validate_brain_state(brain_state):
                            result.success = True
                            result.recovery_method = "latest_incremental"
                            result.brain_state = brain_state
                            result.base_source = file_info['filename']
                            result.patterns_recovered = len(brain_state.patterns)
                            result.total_experiences_recovered = brain_state.total_experiences
                            return result
                        else:
                            result.warnings.append(f"Validation failed for {file_info['filename']}")
                    
                except Exception as e:
                    result.warnings.append(f"Failed to load {file_info['filename']}: {e}")
                    continue
            
            result.errors.append("No valid incremental files found")
            return result
            
        except Exception as e:
            result.errors.append(f"Latest incremental recovery failed: {e}")
            return result
    
    def _recover_from_any_valid_file(self) -> RecoveryResult:
        """Emergency strategy: Scan all directories for any valid brain state."""
        result = RecoveryResult()
        
        try:
            # Scan all directories for brain state files
            search_dirs = [
                (self.config.get_consolidated_dir(), "brain_state_*.json.gz"),
                (self.config.get_incremental_dir(), "delta_*.json.gz"),
                (self.config.get_recovery_dir(), "*.json.gz")
            ]
            
            valid_files = []
            
            for search_dir, pattern in search_dirs:
                if os.path.exists(search_dir):
                    all_files = self.storage.list_files(search_dir, pattern)
                    files = [f for f in all_files if not f['filename'].endswith('.integrity')]
                    for file_info in files:
                        try:
                            # Try to load and validate
                            brain_dict = self.storage.load_json(file_info['filepath'])
                            if brain_dict:
                                brain_dict.pop('incremental_metadata', None)
                                brain_dict.pop('consolidation_metadata', None)
                                
                                brain_state = self.brain_serializer.from_dict(brain_dict)
                                
                                if self._validate_brain_state(brain_state):
                                    valid_files.append((file_info, brain_state))
                                    
                        except Exception as e:
                            result.warnings.append(f"Invalid file {file_info['filename']}: {e}")
                            continue
            
            if not valid_files:
                result.errors.append("No valid brain state files found anywhere")
                return result
            
            # Use the most recent valid file
            valid_files.sort(key=lambda x: x[0]['created_time'], reverse=True)
            best_file, best_state = valid_files[0]
            
            result.success = True
            result.recovery_method = "emergency_scan"
            result.brain_state = best_state
            result.base_source = best_file['filename']
            result.patterns_recovered = len(best_state.patterns)
            result.total_experiences_recovered = best_state.total_experiences
            result.warnings.append(f"Emergency recovery from {best_file['filename']}")
            
            return result
            
        except Exception as e:
            result.errors.append(f"Emergency scan recovery failed: {e}")
            return result
    
    def _create_fresh_state(self, brain_config: dict = None) -> RecoveryResult:
        """Last resort: Create a fresh brain state."""
        result = RecoveryResult()
        
        try:
            # Use provided brain config or defaults
            brain_type = "sparse_goldilocks"
            sensory_dim = 16
            motor_dim = 4
            temporal_dim = 4
            
            if brain_config:
                brain_type = brain_config.get('type', brain_type)
                sensory_dim = brain_config.get('sensory_dim', sensory_dim)
                motor_dim = brain_config.get('motor_dim', motor_dim)
                temporal_dim = brain_config.get('temporal_dim', temporal_dim)
            
            # Create minimal fresh brain state
            fresh_state = SerializedBrainState(
                version=self.brain_serializer.version,
                session_count=1,
                total_cycles=0,
                total_experiences=0,
                save_timestamp=time.time(),
                patterns=[],
                confidence_state={},
                hardware_adaptations={},
                cross_stream_associations={},
                brain_type=brain_type,
                sensory_dim=sensory_dim,
                motor_dim=motor_dim,
                temporal_dim=temporal_dim,
                learning_history=[],
                emergence_events=[]
            )
            
            result.success = True
            result.recovery_method = "fresh_state"
            result.brain_state = fresh_state
            result.base_source = "created_fresh"
            result.patterns_recovered = 0
            result.total_experiences_recovered = 0
            result.warnings.append("No existing state found - created fresh brain state")
            
            return result
            
        except Exception as e:
            result.errors.append(f"Failed to create fresh state: {e}")
            return result
    
    def _load_latest_consolidated_snapshot(self) -> Tuple[Optional[SerializedBrainState], str]:
        """Load the most recent consolidated snapshot."""
        try:
            # Check for latest symlink first
            latest_link = os.path.join(
                self.config.get_consolidated_dir(),
                "latest_consolidated.json"
            )
            
            if os.path.islink(latest_link) and os.path.exists(latest_link):
                brain_dict = self.storage.load_json(latest_link)
                if brain_dict:
                    brain_dict.pop('consolidation_metadata', None)
                    brain_state = self.brain_serializer.from_dict(brain_dict)
                    return brain_state, "latest_consolidated.json"
            
            # Fall back to finding most recent consolidated file (exclude integrity files)
            all_consolidated_files = self.storage.list_files(
                self.config.get_consolidated_dir(),
                "brain_state_*.json.gz"
            )
            consolidated_files = [f for f in all_consolidated_files if not f['filename'].endswith('.integrity')]
            
            if not consolidated_files:
                return None, ""
            
            # Get most recent file
            latest_file = max(consolidated_files, key=lambda x: x['created_time'])
            
            brain_dict = self.storage.load_json(latest_file['filepath'])
            if brain_dict:
                brain_dict.pop('consolidation_metadata', None)
                brain_state = self.brain_serializer.from_dict(brain_dict)
                return brain_state, latest_file['filename']
            
            return None, ""
            
        except Exception as e:
            print(f"⚠️ Failed to load consolidated snapshot: {e}")
            return None, ""
    
    def _find_incremental_files_since_base(self, base_state: SerializedBrainState) -> List[Dict[str, Any]]:
        """Find incremental files created after the base state."""
        try:
            # Get all incremental files (exclude integrity files)
            all_files = self.storage.list_files(
                self.config.get_incremental_dir(),
                "delta_*.json.gz"
            )
            incremental_files = [f for f in all_files if not f['filename'].endswith('.integrity')]
            
            # Filter files created after base state
            base_timestamp = base_state.save_timestamp
            relevant_files = [
                f for f in incremental_files 
                if f['created_time'] > base_timestamp
            ]
            
            # Sort by creation time (oldest first)
            relevant_files.sort(key=lambda x: x['created_time'])
            
            return relevant_files
            
        except Exception as e:
            print(f"⚠️ Failed to find incremental files: {e}")
            return []
    
    def _apply_incremental_files(self, base_state: SerializedBrainState, 
                                incremental_files: List[Dict[str, Any]]) -> Optional[SerializedBrainState]:
        """Apply incremental files to base state in chronological order."""
        try:
            current_state = base_state
            
            for file_info in incremental_files:
                print(f"🔀 Applying incremental: {file_info['filename']}")
                
                # Load incremental state
                brain_dict = self.storage.load_json(file_info['filepath'])
                if not brain_dict:
                    print(f"⚠️ Failed to load {file_info['filename']}")
                    continue
                
                brain_dict.pop('incremental_metadata', None)
                incremental_state = self.brain_serializer.from_dict(brain_dict)
                
                # Apply this incremental to current state
                current_state = self._merge_incremental_state(current_state, incremental_state)
                
                if not current_state:
                    print(f"⚠️ Failed to merge {file_info['filename']}")
                    return None
            
            return current_state
            
        except Exception as e:
            print(f"⚠️ Failed to apply incremental files: {e}")
            return None
    
    def _merge_incremental_state(self, base_state: SerializedBrainState, 
                                incremental_state: SerializedBrainState) -> Optional[SerializedBrainState]:
        """Merge an incremental state into the base state."""
        try:
            # Use incremental state's scalar values (more recent)
            merged_state = SerializedBrainState(
                version=incremental_state.version,
                session_count=incremental_state.session_count,
                total_cycles=incremental_state.total_cycles,
                total_experiences=incremental_state.total_experiences,
                save_timestamp=incremental_state.save_timestamp,
                patterns=[],  # Will merge below
                confidence_state=incremental_state.confidence_state,
                hardware_adaptations=incremental_state.hardware_adaptations,
                cross_stream_associations=incremental_state.cross_stream_associations,
                brain_type=incremental_state.brain_type,
                sensory_dim=incremental_state.sensory_dim,
                motor_dim=incremental_state.motor_dim,
                temporal_dim=incremental_state.temporal_dim,
                learning_history=incremental_state.learning_history,
                emergence_events=incremental_state.emergence_events
            )
            
            # Merge patterns (keep most recent version of each pattern)
            pattern_map = {}
            
            # Add base patterns
            for pattern in base_state.patterns:
                pattern_map[pattern.pattern_id] = pattern
            
            # Override with incremental patterns (more recent)
            for pattern in incremental_state.patterns:
                pattern_map[pattern.pattern_id] = pattern
            
            merged_state.patterns = list(pattern_map.values())
            
            return merged_state
            
        except Exception as e:
            print(f"⚠️ Failed to merge incremental state: {e}")
            return None
    
    def _validate_brain_state(self, brain_state: SerializedBrainState) -> bool:
        """Validate that brain state is consistent and usable."""
        try:
            # Basic structure validation
            if not isinstance(brain_state.patterns, list):
                return False
            
            if brain_state.total_experiences < 0 or brain_state.total_cycles < 0:
                return False
            
            if brain_state.session_count < 1:
                return False
            
            # Pattern validation
            for pattern in brain_state.patterns:
                if not hasattr(pattern, 'pattern_id') or not pattern.pattern_id:
                    return False
                
                if not hasattr(pattern, 'pattern_data'):
                    return False
            
            # Architecture validation
            if brain_state.sensory_dim <= 0 or brain_state.motor_dim <= 0:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Brain state validation failed: {e}")
            return False
    
    def _update_recovery_stats(self, result: RecoveryResult):
        """Update recovery performance statistics."""
        stats = self.recovery_stats
        stats['total_recoveries'] += 1
        
        if result.success:
            stats['successful_recoveries'] += 1
        else:
            stats['failed_recoveries'] += 1
        
        # Update average recovery time
        total_recoveries = stats['total_recoveries']
        stats['avg_recovery_time_ms'] = (
            (stats['avg_recovery_time_ms'] * (total_recoveries - 1) + result.recovery_time_ms) /
            total_recoveries
        )
        
        # Track specific recovery types
        if result.recovery_method in ['emergency_scan', 'latest_incremental']:
            stats['fallback_recoveries'] += 1
        
        if result.warnings or result.errors:
            stats['corruption_recoveries'] += 1
        
        stats['incremental_files_processed'] += result.incremental_files_applied
    
    def _report_recovery_results(self, result: RecoveryResult):
        """Report recovery results to console."""
        if result.success:
            print(f"✅ Brain state recovery successful:")
            print(f"   Method: {result.recovery_method}")
            print(f"   Source: {result.base_source}")
            print(f"   Patterns: {result.patterns_recovered}")
            print(f"   Experiences: {result.total_experiences_recovered}")
            print(f"   Incremental files applied: {result.incremental_files_applied}")
            print(f"   Recovery time: {result.recovery_time_ms:.1f}ms")
            
            if result.warnings:
                print(f"   Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3
                    print(f"     - {warning}")
        else:
            print(f"❌ Brain state recovery failed:")
            print(f"   Recovery time: {result.recovery_time_ms:.1f}ms")
            print(f"   Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3
                print(f"     - {error}")
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status."""
        # Check what files are available (exclude integrity files)
        all_consolidated_files = self.storage.list_files(
            self.config.get_consolidated_dir(),
            "brain_state_*.json.gz"
        )
        consolidated_files = [f for f in all_consolidated_files if not f['filename'].endswith('.integrity')]
        
        all_files = self.storage.list_files(
            self.config.get_incremental_dir(),
            "delta_*.json.gz"
        )
        incremental_files = [f for f in all_files if not f['filename'].endswith('.integrity')]
        
        return {
            'consolidated_snapshots_available': len(consolidated_files),
            'incremental_files_available': len(incremental_files),
            'latest_consolidated_size_mb': max(f['size_mb'] for f in consolidated_files) if consolidated_files else 0,
            'total_incremental_size_mb': sum(f['size_mb'] for f in incremental_files),
            'recovery_stats': self.recovery_stats
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery manager statistics."""
        return {
            **self.recovery_stats,
            **self.get_recovery_status()
        }