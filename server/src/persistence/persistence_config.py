"""
Persistence Configuration and Policies

Defines configuration options and policies for the brain persistence subsystem.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ConsolidationPolicy:
    """Policies for when to trigger consolidation of incremental saves."""
    
    # File count trigger
    max_incremental_files: int = 50
    
    # Size trigger  
    max_incremental_size_mb: int = 100
    
    # Time trigger
    max_hours_since_consolidation: int = 24
    
    # Memory pressure trigger
    max_memory_usage_mb: int = 500
    
    # Force consolidation on clean shutdown
    consolidate_on_shutdown: bool = True


@dataclass 
class PersistenceConfig:
    """Complete configuration for the persistence subsystem."""
    
    # Base directory for all persistence files
    memory_root_path: str = "./server/robot_memory"
    
    # Incremental save frequency
    incremental_save_interval_cycles: int = 100
    incremental_save_interval_seconds: float = 30.0
    
    # File organization
    consolidated_dir: str = "consolidated"
    incremental_dir: str = "incremental"
    metadata_dir: str = "metadata"
    recovery_dir: str = "recovery"
    
    # Backup and safety
    keep_backup_snapshots: int = 5
    keep_incremental_history_days: int = 7
    enable_corruption_detection: bool = True
    enable_compression: bool = True
    
    # Consolidation policy
    consolidation_policy: ConsolidationPolicy = None
    
    # Performance settings
    background_save_thread: bool = True
    save_queue_max_size: int = 1000
    consolidation_thread: bool = True
    
    # Recovery settings
    max_recovery_attempts: int = 3
    recovery_validation: bool = True
    
    def __post_init__(self):
        """Initialize default consolidation policy if not provided."""
        if self.consolidation_policy is None:
            self.consolidation_policy = ConsolidationPolicy()
        
        # Ensure all paths are absolute
        if not os.path.isabs(self.memory_root_path):
            self.memory_root_path = os.path.abspath(self.memory_root_path)
    
    def get_consolidated_dir(self) -> str:
        """Get full path to consolidated files directory."""
        return os.path.join(self.memory_root_path, self.consolidated_dir)
    
    def get_incremental_dir(self) -> str:
        """Get full path to incremental files directory."""
        return os.path.join(self.memory_root_path, self.incremental_dir)
    
    def get_metadata_dir(self) -> str:
        """Get full path to metadata directory."""
        return os.path.join(self.memory_root_path, self.metadata_dir)
    
    def get_recovery_dir(self) -> str:
        """Get full path to recovery directory."""
        return os.path.join(self.memory_root_path, self.recovery_dir)
    
    def should_consolidate(self, incremental_count: int, 
                          incremental_size_mb: float,
                          hours_since_last: float,
                          memory_usage_mb: float = 0) -> bool:
        """Check if consolidation should be triggered based on current metrics."""
        policy = self.consolidation_policy
        
        return (
            incremental_count >= policy.max_incremental_files or
            incremental_size_mb >= policy.max_incremental_size_mb or
            hours_since_last >= policy.max_hours_since_consolidation or
            memory_usage_mb >= policy.max_memory_usage_mb
        )


def create_default_config(memory_path: str = "./server/robot_memory") -> PersistenceConfig:
    """Create a default persistence configuration."""
    return PersistenceConfig(
        memory_root_path=memory_path,
        incremental_save_interval_cycles=100,
        incremental_save_interval_seconds=30.0,
        consolidation_policy=ConsolidationPolicy(
            max_incremental_files=50,
            max_incremental_size_mb=100,
            max_hours_since_consolidation=24
        )
    )


def create_high_frequency_config(memory_path: str = "./server/robot_memory") -> PersistenceConfig:
    """Create a high-frequency save configuration for critical applications."""
    return PersistenceConfig(
        memory_root_path=memory_path,
        incremental_save_interval_cycles=10,  # Every 10 cycles
        incremental_save_interval_seconds=5.0,  # Every 5 seconds
        consolidation_policy=ConsolidationPolicy(
            max_incremental_files=20,  # Consolidate more frequently
            max_incremental_size_mb=50,
            max_hours_since_consolidation=12
        )
    )


def create_low_overhead_config(memory_path: str = "./server/robot_memory") -> PersistenceConfig:
    """Create a low-overhead configuration for resource-constrained environments."""
    return PersistenceConfig(
        memory_root_path=memory_path,
        incremental_save_interval_cycles=500,  # Less frequent
        incremental_save_interval_seconds=120.0,  # Every 2 minutes
        enable_compression=True,
        consolidation_policy=ConsolidationPolicy(
            max_incremental_files=100,  # Allow more files before consolidation
            max_incremental_size_mb=200,
            max_hours_since_consolidation=48
        )
    )