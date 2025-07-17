"""
Statistics Collection Control System

This module provides clean feature flags to control when expensive statistics
gathering happens, ensuring zero performance impact in normal operation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import json


@dataclass
class StatisticsConfig:
    """Configuration for statistics collection levels."""
    
    # Core statistics (always fast, basic counters)
    enable_core_stats: bool = True
    
    # Stream statistics (expensive: pattern counting, similarity calculations)
    enable_stream_stats: bool = False
    
    # Coactivation statistics (very expensive: O(n²) tensor operations)
    enable_coactivation_stats: bool = False
    
    # Cortical column statistics (very expensive: pairwise similarity)
    enable_column_stats: bool = False
    
    # Competitive dynamics statistics (expensive: cluster analysis)
    enable_competition_stats: bool = False
    
    # Temporal hierarchy statistics (expensive: layer analysis)
    enable_hierarchy_stats: bool = False
    
    # Performance profiling (expensive: detailed timing)
    enable_performance_profiling: bool = False
    
    # Debug statistics (very expensive: full state dumps)
    enable_debug_stats: bool = False


class StatisticsController:
    """
    Centralized controller for statistics collection.
    
    This ensures zero performance impact when statistics are disabled
    by completely avoiding method calls, not just returning empty results.
    """
    
    def __init__(self, config: Optional[StatisticsConfig] = None):
        self.config = config or StatisticsConfig()
        self._load_config_from_env()
    
    def _load_config_from_env(self):
        """Load configuration from environment variables or config file."""
        # Check for environment variables
        if os.getenv('BRAIN_ENABLE_STREAM_STATS') == 'true':
            self.config.enable_stream_stats = True
        
        if os.getenv('BRAIN_ENABLE_COACTIVATION_STATS') == 'true':
            self.config.enable_coactivation_stats = True
        
        if os.getenv('BRAIN_ENABLE_COLUMN_STATS') == 'true':
            self.config.enable_column_stats = True
        
        if os.getenv('BRAIN_ENABLE_COMPETITION_STATS') == 'true':
            self.config.enable_competition_stats = True
        
        if os.getenv('BRAIN_ENABLE_HIERARCHY_STATS') == 'true':
            self.config.enable_hierarchy_stats = True
        
        if os.getenv('BRAIN_ENABLE_PERFORMANCE_PROFILING') == 'true':
            self.config.enable_performance_profiling = True
        
        if os.getenv('BRAIN_ENABLE_DEBUG_STATS') == 'true':
            self.config.enable_debug_stats = True
        
        # Check for config file
        config_file = os.getenv('BRAIN_STATS_CONFIG', 'statistics_config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    
                # Update config with file values
                for key, value in file_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            except (json.JSONDecodeError, IOError):
                # Continue with default/env config if file is invalid
                pass
    
    def should_collect_stream_stats(self) -> bool:
        """Check if stream statistics should be collected."""
        return self.config.enable_stream_stats
    
    def should_collect_coactivation_stats(self) -> bool:
        """Check if coactivation statistics should be collected."""
        return self.config.enable_coactivation_stats
    
    def should_collect_column_stats(self) -> bool:
        """Check if cortical column statistics should be collected."""
        return self.config.enable_column_stats
    
    def should_collect_competition_stats(self) -> bool:
        """Check if competitive dynamics statistics should be collected."""
        return self.config.enable_competition_stats
    
    def should_collect_hierarchy_stats(self) -> bool:
        """Check if temporal hierarchy statistics should be collected."""
        return self.config.enable_hierarchy_stats
    
    def should_enable_performance_profiling(self) -> bool:
        """Check if performance profiling should be enabled."""
        return self.config.enable_performance_profiling
    
    def should_collect_debug_stats(self) -> bool:
        """Check if debug statistics should be collected."""
        return self.config.enable_debug_stats
    
    def get_core_stats_only(self) -> Dict[str, Any]:
        """Get only core statistics that are always fast."""
        return {
            'core_stats_enabled': True,
            'stream_stats_enabled': self.config.enable_stream_stats,
            'coactivation_stats_enabled': self.config.enable_coactivation_stats,
            'column_stats_enabled': self.config.enable_column_stats,
            'competition_stats_enabled': self.config.enable_competition_stats,
            'hierarchy_stats_enabled': self.config.enable_hierarchy_stats,
            'performance_profiling_enabled': self.config.enable_performance_profiling,
            'debug_stats_enabled': self.config.enable_debug_stats
        }
    
    def create_investigation_config(self) -> 'StatisticsConfig':
        """Create configuration for problem investigation (enables most stats)."""
        return StatisticsConfig(
            enable_core_stats=True,
            enable_stream_stats=True,
            enable_coactivation_stats=True,
            enable_column_stats=True,
            enable_competition_stats=True,
            enable_hierarchy_stats=True,
            enable_performance_profiling=True,
            enable_debug_stats=False  # Keep this off unless specifically needed
        )
    
    def create_performance_config(self) -> 'StatisticsConfig':
        """Create configuration for performance investigation."""
        return StatisticsConfig(
            enable_core_stats=True,
            enable_stream_stats=False,
            enable_coactivation_stats=False,
            enable_column_stats=False,
            enable_competition_stats=False,
            enable_hierarchy_stats=False,
            enable_performance_profiling=True,
            enable_debug_stats=False
        )
    
    def create_production_config(self) -> 'StatisticsConfig':
        """Create configuration for production (minimal stats)."""
        return StatisticsConfig(
            enable_core_stats=True,
            enable_stream_stats=False,
            enable_coactivation_stats=False,
            enable_column_stats=False,
            enable_competition_stats=False,
            enable_hierarchy_stats=False,
            enable_performance_profiling=False,
            enable_debug_stats=False
        )


# Global instance for easy access
_global_stats_controller = StatisticsController()


def get_stats_controller() -> StatisticsController:
    """Get the global statistics controller instance."""
    return _global_stats_controller


def set_stats_controller(controller: StatisticsController):
    """Set the global statistics controller instance."""
    global _global_stats_controller
    _global_stats_controller = controller


# Convenience functions for common checks
def should_collect_stream_stats() -> bool:
    """Check if stream statistics should be collected."""
    return _global_stats_controller.should_collect_stream_stats()


def should_collect_coactivation_stats() -> bool:
    """Check if coactivation statistics should be collected."""
    return _global_stats_controller.should_collect_coactivation_stats()


def should_collect_column_stats() -> bool:
    """Check if cortical column statistics should be collected."""
    return _global_stats_controller.should_collect_column_stats()


def should_collect_competition_stats() -> bool:
    """Check if competitive dynamics statistics should be collected."""
    return _global_stats_controller.should_collect_competition_stats()


def should_collect_hierarchy_stats() -> bool:
    """Check if temporal hierarchy statistics should be collected."""
    return _global_stats_controller.should_collect_hierarchy_stats()


def should_enable_performance_profiling() -> bool:
    """Check if performance profiling should be enabled."""
    return _global_stats_controller.should_enable_performance_profiling()


def should_collect_debug_stats() -> bool:
    """Check if debug statistics should be collected."""
    return _global_stats_controller.should_collect_debug_stats()


# Configuration helpers
def enable_investigation_mode():
    """Enable statistics collection for problem investigation."""
    controller = get_stats_controller()
    controller.config = controller.create_investigation_config()


def enable_performance_mode():
    """Enable statistics collection for performance investigation."""
    controller = get_stats_controller()
    controller.config = controller.create_performance_config()


def enable_production_mode():
    """Enable minimal statistics for production."""
    controller = get_stats_controller()
    controller.config = controller.create_production_config()


def print_current_config():
    """Print current statistics configuration."""
    controller = get_stats_controller()
    config = controller.config
    
    print("📊 STATISTICS COLLECTION CONFIG")
    print("=" * 35)
    print(f"Core stats: {'✅' if config.enable_core_stats else '❌'}")
    print(f"Stream stats: {'✅' if config.enable_stream_stats else '❌'}")
    print(f"Coactivation stats: {'✅' if config.enable_coactivation_stats else '❌'}")
    print(f"Column stats: {'✅' if config.enable_column_stats else '❌'}")
    print(f"Competition stats: {'✅' if config.enable_competition_stats else '❌'}")
    print(f"Hierarchy stats: {'✅' if config.enable_hierarchy_stats else '❌'}")
    print(f"Performance profiling: {'✅' if config.enable_performance_profiling else '❌'}")
    print(f"Debug stats: {'✅' if config.enable_debug_stats else '❌'}")