#!/usr/bin/env python3
"""
Statistics Control Helper Script

Easy command-line control of statistics collection modes.
"""

import sys
import os
import json
import argparse

def create_config_file(mode: str):
    """Create a configuration file for the specified mode."""
    
    configs = {
        'production': {
            "enable_core_stats": True,
            "enable_stream_stats": False,
            "enable_coactivation_stats": False,
            "enable_column_stats": False,
            "enable_competition_stats": False,
            "enable_hierarchy_stats": False,
            "enable_performance_profiling": False,
            "enable_debug_stats": False
        },
        'performance': {
            "enable_core_stats": True,
            "enable_stream_stats": False,
            "enable_coactivation_stats": False,
            "enable_column_stats": False,
            "enable_competition_stats": False,
            "enable_hierarchy_stats": False,
            "enable_performance_profiling": True,
            "enable_debug_stats": False
        },
        'investigation': {
            "enable_core_stats": True,
            "enable_stream_stats": True,
            "enable_coactivation_stats": True,
            "enable_column_stats": True,
            "enable_competition_stats": True,
            "enable_hierarchy_stats": True,
            "enable_performance_profiling": True,
            "enable_debug_stats": False
        },
        'debug': {
            "enable_core_stats": True,
            "enable_stream_stats": True,
            "enable_coactivation_stats": True,
            "enable_column_stats": True,
            "enable_competition_stats": True,
            "enable_hierarchy_stats": True,
            "enable_performance_profiling": True,
            "enable_debug_stats": True
        }
    }
    
    if mode not in configs:
        print(f"❌ Unknown mode: {mode}")
        print(f"Available modes: {', '.join(configs.keys())}")
        return False
    
    config_file = 'statistics_config.json'
    with open(config_file, 'w') as f:
        json.dump(configs[mode], f, indent=2)
    
    print(f"✅ Created {config_file} for {mode} mode")
    return True

def set_env_vars(mode: str):
    """Print environment variables for the specified mode."""
    
    env_vars = {
        'production': {},
        'performance': {
            'BRAIN_ENABLE_PERFORMANCE_PROFILING': 'true'
        },
        'investigation': {
            'BRAIN_ENABLE_STREAM_STATS': 'true',
            'BRAIN_ENABLE_COACTIVATION_STATS': 'true',
            'BRAIN_ENABLE_COLUMN_STATS': 'true',
            'BRAIN_ENABLE_COMPETITION_STATS': 'true',
            'BRAIN_ENABLE_HIERARCHY_STATS': 'true',
            'BRAIN_ENABLE_PERFORMANCE_PROFILING': 'true'
        },
        'debug': {
            'BRAIN_ENABLE_STREAM_STATS': 'true',
            'BRAIN_ENABLE_COACTIVATION_STATS': 'true',
            'BRAIN_ENABLE_COLUMN_STATS': 'true',
            'BRAIN_ENABLE_COMPETITION_STATS': 'true',
            'BRAIN_ENABLE_HIERARCHY_STATS': 'true',
            'BRAIN_ENABLE_PERFORMANCE_PROFILING': 'true',
            'BRAIN_ENABLE_DEBUG_STATS': 'true'
        }
    }
    
    if mode not in env_vars:
        print(f"❌ Unknown mode: {mode}")
        return False
    
    print(f"🔧 Environment variables for {mode} mode:")
    
    if not env_vars[mode]:
        print("   (No environment variables needed - uses defaults)")
    else:
        for key, value in env_vars[mode].items():
            print(f"   export {key}={value}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Control brain statistics collection')
    parser.add_argument('mode', choices=['production', 'performance', 'investigation', 'debug'],
                       help='Statistics collection mode')
    parser.add_argument('--config', action='store_true',
                       help='Create configuration file')
    parser.add_argument('--env', action='store_true',
                       help='Show environment variables')
    
    args = parser.parse_args()
    
    if args.config:
        create_config_file(args.mode)
    
    if args.env:
        set_env_vars(args.mode)
    
    if not args.config and not args.env:
        # Default: create config file
        create_config_file(args.mode)

if __name__ == "__main__":
    main()