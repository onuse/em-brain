#!/usr/bin/env python3
"""
Performance Mode Manager

Utility for switching between different brain performance modes at runtime.
Provides command-line interface for switching between realtime/balanced/quality modes.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.append(str(src_dir))

SETTINGS_FILE = current_dir.parent / 'settings.json'

def load_settings() -> Dict[str, Any]:
    """Load current settings from settings.json."""
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(f"Settings file not found: {SETTINGS_FILE}")
    
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings back to settings.json."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def get_available_modes(settings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get available performance modes."""
    return settings.get('performance_modes', {})

def get_current_mode(settings: Dict[str, Any]) -> str:
    """Get current performance mode."""
    return settings.get('brain', {}).get('performance_mode', 'balanced')

def apply_performance_mode(settings: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    """Apply a performance mode to brain settings."""
    modes = get_available_modes(settings)
    
    if mode_name not in modes:
        available_modes = list(modes.keys())
        raise ValueError(f"Unknown performance mode '{mode_name}'. Available modes: {available_modes}")
    
    mode_config = modes[mode_name]
    brain_settings = settings.get('brain', {})
    
    # Apply mode-specific settings to brain configuration
    for key, value in mode_config.items():
        if key != 'description' and key != 'expected_performance_ms':
            brain_settings[key] = value
    
    # Update the performance mode indicator
    brain_settings['performance_mode'] = mode_name
    settings['brain'] = brain_settings
    
    return settings

def list_modes(settings: Dict[str, Any], show_details: bool = False) -> None:
    """List available performance modes."""
    modes = get_available_modes(settings)
    current = get_current_mode(settings)
    
    print("Available Performance Modes:")
    print("=" * 40)
    
    for mode_name, mode_config in modes.items():
        marker = "→" if mode_name == current else " "
        description = mode_config.get('description', 'No description')
        expected_ms = mode_config.get('expected_performance_ms', 'N/A')
        
        print(f"{marker} {mode_name.upper()}: {description}")
        print(f"    Expected performance: {expected_ms}ms")
        
        if show_details:
            print(f"    Configuration:")
            for key, value in mode_config.items():
                if key not in ['description', 'expected_performance_ms']:
                    print(f"      {key}: {value}")
        print()
    
    if current:
        print(f"Current mode: {current.upper()}")

def switch_mode(mode_name: str, dry_run: bool = False) -> None:
    """Switch to a different performance mode."""
    try:
        settings = load_settings()
        current_mode = get_current_mode(settings)
        
        if mode_name == current_mode:
            print(f"Already in {mode_name.upper()} mode")
            return
        
        # Apply the new mode
        updated_settings = apply_performance_mode(settings, mode_name)
        
        if dry_run:
            print(f"DRY RUN: Would switch from {current_mode.upper()} to {mode_name.upper()}")
            print("Changes that would be applied:")
            
            # Show what would change
            modes = get_available_modes(settings)
            mode_config = modes[mode_name]
            for key, value in mode_config.items():
                if key not in ['description', 'expected_performance_ms']:
                    current_value = settings.get('brain', {}).get(key, 'N/A')
                    if current_value != value:
                        print(f"  {key}: {current_value} → {value}")
        else:
            save_settings(updated_settings)
            mode_config = get_available_modes(settings)[mode_name]
            expected_ms = mode_config.get('expected_performance_ms', 'N/A')
            
            print(f"✅ Switched from {current_mode.upper()} to {mode_name.upper()} mode")
            print(f"Expected performance: {expected_ms}ms")
            print("\nRestart the brain server to apply changes.")
            
    except Exception as e:
        print(f"❌ Error switching performance mode: {e}")
        sys.exit(1)

def benchmark_mode(mode_name: str) -> None:
    """Show benchmark information for a specific mode."""
    try:
        settings = load_settings()
        modes = get_available_modes(settings)
        
        if mode_name not in modes:
            available_modes = list(modes.keys())
            print(f"Unknown mode '{mode_name}'. Available modes: {available_modes}")
            return
        
        mode_config = modes[mode_name]
        print(f"Performance Mode: {mode_name.upper()}")
        print("=" * 30)
        print(f"Description: {mode_config.get('description', 'N/A')}")
        print(f"Expected performance: {mode_config.get('expected_performance_ms', 'N/A')}ms")
        print(f"Spatial resolution: {mode_config.get('field_spatial_resolution', 'N/A')}")
        print(f"Enhanced dynamics: {mode_config.get('enable_enhanced_dynamics', 'N/A')}")
        print(f"Attention guidance: {mode_config.get('enable_attention_guidance', 'N/A')}")
        print(f"Hierarchical processing: {mode_config.get('enable_hierarchical_processing', 'N/A')}")
        
        if mode_config.get('enable_hierarchical_processing'):
            print(f"Hierarchical time budget: {mode_config.get('hierarchical_max_time_ms', 'N/A')}ms")
        
    except Exception as e:
        print(f"❌ Error getting benchmark info: {e}")

def main():
    parser = argparse.ArgumentParser(description='Manage brain performance modes')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available performance modes')
    list_parser.add_argument('--details', action='store_true', help='Show detailed configuration')
    
    # Switch command
    switch_parser = subparsers.add_parser('switch', help='Switch to a performance mode')
    switch_parser.add_argument('mode', help='Performance mode name (realtime/balanced/quality)')
    switch_parser.add_argument('--dry-run', action='store_true', help='Show what would change without applying')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Show benchmark info for a mode')
    benchmark_parser.add_argument('mode', help='Performance mode name')
    
    # Current command
    subparsers.add_parser('current', help='Show current performance mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            settings = load_settings()
            list_modes(settings, show_details=args.details)
            
        elif args.command == 'switch':
            switch_mode(args.mode.lower(), dry_run=args.dry_run)
            
        elif args.command == 'benchmark':
            benchmark_mode(args.mode.lower())
            
        elif args.command == 'current':
            settings = load_settings()
            current = get_current_mode(settings)
            modes = get_available_modes(settings)
            
            if current in modes:
                mode_config = modes[current]
                expected_ms = mode_config.get('expected_performance_ms', 'N/A')
                description = mode_config.get('description', 'N/A')
                print(f"Current mode: {current.upper()}")
                print(f"Description: {description}")
                print(f"Expected performance: {expected_ms}ms")
            else:
                print(f"Current mode: {current} (unknown)")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()