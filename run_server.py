#!/usr/bin/env python3
"""Run the brain server."""

import argparse
from server.src.core.simple_brain_service import SimpleBrainService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the brain server')
    parser.add_argument('--port', type=int, default=9999, help='Server port (default: 9999)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--brain', type=str, default='enhanced', 
                       choices=['unified', 'critical_mass', 'emergence', 'enhanced'],
                       help='Brain type (default: enhanced with full learning systems)')
    parser.add_argument('--target', type=str, default='speed',
                       choices=['speed', 'balanced', 'intelligence'],
                       help='Optimization target (default: speed for faster processing)')
    
    # Auto-save options (sensible defaults for persistence)
    parser.add_argument('--save-interval', type=int, default=300,
                       help='Auto-save interval in seconds (default: 300, use 0 to disable)')
    parser.add_argument('--no-save-on-exit', action='store_true',
                       help='Disable saving brain state when disconnecting')
    parser.add_argument('--fresh-brain', action='store_true',
                       help='Start with a fresh brain instead of loading previous state')
    parser.add_argument('--load-state', type=str,
                       help='Path to specific brain state file to load (overrides auto-detection)')
    
    args = parser.parse_args()
    
    # Determine brain state path
    brain_state_path = args.load_state
    if not brain_state_path and not args.fresh_brain:
        # Auto-detect previous brain state
        import os
        default_path = f"brain_states/{args.brain}_autosave.brain"
        if os.path.exists(default_path):
            brain_state_path = default_path
            print(f"🧠 Found previous brain state: {default_path}")
        else:
            print(f"🆕 No previous brain state found, starting fresh")
    elif args.fresh_brain:
        print(f"🆕 Starting with fresh brain (--fresh-brain specified)")
    
    service = SimpleBrainService(
        port=args.port, 
        debug=args.debug,
        brain_type=args.brain,
        target=args.target,
        auto_save_interval=args.save_interval,
        save_on_exit=not args.no_save_on_exit,  # Inverted logic - save by default
        brain_state_path=brain_state_path
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()