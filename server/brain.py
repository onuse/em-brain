#!/usr/bin/env python3
"""
Simple Brain Server - Just run the brain.

No complex abstractions. Just a TCP server for one brain.
"""

import sys
import os
import argparse
import signal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.simple_brain_service import SimpleBrainService


def main():
    """Run the brain server."""
    parser = argparse.ArgumentParser(description='Simple Brain Server')
    parser.add_argument('--port', type=int, default=9999,
                       help='TCP port (default: 9999)')
    parser.add_argument('--config', choices=['speed', 'balanced', 'intelligence'],
                       default='balanced',
                       help='Brain configuration (default: balanced)')
    parser.add_argument('--safe-mode', action='store_true',
                       help='Use smaller brain for safety')
    parser.add_argument('--load', type=str,
                       help='Load brain state from file')
    parser.add_argument('--autosave', action='store_true',
                       help='Enable auto-saving every 1000 cycles')
    # Stream configuration is now negotiated with client, not set by server
    
    args = parser.parse_args()
    
    # Override config in safe mode
    if args.safe_mode:
        print("🦺 SAFE MODE: Using minimal brain configuration")
        args.config = 'speed'
    
    print("="*60)
    print("SIMPLE BRAIN SERVER")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print(f"Streams: Configured by client during handshake")
    print()
    
    # Create and run service
    service = SimpleBrainService(port=args.port, brain_config=args.config)
    
    # Load brain state if requested (note: brain will be created during handshake)
    if args.load:
        print(f"📁 Brain state will be loaded from {args.load} after handshake")
        service.brain_state_to_load = args.load
    else:
        service.brain_state_to_load = None
    
    # Handle shutdown
    def shutdown(sig, frame):
        print("\n📛 Shutting down...")
        # Save brain state on shutdown (if brain exists)
        if service.brain is not None:
            save_path = service.brain.save("shutdown_save.pt")
            print(f"💾 Saved brain state to {save_path}")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run server (streams configured by client)
    try:
        service.start()
    except KeyboardInterrupt:
        shutdown(None, None)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()