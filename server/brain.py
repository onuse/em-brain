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
    parser.add_argument('--no-vision', action='store_true',
                       help='Disable vision stream (port 10002)')
    parser.add_argument('--enable-audio', action='store_true',
                       help='Enable audio stream (port 10006)')
    parser.add_argument('--vision-res', type=str, default='320x240',
                       help='Vision resolution WxH (default: 320x240)')
    
    args = parser.parse_args()
    
    # Override config in safe mode
    if args.safe_mode:
        print("🦺 SAFE MODE: Using minimal brain configuration")
        args.config = 'speed'
    
    # Parse vision resolution
    try:
        w, h = map(int, args.vision_res.split('x'))
        vision_res = (w, h)
    except:
        vision_res = (320, 240)
        print(f"Invalid resolution '{args.vision_res}', using 320x240")
    
    print("="*60)
    print("SIMPLE BRAIN SERVER")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print(f"Vision: {'DISABLED' if args.no_vision else f'ENABLED ({vision_res[0]}x{vision_res[1]})'}")
    print(f"Audio: {'ENABLED' if args.enable_audio else 'DISABLED'}")
    print()
    
    # Create and run service
    service = SimpleBrainService(port=args.port, brain_config=args.config)
    
    # Load brain state if requested
    if args.load:
        if service.brain.load(args.load):
            print(f"✅ Loaded brain state from {args.load}")
        else:
            print(f"❌ Failed to load {args.load}, starting fresh")
    
    # Handle shutdown
    def shutdown(sig, frame):
        print("\n📛 Shutting down...")
        # Save brain state on shutdown
        save_path = service.brain.save("shutdown_save.pt")
        print(f"💾 Saved brain state to {save_path}")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run with stream configuration
    try:
        service.start(
            enable_vision=not args.no_vision,
            enable_audio=args.enable_audio,
            vision_resolution=vision_res
        )
    except KeyboardInterrupt:
        shutdown(None, None)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()