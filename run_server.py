#!/usr/bin/env python3
"""Run the brain server."""

import argparse
from server.src.core.simple_brain_service import SimpleBrainService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the brain server')
    parser.add_argument('--port', type=int, default=9999, help='Server port (default: 9999)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    service = SimpleBrainService(port=args.port, debug=args.debug)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()