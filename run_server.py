#!/usr/bin/env python3
"""Run the brain server."""

import argparse
from server.src.core.simple_brain_service import SimpleBrainService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the brain server')
    parser.add_argument('--port', type=int, default=9999, help='Server port (default: 9999)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--brain', type=str, default='unified', 
                       choices=['unified', 'critical_mass', 'emergence', 'enhanced'],
                       help='Brain type: unified (original), critical_mass (emergence testing), enhanced (with full learning systems)')
    parser.add_argument('--target', type=str, default='balanced',
                       choices=['speed', 'balanced', 'intelligence'],
                       help='Optimization target: speed, balanced, or intelligence')
    args = parser.parse_args()
    
    service = SimpleBrainService(
        port=args.port, 
        debug=args.debug,
        brain_type=args.brain,
        target=args.target
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()