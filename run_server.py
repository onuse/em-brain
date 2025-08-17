#!/usr/bin/env python3
"""Run the brain server."""

from server.src.core.simple_brain_service import SimpleBrainService

if __name__ == "__main__":
    service = SimpleBrainService(port=9999)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()