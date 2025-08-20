#!/usr/bin/env python3
"""
Brain State Manager - Save and load brain states
"""

import os
import sys
import time
from datetime import datetime

sys.path.append('server/src')

def save_current_brain(brain_type='enhanced', name=None):
    """Save the current brain state"""
    from core.simple_factory import create_brain
    
    # Create brain instance
    brain = create_brain(brain_type=brain_type, target='speed', quiet=True)
    
    # Generate filename
    if name:
        filename = f"brain_states/{name}.brain"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brain_states/{brain_type}_{timestamp}.brain"
    
    # Create directory if needed
    os.makedirs("brain_states", exist_ok=True)
    
    # Save
    if brain.save_state(filename):
        print(f"✓ Brain saved to {filename}")
        return filename
    else:
        print(f"✗ Failed to save brain")
        return None

def load_brain_state(filepath):
    """Load a brain state"""
    from core.simple_factory import create_brain
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return None
    
    # Detect brain type from filename
    brain_type = 'enhanced' if 'enhanced' in filepath else 'critical_mass'
    
    # Create brain
    brain = create_brain(brain_type=brain_type, target='speed', quiet=True)
    
    # Load state
    if brain.load_state(filepath):
        print(f"✓ Brain loaded from {filepath}")
        return brain
    else:
        print(f"✗ Failed to load brain")
        return None

def list_saved_brains():
    """List all saved brain states"""
    if not os.path.exists("brain_states"):
        print("No saved brains found")
        return []
    
    files = [f for f in os.listdir("brain_states") if f.endswith('.brain')]
    
    if not files:
        print("No saved brains found")
        return []
    
    print("\nSaved Brain States:")
    print("-" * 60)
    
    for f in sorted(files):
        path = f"brain_states/{f}"
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        mtime = os.path.getmtime(path)
        age = (time.time() - mtime) / 3600  # hours
        
        print(f"  {f}")
        print(f"    Size: {size:.1f} MB")
        print(f"    Age: {age:.1f} hours ago")
    
    return files

def auto_save_setup():
    """Setup auto-save for the running brain"""
    print("""
Auto-Save Setup
===============

To enable auto-save, add this to your run_server.py:

    --save-interval 300  # Save every 5 minutes
    --save-on-exit      # Save when shutting down
    
Example:
    python3 run_server.py --brain enhanced --target speed --save-interval 300 --save-on-exit

Or modify the server to auto-save every N cycles:

    if self.telemetry.cycles % 1000 == 0:
        brain.save_state(f"brain_states/autosave_{self.telemetry.cycles}.brain")
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain State Manager')
    parser.add_argument('command', choices=['save', 'load', 'list', 'auto'],
                       help='Command to execute')
    parser.add_argument('--file', help='Brain state file')
    parser.add_argument('--name', help='Name for saved brain')
    parser.add_argument('--type', default='enhanced', 
                       choices=['enhanced', 'critical_mass'],
                       help='Brain type')
    
    args = parser.parse_args()
    
    if args.command == 'save':
        save_current_brain(args.type, args.name)
    
    elif args.command == 'load':
        if not args.file:
            print("Please specify --file")
        else:
            load_brain_state(args.file)
    
    elif args.command == 'list':
        list_saved_brains()
    
    elif args.command == 'auto':
        auto_save_setup()