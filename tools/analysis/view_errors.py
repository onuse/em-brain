#!/usr/bin/env python3
"""
View recent brain errors from log files

Usage:
  python3 tools/view_errors.py              # View last 10 errors
  python3 tools/view_errors.py --limit 20   # View last 20 errors
  python3 tools/view_errors.py --watch      # Watch for new errors
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

def view_recent_errors(limit: int = 10, watch: bool = False):
    """View recent errors from the error log file."""
    
    # Adjust path based on current working directory
    error_log_file = Path("logs/brain_errors.jsonl")
    if not error_log_file.exists():
        # Try from server directory
        error_log_file = Path("../logs/brain_errors.jsonl")
        if not error_log_file.exists():
            print("❌ No error log file found")
            print("   Expected locations:")
            print("     - logs/brain_errors.jsonl")
            print("     - ../logs/brain_errors.jsonl")
            return
    
    print(f"📋 Recent Brain Errors (from {error_log_file})")
    print("=" * 80)
    
    def read_errors():
        """Read errors from file."""
        try:
            with open(error_log_file, 'r') as f:
                lines = f.readlines()
                return [json.loads(line.strip()) for line in lines[-limit:]]
        except Exception as e:
            print(f"❌ Failed to read error log: {e}")
            return []
    
    def display_errors(errors):
        """Display errors in a formatted way."""
        if not errors:
            print("✅ No errors found")
            return
        
        for i, error in enumerate(errors, 1):
            # Parse timestamp
            timestamp = datetime.fromisoformat(error['timestamp'])
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format severity
            severity = error['severity']
            if severity == 'CRITICAL':
                icon = '🚨'
            elif severity == 'ERROR':
                icon = '❌'
            elif severity == 'WARNING':
                icon = '⚠️'
            else:
                icon = 'ℹ️'
            
            # Display error
            print(f"\n{i}. {icon} {time_str} - {error['error_name']} ({error['error_code']})")
            print(f"   Client: {error['client_id'] or 'N/A'}")
            print(f"   Message: {error['message']}")
            print(f"   Resolution: {error['resolution']}")
            
            # Show context if available
            if error['context']:
                print(f"   Context: {error['context']}")
            
            # Show exception if available
            if error['exception']:
                print(f"   Exception: {error['exception_type']}: {error['exception']}")
    
    if watch:
        # Watch mode - continuously monitor for new errors
        print("👁️  Watching for new errors (Ctrl+C to stop)...")
        last_count = 0
        
        try:
            while True:
                errors = read_errors()
                current_count = len(errors)
                
                if current_count > last_count:
                    # New errors detected
                    new_errors = errors[last_count:]
                    print(f"\n🔔 {len(new_errors)} new error(s) detected:")
                    display_errors(new_errors)
                    last_count = current_count
                
                time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            print("\n⏹️  Stopped watching")
    else:
        # One-time view
        errors = read_errors()
        display_errors(errors)

def main():
    parser = argparse.ArgumentParser(description='View recent brain errors')
    parser.add_argument('--limit', type=int, default=10, help='Number of recent errors to show')
    parser.add_argument('--watch', action='store_true', help='Watch for new errors')
    
    args = parser.parse_args()
    
    view_recent_errors(args.limit, args.watch)

if __name__ == "__main__":
    main()