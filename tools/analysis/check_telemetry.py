#!/usr/bin/env python3
"""
Quick script to check what telemetry data is actually available
"""

import sys
import os
from pathlib import Path

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication.monitoring_client import create_monitoring_client
import json
import time


def main():
    print("🔍 Checking telemetry data...")
    
    # Connect to monitoring server
    client = create_monitoring_client()
    if not client:
        print("❌ Failed to connect to monitoring server")
        return
    
    print("✅ Connected to monitoring server")
    
    # Wait a moment for any brain sessions to register
    time.sleep(1)
    
    # Get all telemetry
    print("\n📊 Requesting all telemetry...")
    
    # Use the client's request_data method which returns full response
    response = client.request_data("telemetry")
    
    if response and isinstance(response, dict):
        print(f"   📥 Raw response: {response}")
        if response.get('status') == 'success':
            data = response.get('data', {})
        else:
            error_msg = response.get('message', response.get('error', 'Unknown error'))
            print(f"❌ Server error: {error_msg}")
            print(f"   Full response: {response}")
            client.disconnect()
            return
    else:
        print(f"Failed to get telemetry - no response")
        client.disconnect()
        return
    
    print(f"Response type: {type(data)}")
    print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    
    if isinstance(data, dict):
        # Check if this is a dict of sessions
        session_keys = [k for k in data.keys() if k.startswith('session_')]
        if session_keys:
            # Multiple sessions
            for session_id in session_keys:
                telemetry = data[session_id]
                print(f"\n🧠 Session: {session_id}")
                if isinstance(telemetry, dict):
                    # Show key metrics
                    print(f"  Prediction confidence: {telemetry.get('prediction_confidence', 'N/A')}")
                    print(f"  Field energy: {telemetry.get('field_energy', 'N/A')}")
                    
                    # Evolution state
                    evo_state = telemetry.get('evolution_state', {})
                    if evo_state:
                        print(f"  Evolution state:")
                        print(f"    Self-modification: {evo_state.get('self_modification_strength', 'N/A')}")
                        print(f"    Evolution cycles: {evo_state.get('evolution_cycles', 'N/A')}")
                        print(f"    Working memory: {evo_state.get('working_memory', 'N/A')}")
                    else:
                        print(f"  Evolution state: Not found")
                    
                    # Show all keys
                    print(f"  All telemetry keys: {list(telemetry.keys())}")
                else:
                    print(f"  Telemetry type: {type(telemetry)}")
        else:
            # Single session or empty
            print("No active sessions found in telemetry")
    else:
        print(f"Unexpected data format: {data}")
    
    # Try session_info
    print("\n📋 Requesting session info...")
    
    # Use the client's request_data method
    sessions_response = client.request_data("session_info")
    
    if sessions_response and isinstance(sessions_response, dict):
        if sessions_response.get('status') == 'success':
            sessions = sessions_response.get('data', [])
            print(f"Active sessions: {len(sessions) if isinstance(sessions, list) else 'Unknown'}")
            if isinstance(sessions, list):
                for session in sessions:
                    if isinstance(session, dict):
                        print(f"  - {session.get('session_id')} ({session.get('robot_type')}) - {session.get('cycles')} cycles")
                    else:
                        print(f"  - Session data type: {type(session)}")
            else:
                print(f"Sessions data type: {type(sessions)}")
        else:
            print(f"❌ Server error: {sessions_response.get('message', sessions_response.get('error', 'Unknown error'))}")
    else:
        print("Failed to get session info")
    
    client.disconnect()


if __name__ == "__main__":
    main()