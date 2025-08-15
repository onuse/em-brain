#!/usr/bin/env python3
"""Quick test of initialization time"""

import sys
import os
from pathlib import Path
import time

brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory

print("Testing initialization...")

start = time.time()

# Just the basics
robot_registry = RobotRegistry()
print(f"RobotRegistry: {time.time() - start:.2f}s")

brain_config = {'quiet_mode': True, 'spatial_resolution': 4}
brain_factory = DynamicBrainFactory(brain_config)
print(f"BrainFactory: {time.time() - start:.2f}s")

brain_pool = BrainPool(brain_factory)
print(f"BrainPool: {time.time() - start:.2f}s")

adapter_factory = AdapterFactory()
print(f"AdapterFactory: {time.time() - start:.2f}s")

brain_service = BrainService(brain_pool, adapter_factory)
print(f"BrainService: {time.time() - start:.2f}s")

connection_handler = ConnectionHandler(robot_registry, brain_service)
print(f"ConnectionHandler: {time.time() - start:.2f}s")

# Try handshake
client_id = "test"
capabilities = [1.0, 16.0, 4.0, 0.0, 0.0]
print("\nPerforming handshake...")
handshake_start = time.time()
response = connection_handler.handle_handshake(client_id, capabilities)
print(f"Handshake: {time.time() - handshake_start:.2f}s")
print(f"Response: {response}")

print(f"\nTotal time: {time.time() - start:.2f}s")