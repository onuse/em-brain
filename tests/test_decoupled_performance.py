#!/usr/bin/env python3
"""
Test the decoupled brain performance to verify it's no longer limited by GUI.
"""

import time
import sys
import threading
sys.path.append('.')

from core.async_brain_server import AsyncBrainServer
from simulation.brainstem_sim import GridWorldBrainstem
import queue


def test_brain_performance_headless():
    """Test brain performance without any GUI."""
    print("🧠 Testing Brain Performance (Headless)")
    print("=" * 50)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    
    # Start brain server
    brain_server.start()
    
    # Let it run for 10 seconds
    print("Running brain for 10 seconds...")
    time.sleep(10)
    
    # Get performance stats
    stats = brain_server.get_performance_stats()
    
    print(f"🚀 Brain Performance Results:")
    print(f"   Brain FPS: {stats['brain_fps']:.1f}")
    print(f"   Total steps: {stats['step_count']}")
    print(f"   Average frame time: {stats['average_frame_time']*1000:.2f}ms")
    print(f"   Min frame time: {stats['min_frame_time']*1000:.2f}ms")
    print(f"   Max frame time: {stats['max_frame_time']*1000:.2f}ms")
    
    # Stop brain server
    brain_server.stop()
    
    return stats['brain_fps']


def test_brain_performance_with_observer():
    """Test brain performance with GUI observer attached."""
    print("\n👁️  Testing Brain Performance (With Observer)")
    print("=" * 50)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    
    # Start brain server
    brain_server.start()
    
    # Add observer
    observer_queue = brain_server.register_observer("test_observer", update_frequency=30.0)
    
    # Create observer thread that processes updates
    observer_running = True
    observer_updates = 0
    
    def observer_thread():
        nonlocal observer_updates, observer_running
        while observer_running:
            try:
                state = observer_queue.get(timeout=0.1)
                observer_updates += 1
            except queue.Empty:
                pass
    
    # Start observer thread
    obs_thread = threading.Thread(target=observer_thread, daemon=True)
    obs_thread.start()
    
    # Let it run for 10 seconds
    print("Running brain with observer for 10 seconds...")
    time.sleep(10)
    
    # Stop observer
    observer_running = False
    brain_server.unregister_observer("test_observer")
    
    # Get performance stats
    stats = brain_server.get_performance_stats()
    
    print(f"🚀 Brain Performance Results (With Observer):")
    print(f"   Brain FPS: {stats['brain_fps']:.1f}")
    print(f"   Total steps: {stats['step_count']}")
    print(f"   Observer updates: {observer_updates}")
    print(f"   Observer FPS: {observer_updates / 10:.1f}")
    
    # Stop brain server
    brain_server.stop()
    
    return stats['brain_fps'], observer_updates / 10


def test_multiple_observers():
    """Test brain performance with multiple observers."""
    print("\n👥 Testing Brain Performance (Multiple Observers)")
    print("=" * 50)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    
    # Start brain server
    brain_server.start()
    
    # Add multiple observers
    observers = {}
    observer_threads = {}
    observer_counts = {}
    
    for i in range(5):
        observer_id = f"observer_{i}"
        obs_queue = brain_server.register_observer(observer_id, update_frequency=10.0 + i * 5)
        observers[observer_id] = obs_queue
        observer_counts[observer_id] = 0
        
        # Create observer thread
        def make_observer_thread(obs_id, obs_queue):
            def observer_thread():
                while observer_id in observers:
                    try:
                        state = obs_queue.get(timeout=0.1)
                        observer_counts[obs_id] += 1
                    except queue.Empty:
                        pass
            return observer_thread
        
        thread = threading.Thread(target=make_observer_thread(observer_id, obs_queue), daemon=True)
        thread.start()
        observer_threads[observer_id] = thread
    
    # Let it run for 10 seconds
    print("Running brain with 5 observers for 10 seconds...")
    time.sleep(10)
    
    # Stop observers
    for observer_id in list(observers.keys()):
        brain_server.unregister_observer(observer_id)
    observers.clear()
    
    # Get performance stats
    stats = brain_server.get_performance_stats()
    
    print(f"🚀 Brain Performance Results (5 Observers):")
    print(f"   Brain FPS: {stats['brain_fps']:.1f}")
    print(f"   Total steps: {stats['step_count']}")
    
    for observer_id, count in observer_counts.items():
        print(f"   {observer_id}: {count} updates ({count/10:.1f} FPS)")
    
    # Stop brain server
    brain_server.stop()
    
    return stats['brain_fps']


def main():
    """Run all decoupling tests."""
    print("🔬 DECOUPLED BRAIN PERFORMANCE TESTS")
    print("=" * 60)
    
    # Test 1: Headless performance
    headless_fps = test_brain_performance_headless()
    
    # Test 2: Performance with observer
    observer_fps, observer_rate = test_brain_performance_with_observer()
    
    # Test 3: Performance with multiple observers
    multi_observer_fps = test_multiple_observers()
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Brain FPS (headless):        {headless_fps:.1f}")
    print(f"Brain FPS (with observer):   {observer_fps:.1f}")
    print(f"Brain FPS (5 observers):     {multi_observer_fps:.1f}")
    print(f"Observer update rate:        {observer_rate:.1f} FPS")
    
    # Check if decoupling is working
    fps_degradation = (headless_fps - observer_fps) / headless_fps * 100
    
    print(f"\n🎯 DECOUPLING EFFECTIVENESS:")
    if fps_degradation < 5:
        print(f"✅ Excellent: Only {fps_degradation:.1f}% FPS loss with observers")
    elif fps_degradation < 15:
        print(f"✅ Good: {fps_degradation:.1f}% FPS loss with observers")
    else:
        print(f"⚠️  Poor: {fps_degradation:.1f}% FPS loss with observers")
    
    # Check if brain is running at reasonable speed
    if headless_fps > 500:
        print("🚀 Brain is running at excellent speed (>500 FPS)")
    elif headless_fps > 100:
        print("✅ Brain is running at good speed (>100 FPS)")
    else:
        print("⚠️  Brain speed needs improvement")
    
    # Final verdict
    if fps_degradation < 15 and headless_fps > 100:
        print("\n🎉 DECOUPLING SUCCESS: Brain runs independently of observers!")
    else:
        print("\n❌ DECOUPLING NEEDS WORK: Brain still affected by observers")


if __name__ == "__main__":
    main()