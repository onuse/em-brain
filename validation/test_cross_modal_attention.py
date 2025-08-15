#!/usr/bin/env python3
"""
Test Cross-Modal Object Attention System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
import cv2
import time
from src.attention.object_attention import CrossModalAttentionSystem, CrossModalObject, ObjectState
from src.attention.signal_attention import ModalityType

def test_cross_modal_attention():
    """Test cross-modal object attention system"""
    print("🎯 Testing Cross-Modal Object Attention System")
    print("=" * 55)
    
    # Create attention system with emergent parameters
    attention_system = CrossModalAttentionSystem(
        target_framerate=30.0,
        power_budget_watts=10.0
    )
    
    # Test 1: Single object across modalities
    print("\n🔗 Test 1: Single Object Cross-Modal Binding")
    
    # Create correlated visual and audio signals
    visual_signal = create_bouncing_ball_visual(64, 64, frame=0)
    audio_signal = create_bouncing_ball_audio(1000, frame=0)
    
    sensory_streams = {
        ModalityType.VISUAL: {
            'signal': visual_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.8
        },
        ModalityType.AUDIO: {
            'signal': audio_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.7
        }
    }
    
    # Process several frames to show binding
    for frame in range(8):
        # Update signals to show coherent motion
        sensory_streams[ModalityType.VISUAL]['signal'] = create_bouncing_ball_visual(64, 64, frame)
        sensory_streams[ModalityType.AUDIO]['signal'] = create_bouncing_ball_audio(1000, frame)
        
        attention_state = attention_system.update(sensory_streams)
        
        print(f"   Frame {frame}: Objects={attention_state['active_objects']}, "
              f"Attended={attention_state['attended_object'].object_id if attention_state['attended_object'] else 'None'}, "
              f"Bindings={attention_state['binding_events']}")
        
        time.sleep(0.1)
    
    # Test 2: Multiple competing objects
    print("\n🏆 Test 2: Multiple Objects Competition")
    attention_system.reset()
    
    # Create multiple objects
    for frame in range(15):
        visual_ball = create_bouncing_ball_visual(64, 64, frame)
        audio_ball = create_bouncing_ball_audio(1000, frame)
        
        # Add second object (car)
        visual_car = create_moving_car_visual(64, 64, frame)
        audio_car = create_moving_car_audio(1000, frame)
        
        # Combine signals (simplified - in reality would be spatial)
        combined_visual = np.maximum(visual_ball, visual_car)
        combined_audio = visual_ball + visual_car  # Simplified mixing
        
        sensory_streams = {
            ModalityType.VISUAL: {
                'signal': combined_visual,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.6
            },
            ModalityType.AUDIO: {
                'signal': combined_audio,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.5
            }
        }
        
        attention_state = attention_system.update(sensory_streams)
        
        attended_id = attention_state['attended_object'].object_id if attention_state['attended_object'] else 'None'
        
        print(f"   Frame {frame:2d}: Objects={attention_state['active_objects']}, "
              f"Attended={attended_id}, Switches={attention_state['attention_switches']}")
        
        time.sleep(0.05)
    
    # Test 3: Cross-modal binding with tactile
    print("\n🤚 Test 3: Visual-Audio-Tactile Binding")
    attention_system.reset()
    
    for frame in range(10):
        # Create correlated signals across 3 modalities
        visual_signal = create_button_press_visual(32, 32, frame)
        audio_signal = create_button_press_audio(500, frame)
        tactile_signal = create_button_press_tactile(50, frame)
        
        sensory_streams = {
            ModalityType.VISUAL: {
                'signal': visual_signal,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.9
            },
            ModalityType.AUDIO: {
                'signal': audio_signal,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.8
            },
            ModalityType.TACTILE: {
                'signal': tactile_signal,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.7
            }
        }
        
        attention_state = attention_system.update(sensory_streams)
        
        object_details = attention_state['object_details']
        
        print(f"   Frame {frame}: Objects={len(object_details)}, "
              f"Bindings={attention_state['binding_events']}")
        
        if object_details:
            for obj in object_details:
                modalities = [m.name for m in obj['modalities']]
                print(f"     Object {obj['id']}: modalities={modalities}, "
                      f"coherence={obj['coherence']:.3f}")
    
    # Test 4: Inhibition of return
    print("\n🔄 Test 4: Inhibition of Return")
    attention_system.reset()
    
    # Create two distinct objects
    inhibition_sequence = []
    
    for frame in range(20):
        # Object 1: Left side
        visual_left = create_object_at_position(64, 64, 16, 32, frame)
        audio_left = create_audio_at_frequency(1000, 440, frame)
        
        # Object 2: Right side  
        visual_right = create_object_at_position(64, 64, 48, 32, frame)
        audio_right = create_audio_at_frequency(1000, 880, frame)
        
        combined_visual = visual_left + visual_right
        combined_audio = audio_left + audio_right
        
        sensory_streams = {
            ModalityType.VISUAL: {
                'signal': combined_visual,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.5
            },
            ModalityType.AUDIO: {
                'signal': combined_audio,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.5
            }
        }
        
        attention_state = attention_system.update(sensory_streams)
        
        attended_id = attention_state['attended_object'].object_id if attention_state['attended_object'] else 'None'
        
        inhibition_sequence.append({
            'frame': frame,
            'attended_object': attended_id,
            'switches': attention_state['attention_switches']
        })
        
        print(f"   Frame {frame:2d}: Attended={attended_id}, "
              f"Switches={attention_state['attention_switches']}")
        
        time.sleep(0.1)
    
    # Test 5: Computational constraints
    print("\n💻 Test 5: Computational Constraints")
    attention_system.reset()
    
    # Stress test with many objects
    for frame in range(12):
        # Create many competing objects
        signals = []
        for i in range(5):  # 5 objects
            visual = create_object_at_position(64, 64, 10 + i*10, 20 + i*5, frame)
            audio = create_audio_at_frequency(1000, 200 + i*100, frame)
            signals.append((visual, audio))
        
        # Combine all signals
        combined_visual = np.sum([s[0] for s in signals], axis=0)
        combined_audio = np.sum([s[1] for s in signals], axis=0)
        
        sensory_streams = {
            ModalityType.VISUAL: {
                'signal': combined_visual,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.8
            },
            ModalityType.AUDIO: {
                'signal': combined_audio,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.7
            }
        }
        
        attention_state = attention_system.update(sensory_streams)
        
        stats = attention_system.get_statistics()
        compute_util = stats.get('compute_utilization', 0)
        
        print(f"   Frame {frame:2d}: Objects={attention_state['active_objects']}, "
              f"Compute={stats['remaining_compute']}, "
              f"Util={compute_util:.2f}")
    
    # Test 6: Attention mask generation
    print("\n🎭 Test 6: Attention Mask Generation")
    attention_system.reset()
    
    # Create focused object
    visual_signal = create_focused_object(64, 64)
    audio_signal = create_focused_audio(1000)
    
    sensory_streams = {
        ModalityType.VISUAL: {
            'signal': visual_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.9
        },
        ModalityType.AUDIO: {
            'signal': audio_signal,
            'brain_output': np.random.uniform(0, 1, 16),
            'novelty': 0.8
        }
    }
    
    attention_state = attention_system.update(sensory_streams)
    attention_mask = attention_state['attention_mask']
    
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Attended pixels: {np.sum(attention_mask > 0)}")
    print(f"   Max attention: {np.max(attention_mask):.3f}")
    print(f"   Attention center: {np.unravel_index(np.argmax(attention_mask), attention_mask.shape)}")
    
    # Test 7: Performance test
    print("\n⚡ Test 7: Performance Test")
    attention_system.reset()
    
    performance_times = []
    
    for i in range(50):
        start_time = time.time()
        
        # Create test signals
        visual = create_bouncing_ball_visual(64, 64, i)
        audio = create_bouncing_ball_audio(1000, i)
        
        sensory_streams = {
            ModalityType.VISUAL: {
                'signal': visual,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.5
            },
            ModalityType.AUDIO: {
                'signal': audio,
                'brain_output': np.random.uniform(0, 1, 16),
                'novelty': 0.5
            }
        }
        
        attention_state = attention_system.update(sensory_streams)
        
        processing_time = (time.time() - start_time) * 1000
        performance_times.append(processing_time)
    
    avg_time = np.mean(performance_times)
    max_time = np.max(performance_times)
    
    print(f"   Average processing time: {avg_time:.2f}ms")
    print(f"   Max processing time: {max_time:.2f}ms")
    print(f"   Real-time capable: {'✅ Yes' if avg_time < 20 else '⚠️ Marginal' if avg_time < 50 else '❌ No'}")
    
    # Final statistics
    final_stats = attention_system.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total objects created: {final_stats['total_objects_created']}")
    print(f"   Binding events: {final_stats['binding_events']}")
    print(f"   Attention switches: {final_stats['attention_switches']}")
    print(f"   Compute utilization: {final_stats['compute_utilization']:.2f}")
    
    print(f"\n🎉 Cross-Modal Object Attention System Test Complete!")
    
    # Assessment
    success_criteria = [
        final_stats['total_objects_created'] > 0,    # Objects created
        final_stats['binding_events'] > 0,          # Cross-modal binding
        final_stats['attention_switches'] > 2,      # Attention switching
        avg_time < 50,                              # Performance acceptable
        np.max(attention_mask) > 0.5,              # Attention mask working
        final_stats['compute_utilization'] > 0.1   # Compute constraints active
    ]
    
    passed = sum(success_criteria)
    total = len(success_criteria)
    
    print(f"   ✅ Tests passed: {passed}/{total}")
    print(f"   🧠 Constraint-based emergence: {'✅ Strong' if passed >= 5 else '⚠️ Moderate' if passed >= 3 else '❌ Weak'}")
    
    return passed >= 4

# Helper functions to create test signals

def create_bouncing_ball_visual(width, height, frame):
    """Create visual signal of bouncing ball"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Ball position based on frame
    x = int(10 + (frame * 3) % (width - 20))
    y = int(10 + (frame * 2) % (height - 20))
    
    # Draw ball
    cv2.circle(signal, (x, y), 5, 0.9, -1)
    
    return signal

def create_bouncing_ball_audio(length, frame):
    """Create audio signal of bouncing ball"""
    signal = np.zeros(length, dtype=np.float32)
    
    # Ball bounce sound at regular intervals
    bounce_interval = 50
    if frame % bounce_interval < 10:
        # Bounce sound
        t = np.linspace(0, 0.1, length)
        signal = 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 20)
    
    return signal

def create_moving_car_visual(width, height, frame):
    """Create visual signal of moving car"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Car position
    x = int(5 + (frame * 2) % (width - 15))
    y = int(height - 20)
    
    # Draw car (rectangle)
    cv2.rectangle(signal, (x, y), (x+10, y+8), 0.7, -1)
    
    return signal

def create_moving_car_audio(length, frame):
    """Create audio signal of moving car"""
    t = np.linspace(0, 1, length)
    
    # Car engine sound (low frequency)
    signal = 0.3 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 160 * t)
    
    return signal

def create_button_press_visual(width, height, frame):
    """Create visual signal of button press"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Button press animation
    if frame < 5:
        # Button being pressed
        cv2.rectangle(signal, (10, 10), (22, 22), 0.8, -1)
    else:
        # Button released
        cv2.rectangle(signal, (10, 10), (22, 22), 0.4, -1)
    
    return signal

def create_button_press_audio(length, frame):
    """Create audio signal of button press"""
    signal = np.zeros(length, dtype=np.float32)
    
    if frame == 0:
        # Click sound
        t = np.linspace(0, 0.05, length)
        signal = 0.8 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 50)
    
    return signal

def create_button_press_tactile(length, frame):
    """Create tactile signal of button press"""
    signal = np.zeros(length, dtype=np.float32)
    
    if frame < 5:
        # Pressure during press
        signal[20:30] = 0.9
    
    return signal

def create_object_at_position(width, height, x, y, frame):
    """Create visual object at specific position"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Object at position
    cv2.circle(signal, (x, y), 4, 0.8, -1)
    
    return signal

def create_audio_at_frequency(length, frequency, frame):
    """Create audio signal at specific frequency"""
    t = np.linspace(0, 0.5, length)
    signal = 0.4 * np.sin(2 * np.pi * frequency * t)
    
    return signal

def create_focused_object(width, height):
    """Create single focused object"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Central object
    cv2.circle(signal, (width//2, height//2), 8, 0.9, -1)
    
    return signal

def create_focused_audio(length):
    """Create focused audio signal"""
    t = np.linspace(0, 1, length)
    signal = 0.6 * np.sin(2 * np.pi * 440 * t)
    
    return signal

if __name__ == "__main__":
    success = test_cross_modal_attention()
    if success:
        print("\n✅ Cross-Modal Object Attention System ready for deployment!")
    else:
        print("\n⚠️ System needs further optimization")