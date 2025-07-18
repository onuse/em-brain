#!/usr/bin/env python3
"""
Debug camera capture to isolate visualization issue
"""

import cv2
import numpy as np
import time

def test_basic_camera():
    """Test basic camera capture without any brain processing"""
    print("🎥 Testing basic camera capture...")
    
    # Try to initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Cannot open camera")
        return False
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    print("✅ Camera initialized")
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add simple overlay to verify it's working
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Basic Camera Test", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Debug Camera Test', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"validation/debug_camera_frame_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
    
    camera.release()
    cv2.destroyAllWindows()
    print(f"📊 Captured {frame_count} frames successfully")
    return True

def test_overlay_effects():
    """Test overlay rendering that might be causing issues"""
    print("\n🎨 Testing overlay effects...")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    # Test different overlay techniques
    overlay1 = test_frame.copy()
    
    # Test 1: Simple color overlay
    overlay_color = np.zeros_like(overlay1)
    overlay_color[:, :] = (255, 255, 0)  # Yellow
    
    # This might be the issue - if alpha is too high, it masks the original
    alpha = 0.8  # Very high alpha
    cv2.addWeighted(overlay1, 1 - alpha, overlay_color, alpha, 0, overlay1)
    
    cv2.imshow('High Alpha Overlay (Problematic)', overlay1)
    cv2.waitKey(2000)  # Show for 2 seconds
    
    # Test 2: Lower alpha
    overlay2 = test_frame.copy()
    alpha = 0.2  # Much lower alpha
    cv2.addWeighted(overlay2, 1 - alpha, overlay_color, alpha, 0, overlay2)
    
    cv2.imshow('Low Alpha Overlay (Better)', overlay2)
    cv2.waitKey(2000)  # Show for 2 seconds
    
    cv2.destroyAllWindows()
    print("✅ Overlay test complete")

if __name__ == "__main__":
    print("🔍 Camera Debug Test")
    print("=" * 40)
    
    # Test basic camera first
    if test_basic_camera():
        test_overlay_effects()
    else:
        print("❌ Basic camera test failed")