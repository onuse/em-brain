# 3D Wireframe Visualization Plan

## Overview

Upgrade the PiCar-X simulation to include a 3D wireframe visualization with a freely movable camera viewport. This will provide better spatial understanding of the robot's behavior while maintaining simplicity and performance.

## Technology Stack

### Primary: Pygame + PyOpenGL
- **Why**: Integrates with existing pygame setup, lightweight, perfect for wireframe
- **Benefits**: Simple to implement, good performance, extensive documentation
- **Requirements**: `pip install PyOpenGL PyOpenGL_accelerate`

### Alternative Options Considered
- VPython: Too simplified, limited camera control
- Panda3D: Overkill for wireframe visualization
- Vispy: Great but requires learning new framework
- Pure OpenGL: Too low-level, more complex

## Visual Design

### Wireframe Aesthetic
```
┌─────────────────────────────────┐
│  ╱────────────────────────╲     │  3D Wireframe View
│ ╱                          ╲    │  - Clean lines
│├──────────────────────────┤│    │  - Technical look
││     ┌─┐                  ││    │  - Easy to render
││     └─┘  ←── Vehicle     ││    │  - Clear visibility
│└──────────────────────────┘│    │
│ ╲                          ╱    │
│  ╲────────────────────────╱     │
└─────────────────────────────────┘
```

### Color Scheme
- **Grid/Floor**: Dark gray (0.3, 0.3, 0.3)
- **Walls**: Light gray (0.7, 0.7, 0.7)
- **Vehicle**: Cyan (0.0, 0.8, 1.0)
- **Obstacles**: Orange (1.0, 0.6, 0.0)
- **Sensor rays**: Red (1.0, 0.2, 0.2)
- **Trail**: Green (0.0, 1.0, 0.5)

## Implementation Architecture

### 1. Renderer3D Class Structure
```python
class Renderer3D:
    def __init__(self):
        self.camera = Camera3D()
        self.wireframe_renderer = WireframeRenderer()
        self.overlay_renderer = OverlayRenderer()
    
    def render_frame(self):
        # Clear and setup 3D view
        self.setup_3d_viewport()
        
        # Render 3D elements
        self.render_grid()
        self.render_room()
        self.render_vehicle()
        self.render_sensors()
        
        # Render 2D overlay
        self.setup_2d_viewport()
        self.render_data_overlay()
```

### 2. Camera System
```python
class Camera3D:
    def __init__(self):
        self.position = [5, 5, 3]    # Start above and away
        self.target = [2, 1.5, 0]    # Look at room center
        self.up = [0, 0, 1]          # Z-up
        
        # Camera controls
        self.yaw = -45               # Horizontal rotation
        self.pitch = -30             # Vertical rotation
        self.distance = 5            # Distance from target
        
        # Movement speeds
        self.move_speed = 2.0        # m/s
        self.rotate_speed = 90.0     # deg/s
        self.zoom_speed = 2.0        # m/s
```

## Features to Implement

### Phase 1: Core 3D Rendering
1. **Basic OpenGL setup** with pygame
2. **Wireframe rendering functions** (lines, boxes, circles)
3. **Room wireframe** (floor grid + wall outlines)
4. **Vehicle wireframe box** with orientation
5. **Simple camera controls** (rotate around target)

### Phase 2: Enhanced Visualization
1. **Obstacle wireframes** (boxes, cylinders)
2. **Sensor visualization**:
   - Ultrasonic cone (red wireframe cone)
   - Camera frustum (yellow wireframe pyramid)
   - IMU axes (small coordinate frame)
3. **Vehicle trail** (fading line in 3D)
4. **Grid with distance markers**

### Phase 3: Camera & UI
1. **Advanced camera modes**:
   - Free camera (WASD + mouse)
   - Follow camera (behind vehicle)
   - Top-down view (2D-like)
   - Side view (profile)
2. **2D overlay panels**:
   - Vehicle status (top-left)
   - Sensor readings (top-right)
   - Brain stats (bottom-left)
   - Controls help (bottom-right)
3. **Viewport switching** (hotkeys 1-4)

## Camera Controls

### Free Camera Mode
- **WASD**: Move camera (forward/back/left/right)
- **QE**: Move up/down
- **Mouse drag**: Rotate view
- **Scroll wheel**: Zoom in/out
- **Shift**: Move faster
- **Space**: Reset camera

### Follow Camera Mode
- **Mouse drag**: Orbit around vehicle
- **Scroll**: Adjust follow distance
- **Arrow keys**: Adjust follow angle

### Fixed Views
- **1**: Free camera
- **2**: Follow camera
- **3**: Top-down view
- **4**: Side view

## Wireframe Primitives

### Basic Shapes
```python
def draw_wireframe_box(x, y, z, width, height, depth):
    """Draw 3D wireframe box"""
    
def draw_wireframe_cylinder(x, y, z, radius, height, segments=8):
    """Draw 3D wireframe cylinder"""
    
def draw_wireframe_cone(x, y, z, radius, height, segments=8):
    """Draw 3D wireframe cone (for sensors)"""
    
def draw_wireframe_grid(size, spacing):
    """Draw floor grid with spacing"""
```

### Vehicle Components
```python
def draw_vehicle_wireframe(vehicle):
    # Main body
    draw_wireframe_box(...)
    
    # Wheels (4 small cylinders)
    for wheel in vehicle.wheels:
        draw_wireframe_cylinder(...)
    
    # Camera mount
    draw_wireframe_box(...)  # Small box on top
    
    # Direction indicator
    draw_line(...)  # Arrow showing heading
```

## 2D Overlay Information

### Top-Left: Vehicle Status
```
Vehicle Status
─────────────
Position: (2.34, 1.78)
Heading: 142.5°
Speed: 0.23 m/s
Motors: L=0.4 R=0.5
Steering: -5.2°
```

### Top-Right: Sensors
```
Sensor Readings
──────────────
Ultrasonic: 0.84m
Camera: Pan=15° Tilt=-5°
IMU: 
  Accel: [0.1, -0.05, 9.81]
  Gyro: [0, 0, 0.15]
  Heading: 142.5°
```

### Bottom-Left: Brain Stats
```
Brain Connection
───────────────
Status: Connected
Experiences: 1,234
Predictions: 567
Working Memory: 12
Confidence: 0.78
```

### Bottom-Right: Controls
```
Camera Controls
──────────────
WASD+Mouse: Move camera
1-4: Switch views
R: Reset vehicle
G: Toggle grid
T: Toggle trail
O: Toggle overlay
```

## Performance Considerations

### Optimization Strategies
1. **Use vertex arrays** for repeated geometry
2. **Frustum culling** for off-screen objects
3. **LOD system** for distant objects
4. **Batch similar wireframes** together
5. **Limit trail points** (max 200)

### Target Performance
- **60 FPS** with full visualization
- **< 5ms** render time per frame
- **< 50MB** GPU memory usage

## Implementation Order

1. **Basic 3D setup** (pygame + OpenGL initialization)
2. **Simple wireframe primitives** (lines, boxes)
3. **Room and grid rendering**
4. **Vehicle rendering with orientation**
5. **Basic camera controls** (orbit only)
6. **2D overlay system**
7. **Sensor visualizations**
8. **Advanced camera modes**
9. **Polish and optimization**

## Testing Plan

1. **Render accuracy**: Verify 3D positions match 2D
2. **Camera controls**: Test all movement modes
3. **Performance**: Monitor FPS with full scene
4. **Overlay readability**: Ensure text is clear
5. **Integration**: Verify brain connection works

This 3D visualization will provide an immersive view of the robot's behavior while maintaining the technical simplicity of wireframe graphics!