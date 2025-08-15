# Robot Brain Demonstrations

This folder contains demonstrations of the minimal brain architecture controlling a PiCar-X robot simulation.

## Available Demos

### 1. **test_demo.py** - Text-Based Simulation
- Minimal dependencies (only numpy)
- ASCII visualization of robot navigation
- Quick test of brain functionality
- Run: `python3 demo_runner.py test_demo`

### 2. **demo_2d.py** - 2D Grid Visualization  
- Interactive pygame visualization
- Top-down view with colored states
- Trail tracking and collision detection
- Run: `python3 demo_runner.py demo_2d`

### 3. **demo_3d.py** - 3D Scientific Visualization
- Matplotlib 3D rendering
- Physics simulation with brain monitoring
- Performance analytics and learning curves  
- Run: `python3 demo_runner.py demo_3d`

## Other Demos

### **spatial_learning_demo.py** - Basic Spatial Learning
- Simple 10x10 grid world
- Demonstrates emergent navigation
- Not specific to PiCar-X robot
- Run: `python3 demo_runner.py spatial_learning`

## Running Demos

All demos should be run from the brain root directory using the demo runner:

```bash
cd /path/to/brain
python3 demo_runner.py <demo_name>
```

For example:
- `python3 demo_runner.py test_demo` - Run text-based demo
- `python3 demo_runner.py demo_2d` - Run 2D visualization
- `python3 demo_runner.py demo_3d` - Run 3D visualization

## Demo Features

All PiCar-X demos demonstrate:
- Emergent obstacle avoidance
- Spatial navigation and learning
- Real-time brain activity monitoring
- No hardcoded behaviors - everything emerges from the 4-system brain

## Dependencies

- **test_demo**: numpy only
- **demo_2d**: numpy, pygame
- **demo_3d**: numpy, matplotlib
- **spatial_learning_demo**: numpy only

## PiCar-X Simulation Details

The `picar_x_simulation/` subfolder contains:
- **Vehicle model**: Realistic PiCar-X physics
- **Brainstem**: Interface between brain and robot sensors/motors
- **Environment**: 20x20m world with obstacles
- **Visualization**: Various rendering approaches

All demos use the same underlying robot simulation with different visualization methods.