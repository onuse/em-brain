# Dynamic Brain Architecture V2: Proper Separation of Concerns

## Architecture Layers

### 1. Network Layer (Transport Only)
```python
class TCPServer:
    """ONLY handles TCP communication - no business logic"""
    def __init__(self, connection_handler: ConnectionHandler):
        self.connection_handler = connection_handler
    
    def _handle_message(self, client_id, message):
        # Just decode and forward - no logic here
        msg_type, data = self.protocol.decode(message)
        response = self.connection_handler.handle_message(client_id, msg_type, data)
        return self.protocol.encode(response)
```

### 2. Connection Management Layer
```python
class ConnectionHandler:
    """Manages client connections and routes to appropriate services"""
    def __init__(self, robot_registry: RobotRegistry, brain_service: BrainService):
        self.robot_registry = robot_registry
        self.brain_service = brain_service
        self.active_sessions = {}  # client_id -> session
    
    def handle_handshake(self, client_id, capabilities):
        # Create robot profile
        robot = self.robot_registry.register_robot(capabilities)
        
        # Request brain session
        session = self.brain_service.create_session(robot)
        self.active_sessions[client_id] = session
        
        return session.get_handshake_response()
    
    def handle_sensory_input(self, client_id, sensory_data):
        session = self.active_sessions[client_id]
        return session.process_sensory_input(sensory_data)
```

### 3. Robot Registry Layer
```python
@dataclass
class Robot:
    """Pure data class - no logic"""
    robot_id: str
    robot_type: str
    sensory_channels: List[SensorChannel]
    motor_channels: List[MotorChannel]
    capabilities: Dict[str, Any]

class RobotRegistry:
    """Manages robot profiles and types"""
    def register_robot(self, capabilities: List[float]) -> Robot:
        # Parse capabilities into Robot profile
        # This is the ONLY place that understands capability encoding
        return Robot(...)
```

### 4. Brain Service Layer
```python
class BrainService:
    """Manages brain lifecycle and sessions"""
    def __init__(self, brain_pool: BrainPool, adapter_factory: AdapterFactory):
        self.brain_pool = brain_pool
        self.adapter_factory = adapter_factory
    
    def create_session(self, robot: Robot) -> BrainSession:
        # Get or create brain for this robot's profile
        brain = self.brain_pool.get_brain_for_profile(robot.get_profile_key())
        
        # Create adapters for this specific robot
        sensory_adapter = self.adapter_factory.create_sensory_adapter(robot)
        motor_adapter = self.adapter_factory.create_motor_adapter(robot)
        
        return BrainSession(brain, sensory_adapter, motor_adapter)

class BrainSession:
    """Handles one robot's interaction with a brain"""
    def __init__(self, brain, sensory_adapter, motor_adapter):
        self.brain = brain
        self.sensory_adapter = sensory_adapter
        self.motor_adapter = motor_adapter
    
    def process_sensory_input(self, raw_sensory: List[float]) -> List[float]:
        # Adapt sensory to field space
        field_input = self.sensory_adapter.to_field_space(raw_sensory)
        
        # Process through brain (brain knows nothing about robots!)
        field_output = self.brain.process_field_dynamics(field_input)
        
        # Adapt field to motor space
        motor_commands = self.motor_adapter.from_field_space(field_output)
        return motor_commands
```

### 5. Brain Pool Layer
```python
class BrainPool:
    """Manages brain instances - separation from creation"""
    def __init__(self, brain_factory: BrainFactory):
        self.brain_factory = brain_factory
        self.brains = {}  # profile_key -> brain
    
    def get_brain_for_profile(self, profile_key: str):
        if profile_key not in self.brains:
            # Determine optimal brain parameters
            field_dims = self._calculate_field_dimensions(profile_key)
            self.brains[profile_key] = self.brain_factory.create(field_dims)
        return self.brains[profile_key]
```

### 6. Pure Brain Layer
```python
class UnifiedFieldBrain:
    """ONLY knows about field dynamics - nothing about robots"""
    def __init__(self, field_dimensions: int, spatial_resolution: int):
        self.field_dimensions = field_dimensions
        self.unified_field = torch.zeros(
            (field_dimensions, spatial_resolution, spatial_resolution, spatial_resolution)
        )
    
    def process_field_dynamics(self, field_input: torch.Tensor) -> torch.Tensor:
        # Pure field evolution - no robot knowledge needed
        # Input/output are abstract field coordinates
        pass
```

### 7. Adapter Layer
```python
class SensoryAdapter:
    """Translates robot-specific sensors to abstract field space"""
    def __init__(self, robot: Robot, field_dimensions: int):
        self.robot = robot
        self.field_dimensions = field_dimensions
        self.projection = self._create_projection()
    
    def to_field_space(self, sensory: List[float]) -> torch.Tensor:
        # Robot-specific to universal field mapping
        pass

class MotorAdapter:
    """Translates abstract field space to robot-specific motors"""
    def __init__(self, robot: Robot, field_dimensions: int):
        self.robot = robot
        self.field_dimensions = field_dimensions
        self.extraction = self._create_extraction()
    
    def from_field_space(self, field_state: torch.Tensor) -> List[float]:
        # Universal field to robot-specific mapping
        pass
```

## Benefits of This Architecture

1. **Clear Separation**:
   - Network layer only handles transport
   - Robot management is separate from brain management
   - Brain knows nothing about robots
   - Adapters handle all robot-specific translation

2. **Single Responsibility**:
   - TCPServer: Network protocol only
   - ConnectionHandler: Session management
   - RobotRegistry: Robot profile management
   - BrainPool: Brain instance management
   - BrainFactory: Brain creation only
   - UnifiedFieldBrain: Pure field dynamics
   - Adapters: Translation only

3. **Testability**:
   - Each layer can be tested independently
   - Mock boundaries are clear
   - No hidden dependencies

4. **Extensibility**:
   - New robot types just need new adapters
   - Brain improvements don't affect robot interface
   - Network protocol changes don't affect brain logic

## Dependency Flow

```
TCPServer
    ↓
ConnectionHandler
    ↓        ↓
RobotRegistry  BrainService
                ↓        ↓
            BrainPool  AdapterFactory
                ↓
            BrainFactory
                ↓
            UnifiedFieldBrain
```

Each arrow represents a clean interface with no reverse dependencies.

## Implementation Priority

1. **Phase 1**: Create clean interfaces between layers
2. **Phase 2**: Implement RobotRegistry and Robot profile system
3. **Phase 3**: Refactor BrainFactory to only create, add BrainPool
4. **Phase 4**: Implement adapter system
5. **Phase 5**: Clean up TCPServer to only handle transport
6. **Phase 6**: Implement ConnectionHandler as orchestrator

This architecture provides much better separation of concerns while still achieving the goal of dynamic brain adaptation.