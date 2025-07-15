"""
Persistence Data Models

Defines the data structures used for brain state persistence,
including checkpoints and configuration.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PersistenceConfig:
    """Configuration for brain persistence system."""
    
    # Storage paths
    memory_root_path: str = "./robot_memory"
    checkpoints_path: str = "checkpoints"
    deltas_path: str = "deltas" 
    metadata_path: str = "metadata"
    
    # Checkpoint frequency  
    checkpoint_interval_experiences: int = 1000
    checkpoint_interval_seconds: int = 5  # 5 seconds as requested, with throttled checking in brain.py
    
    # Delta batching
    delta_batch_size: int = 100
    
    # Compression and format
    use_compression: bool = True
    checkpoint_format: str = "json"  # "json" or "cbor"
    
    # Cleanup and retention
    max_checkpoints: int = 10
    max_deltas: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistenceConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BrainCheckpoint:
    """
    Complete brain state checkpoint.
    
    Contains all data needed to restore the brain to a specific point in time.
    Designed to be serializable and human-readable for analysis.
    """
    
    # Checkpoint metadata
    checkpoint_id: str
    timestamp: float
    experience_count: int
    session_id: str
    brain_version: str
    
    # Core brain data
    experiences: Dict[str, Dict[str, Any]]  # experience_id -> experience data
    similarity_state: Dict[str, Any]        # learnable similarity parameters
    activation_state: Dict[str, Any]        # activation dynamics state
    prediction_state: Dict[str, Any]        # prediction engine state
    stream_state: Dict[str, Any]           # stream buffer state
    
    # System statistics
    performance_stats: Dict[str, Any]
    learning_metrics: Dict[str, Any]
    
    # Configuration snapshot
    system_config: Dict[str, Any]
    
    def __post_init__(self):
        """Validate checkpoint after creation."""
        if self.experience_count != len(self.experiences):
            raise ValueError(f"Experience count mismatch: {self.experience_count} != {len(self.experiences)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            'metadata': {
                'checkpoint_id': self.checkpoint_id,
                'timestamp': self.timestamp,
                'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
                'experience_count': self.experience_count,
                'session_id': self.session_id,
                'brain_version': self.brain_version,
                'format_version': '1.0'
            },
            'brain_data': {
                'experiences': self.experiences,
                'similarity_state': self.similarity_state,
                'activation_state': self.activation_state,
                'prediction_state': self.prediction_state,
                'stream_state': self.stream_state
            },
            'statistics': {
                'performance_stats': self.performance_stats,
                'learning_metrics': self.learning_metrics
            },
            'config': self.system_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrainCheckpoint':
        """Create checkpoint from dictionary."""
        metadata = data['metadata']
        brain_data = data['brain_data']
        statistics = data['statistics']
        
        return cls(
            checkpoint_id=metadata['checkpoint_id'],
            timestamp=metadata['timestamp'],
            experience_count=metadata['experience_count'],
            session_id=metadata['session_id'],
            brain_version=metadata['brain_version'],
            experiences=brain_data['experiences'],
            similarity_state=brain_data['similarity_state'],
            activation_state=brain_data['activation_state'],
            prediction_state=brain_data['prediction_state'],
            stream_state=brain_data['stream_state'],
            performance_stats=statistics['performance_stats'],
            learning_metrics=statistics['learning_metrics'],
            system_config=data['config']
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get human-readable checkpoint information."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'created': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'experiences': self.experience_count,
            'session': self.session_id,
            'version': self.brain_version,
            'similarity_parameters': len(self.similarity_state.get('feature_weights', [])),
            'active_experiences': sum(1 for exp in self.experiences.values() 
                                    if exp.get('activation_level', 0) > 0),
            'total_predictions': self.performance_stats.get('total_predictions', 0),
            'prediction_accuracy': f"{self.performance_stats.get('accuracy', 0):.2%}"
        }


@dataclass 
class SessionInfo:
    """Information about a brain session."""
    
    session_id: str
    start_time: float
    end_time: Optional[float]
    initial_experience_count: int
    final_experience_count: Optional[int]
    checkpoints_created: List[str]
    brain_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'start_datetime': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': self.end_time,
            'end_datetime': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'initial_experience_count': self.initial_experience_count,
            'final_experience_count': self.final_experience_count,
            'experiences_added': (self.final_experience_count or 0) - self.initial_experience_count,
            'checkpoints_created': self.checkpoints_created,
            'brain_version': self.brain_version,
            'duration_minutes': (self.end_time - self.start_time) / 60 if self.end_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        """Create from dictionary."""
        return cls(
            session_id=data['session_id'],
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            initial_experience_count=data['initial_experience_count'],
            final_experience_count=data.get('final_experience_count'),
            checkpoints_created=data['checkpoints_created'],
            brain_version=data['brain_version']
        )