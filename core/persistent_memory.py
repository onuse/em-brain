"""
Persistent Memory System - Enables lifelong learning and experience accumulation.
Stores the robot's experiences and knowledge across sessions using efficient compression.
"""

import os
import json
import pickle
import gzip
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from pathlib import Path

from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.adaptive_tuning import AdaptiveParameterTuner


@dataclass
class MemorySession:
    """Represents a single session of robot operation."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    experiences_count: int
    final_graph_stats: Dict[str, Any]
    adaptive_tuning_stats: Dict[str, Any]
    session_summary: str


@dataclass
class MemoryArchive:
    """Archive of experiences organized by characteristics for efficient retrieval."""
    high_importance_experiences: List[str]  # Node IDs of critical experiences
    spatial_memory_nodes: List[str]         # Experiences with strong spatial components
    skill_learning_nodes: List[str]         # Experiences that improved motor skills
    prediction_mastery_nodes: List[str]     # Experiences with very low prediction error
    recent_experiences: List[str]           # Most recent experiences for quick access
    
    def add_experience_to_archive(self, node_id: str, experience: ExperienceNode):
        """Categorize and archive an experience based on its characteristics."""
        # High importance: low prediction error and high access frequency
        if experience.prediction_error < 0.2 and experience.access_frequency > 5:
            if node_id not in self.high_importance_experiences:
                self.high_importance_experiences.append(node_id)
        
        # Spatial memory: experiences with spatial context patterns
        if self._has_spatial_context(experience):
            if node_id not in self.spatial_memory_nodes:
                self.spatial_memory_nodes.append(node_id)
        
        # Skill learning: motor actions with good prediction accuracy
        if self._represents_motor_skill(experience):
            if node_id not in self.skill_learning_nodes:
                self.skill_learning_nodes.append(node_id)
        
        # Prediction mastery: very accurate predictions
        if experience.prediction_error < 0.1 and experience.access_frequency > 3:
            if node_id not in self.prediction_mastery_nodes:
                self.prediction_mastery_nodes.append(node_id)
        
        # Recent experiences (sliding window)
        if node_id not in self.recent_experiences:
            self.recent_experiences.append(node_id)
            if len(self.recent_experiences) > 1000:  # Keep last 1000
                self.recent_experiences.pop(0)
    
    def _has_spatial_context(self, experience: ExperienceNode) -> bool:
        """Detect if experience has spatial characteristics."""
        # Look for patterns that might represent spatial relationships
        if len(experience.mental_context) >= 6:
            # Check for position-like values (coordinates, distances)
            spatial_indicators = sum(1 for val in experience.mental_context[:6] 
                                   if abs(val) < 10.0)  # Reasonable spatial range
            return spatial_indicators >= 4
        return False
    
    def _represents_motor_skill(self, experience: ExperienceNode) -> bool:
        """Detect if experience represents motor skill learning."""
        # Motor actions with consistent results (low prediction error)
        has_motor_action = any(abs(val) > 0.1 for val in experience.action_taken.values())
        has_good_prediction = experience.prediction_error < 0.3
        return has_motor_action and has_good_prediction


class PersistentMemoryManager:
    """
    Manages persistent storage and retrieval of robot experiences and knowledge.
    Designed for 2TB storage with efficient compression and indexing.
    """
    
    def __init__(self, memory_root_path: str = "./robot_memory"):
        self.memory_root = Path(memory_root_path)
        self.memory_root.mkdir(exist_ok=True)
        
        # Storage paths
        self.graphs_path = self.memory_root / "graphs"
        self.archives_path = self.memory_root / "archives" 
        self.sessions_path = self.memory_root / "sessions"
        self.adaptive_params_path = self.memory_root / "adaptive_params"
        self.metadata_path = self.memory_root / "metadata"
        
        # Create subdirectories
        for path in [self.graphs_path, self.archives_path, self.sessions_path, 
                     self.adaptive_params_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        # Memory management
        self.current_session_id: Optional[str] = None
        self.auto_save_enabled = True
        self.auto_save_interval = 30.0  # seconds
        self.compression_enabled = True
        
        # Background saving
        self._save_thread = None
        self._stop_saving = threading.Event()
        
        # Memory archive for efficient retrieval
        self.current_archive = MemoryArchive([], [], [], [], [])
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        print(f"Persistent memory initialized at: {self.memory_root}")
        print(f"Available storage paths: graphs, archives, sessions, adaptive_params")
    
    def start_new_session(self, session_summary: str = "Robot session") -> str:
        """Start a new memory session."""
        self.current_session_id = self._generate_session_id()
        
        session = MemorySession(
            session_id=self.current_session_id,
            start_time=datetime.now(),
            end_time=None,
            experiences_count=0,
            final_graph_stats={},
            adaptive_tuning_stats={},
            session_summary=session_summary
        )
        
        # Save initial session metadata
        self._save_session_metadata(session)
        
        # Start background auto-save if enabled
        if self.auto_save_enabled:
            self._start_auto_save()
        
        print(f"Started memory session: {self.current_session_id}")
        return self.current_session_id
    
    def save_world_graph(self, world_graph: WorldGraph, session_id: Optional[str] = None) -> str:
        """Save a WorldGraph to persistent storage with compression."""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No active session. Call start_new_session() first.")
        
        timestamp = datetime.now().isoformat()
        filename = f"graph_{session_id}_{timestamp}.pkl"
        
        if self.compression_enabled:
            filepath = self.graphs_path / f"{filename}.gz"
        else:
            filepath = self.graphs_path / filename
        
        # Prepare graph data for serialization
        graph_data = {
            'nodes': {},
            'metadata': {
                'total_nodes': world_graph.node_count(),
                'latest_node_id': world_graph.latest_node_id,
                'total_nodes_created': world_graph.total_nodes_created,
                'total_merges_performed': world_graph.total_merges_performed,
                'temporal_chain': world_graph.temporal_chain,
                'graph_statistics': world_graph.get_graph_statistics(),
                'emergent_memory_stats': world_graph.get_emergent_memory_stats(),
                'save_timestamp': timestamp,
                'session_id': session_id
            }
        }
        
        # Serialize each node
        for node_id, node in world_graph.nodes.items():
            graph_data['nodes'][node_id] = self._serialize_experience_node(node)
            
            # Update archive with this experience
            self.current_archive.add_experience_to_archive(node_id, node)
        
        # Save with compression if enabled
        try:
            if self.compression_enabled:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata['last_save_time'] = timestamp
            self.metadata['total_saved_graphs'] = self.metadata.get('total_saved_graphs', 0) + 1
            self.metadata['total_experiences'] = len(graph_data['nodes'])
            self._save_metadata()
            
            # Save archive
            self._save_archive()
            
            print(f"Saved world graph: {filepath} ({len(graph_data['nodes'])} experiences)")
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving world graph: {e}")
            raise
    
    def load_latest_world_graph(self) -> Optional[WorldGraph]:
        """Load the most recent WorldGraph from storage."""
        graph_files = list(self.graphs_path.glob("*.pkl*"))
        
        if not graph_files:
            print("No saved graphs found.")
            return None
        
        # Sort by modification time (most recent first)
        latest_file = max(graph_files, key=lambda p: p.stat().st_mtime)
        
        return self.load_world_graph(str(latest_file))
    
    def load_world_graph(self, filepath: str) -> Optional[WorldGraph]:
        """Load a WorldGraph from a specific file."""
        filepath = Path(filepath)
        
        try:
            # Load data with compression detection
            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rb') as f:
                    graph_data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    graph_data = pickle.load(f)
            
            # Reconstruct WorldGraph
            world_graph = WorldGraph()
            
            # Restore metadata
            metadata = graph_data.get('metadata', {})
            world_graph.total_nodes_created = metadata.get('total_nodes_created', 0)
            world_graph.total_merges_performed = metadata.get('total_merges_performed', 0)
            world_graph.latest_node_id = metadata.get('latest_node_id')
            world_graph.temporal_chain = metadata.get('temporal_chain', [])
            
            # Restore nodes
            nodes_data = graph_data.get('nodes', {})
            for node_id, node_data in nodes_data.items():
                experience_node = self._deserialize_experience_node(node_data)
                world_graph.nodes[node_id] = experience_node
                
                # Rebuild strength index
                if hasattr(world_graph, '_update_strength_index'):
                    world_graph._update_strength_index(node_id, experience_node.strength)
            
            print(f"Loaded world graph: {filepath} ({len(nodes_data)} experiences)")
            print(f"Graph stats: {world_graph.get_graph_statistics()}")
            
            return world_graph
            
        except Exception as e:
            print(f"Error loading world graph from {filepath}: {e}")
            return None
    
    def save_adaptive_parameters(self, adaptive_tuner: AdaptiveParameterTuner, 
                               session_id: Optional[str] = None) -> str:
        """Save adaptive parameter tuning state."""
        if session_id is None:
            session_id = self.current_session_id
        
        timestamp = datetime.now().isoformat()
        filename = f"adaptive_params_{session_id}_{timestamp}.json"
        filepath = self.adaptive_params_path / filename
        
        # Prepare adaptive tuning data
        adaptive_data = {
            'current_parameters': adaptive_tuner.current_parameters,
            'adaptation_statistics': adaptive_tuner.get_adaptation_statistics(),
            'sensory_insights': adaptive_tuner.get_sensory_insights(),
            'parameter_performance_summary': adaptive_tuner._get_parameter_performance_summary(),
            'metadata': {
                'save_timestamp': timestamp,
                'session_id': session_id,
                'total_adaptations': adaptive_tuner.total_adaptations,
                'successful_adaptations': adaptive_tuner.successful_adaptations
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(adaptive_data, f, indent=2, default=str)
            
            print(f"Saved adaptive parameters: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving adaptive parameters: {e}")
            raise
    
    def load_latest_adaptive_parameters(self) -> Optional[Dict[str, Any]]:
        """Load the most recent adaptive parameters."""
        param_files = list(self.adaptive_params_path.glob("*.json"))
        
        if not param_files:
            print("No saved adaptive parameters found.")
            return None
        
        # Sort by modification time (most recent first)
        latest_file = max(param_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                adaptive_data = json.load(f)
            
            print(f"Loaded adaptive parameters: {latest_file}")
            return adaptive_data
            
        except Exception as e:
            print(f"Error loading adaptive parameters: {e}")
            return None
    
    def get_archived_experiences_by_type(self, experience_type: str, limit: int = 100) -> List[str]:
        """Retrieve archived experience node IDs by type."""
        archive = self._load_archive()
        
        type_mapping = {
            'high_importance': archive.high_importance_experiences,
            'spatial': archive.spatial_memory_nodes,
            'skills': archive.skill_learning_nodes,
            'mastery': archive.prediction_mastery_nodes,
            'recent': archive.recent_experiences
        }
        
        experiences = type_mapping.get(experience_type, [])
        return experiences[-limit:] if limit > 0 else experiences
    
    def search_experiences_by_context(self, target_context: List[float], 
                                    similarity_threshold: float = 0.7,
                                    limit: int = 10) -> List[Tuple[str, float]]:
        """Search for experiences similar to a target context across all saved graphs."""
        # This would require loading and searching multiple graphs
        # For now, return from current archive
        # In a full implementation, this would use indexing for efficiency
        similar_experiences = []
        
        # Load recent graph for context search
        recent_graph = self.load_latest_world_graph()
        if recent_graph:
            similar_nodes = recent_graph.find_similar_nodes(target_context, similarity_threshold, limit)
            similar_experiences = [(node.node_id, 1.0) for node in similar_nodes]  # Placeholder similarity
        
        return similar_experiences
    
    def end_session(self, world_graph: WorldGraph, adaptive_tuner: AdaptiveParameterTuner) -> Dict[str, Any]:
        """End the current session and save final state."""
        if self.current_session_id is None:
            raise ValueError("No active session to end.")
        
        # Stop auto-save
        if self._save_thread:
            self._stop_saving.set()
            self._save_thread.join(timeout=5.0)
        
        # Save final state
        graph_path = self.save_world_graph(world_graph)
        params_path = self.save_adaptive_parameters(adaptive_tuner)
        
        # Update session metadata
        session_file = self.sessions_path / f"{self.current_session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            session_data['end_time'] = datetime.now().isoformat()
            session_data['experiences_count'] = world_graph.node_count()
            session_data['final_graph_stats'] = world_graph.get_graph_statistics()
            session_data['adaptive_tuning_stats'] = adaptive_tuner.get_adaptation_statistics()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
        
        session_summary = {
            'session_id': self.current_session_id,
            'graph_path': graph_path,
            'params_path': params_path,
            'experiences_count': world_graph.node_count(),
            'total_adaptations': adaptive_tuner.total_adaptations
        }
        
        print(f"Ended session {self.current_session_id}")
        print(f"Saved {world_graph.node_count()} experiences")
        print(f"Total adaptations: {adaptive_tuner.total_adaptations}")
        
        self.current_session_id = None
        return session_summary
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about persistent memory usage."""
        stats = {
            'memory_root': str(self.memory_root),
            'storage_usage': self._calculate_storage_usage(),
            'total_sessions': len(list(self.sessions_path.glob("*.json"))),
            'total_graphs': len(list(self.graphs_path.glob("*.pkl*"))),
            'total_adaptive_saves': len(list(self.adaptive_params_path.glob("*.json"))),
            'current_session': self.current_session_id,
            'metadata': self.metadata,
            'archive_summary': {
                'high_importance': len(self.current_archive.high_importance_experiences),
                'spatial_memory': len(self.current_archive.spatial_memory_nodes),
                'skill_learning': len(self.current_archive.skill_learning_nodes),
                'prediction_mastery': len(self.current_archive.prediction_mastery_nodes),
                'recent_experiences': len(self.current_archive.recent_experiences)
            }
        }
        
        return stats
    
    # Private helper methods
    
    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"session_{timestamp}_{hash_suffix}"
    
    def _serialize_experience_node(self, node: ExperienceNode) -> Dict[str, Any]:
        """Convert ExperienceNode to serializable dictionary."""
        return {
            'mental_context': node.mental_context,
            'action_taken': node.action_taken,
            'predicted_sensory': node.predicted_sensory,
            'actual_sensory': node.actual_sensory,
            'prediction_error': node.prediction_error,
            'node_id': node.node_id,
            'strength': node.strength,
            'timestamp': node.timestamp.isoformat(),
            'activation_level': node.activation_level,
            'recency_bonus': node.recency_bonus,
            'access_frequency': node.access_frequency,
            'connection_weights': node.connection_weights,
            'consolidation_strength': node.consolidation_strength,
            'last_activation_time': node.last_activation_time,
            'temporal_predecessor': node.temporal_predecessor,
            'temporal_successor': node.temporal_successor,
            'prediction_sources': node.prediction_sources,
            'similar_contexts': node.similar_contexts,
            'times_accessed': node.times_accessed,
            'last_accessed': node.last_accessed.isoformat(),
            'merge_count': node.merge_count
        }
    
    def _deserialize_experience_node(self, data: Dict[str, Any]) -> ExperienceNode:
        """Convert serialized dictionary back to ExperienceNode."""
        node = ExperienceNode(
            mental_context=data['mental_context'],
            action_taken=data['action_taken'],
            predicted_sensory=data['predicted_sensory'],
            actual_sensory=data['actual_sensory'],
            prediction_error=data['prediction_error']
        )
        
        # Restore all fields
        node.node_id = data['node_id']
        node.strength = data['strength']
        node.timestamp = datetime.fromisoformat(data['timestamp'])
        node.activation_level = data.get('activation_level', 1.0)
        node.recency_bonus = data.get('recency_bonus', 1.0)
        node.access_frequency = data.get('access_frequency', 0)
        node.connection_weights = data.get('connection_weights', {})
        node.consolidation_strength = data.get('consolidation_strength', 1.0)
        node.last_activation_time = data.get('last_activation_time', time.time())
        node.temporal_predecessor = data.get('temporal_predecessor')
        node.temporal_successor = data.get('temporal_successor')
        node.prediction_sources = data.get('prediction_sources', [])
        node.similar_contexts = data.get('similar_contexts', [])
        node.times_accessed = data.get('times_accessed', 0)
        node.last_accessed = datetime.fromisoformat(data['last_accessed'])
        node.merge_count = data.get('merge_count', 0)
        
        return node
    
    def _save_session_metadata(self, session: MemorySession):
        """Save session metadata."""
        session_file = self.sessions_path / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(session), f, indent=2, default=str)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create metadata."""
        metadata_file = self.metadata_path / "memory_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Create default metadata
        return {
            'created_time': datetime.now().isoformat(),
            'total_saved_graphs': 0,
            'total_experiences': 0,
            'last_save_time': None,
            'memory_version': '1.0'
        }
    
    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_file = self.metadata_path / "memory_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _save_archive(self):
        """Save current memory archive."""
        archive_file = self.archives_path / "current_archive.json"
        with open(archive_file, 'w') as f:
            json.dump(asdict(self.current_archive), f, indent=2)
    
    def _load_archive(self) -> MemoryArchive:
        """Load memory archive or create new one."""
        archive_file = self.archives_path / "current_archive.json"
        
        if archive_file.exists():
            try:
                with open(archive_file, 'r') as f:
                    archive_data = json.load(f)
                return MemoryArchive(**archive_data)
            except Exception as e:
                print(f"Error loading archive: {e}")
        
        return MemoryArchive([], [], [], [], [])
    
    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """Calculate storage usage statistics."""
        def get_dir_size(path):
            total = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total
        
        return {
            'total_bytes': get_dir_size(self.memory_root),
            'graphs_bytes': get_dir_size(self.graphs_path),
            'archives_bytes': get_dir_size(self.archives_path),
            'sessions_bytes': get_dir_size(self.sessions_path),
            'adaptive_params_bytes': get_dir_size(self.adaptive_params_path),
            'metadata_bytes': get_dir_size(self.metadata_path)
        }
    
    def _start_auto_save(self):
        """Start background auto-save thread."""
        if self._save_thread and self._save_thread.is_alive():
            return
        
        self._stop_saving.clear()
        self._save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._save_thread.start()
    
    def _auto_save_loop(self):
        """Background auto-save loop."""
        while not self._stop_saving.is_set():
            if self._stop_saving.wait(self.auto_save_interval):
                break
            
            # Auto-save would be triggered here if needed
            # For now, just indicate it's running
            if self.current_session_id:
                print(f"Auto-save active for session {self.current_session_id}")