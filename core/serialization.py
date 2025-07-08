"""
Graph serialization utilities for debugging and analysis.
Provides methods to save and load WorldGraph instances.
"""

import json
import pickle
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from .world_graph import WorldGraph
from .experience_node import ExperienceNode


class GraphSerializer:
    """Handles serialization and deserialization of WorldGraph instances."""
    
    @staticmethod
    def to_json(graph: WorldGraph, filepath: str, include_metadata: bool = True) -> bool:
        """
        Serialize WorldGraph to JSON format.
        Returns True if successful, False otherwise.
        """
        try:
            # Convert graph to JSON-serializable format
            data = {
                "metadata": {
                    "serialization_time": datetime.now().isoformat(),
                    "total_nodes": graph.node_count(),
                    "total_merges": graph.total_merges_performed,
                    "latest_node_id": graph.latest_node_id
                } if include_metadata else {},
                "nodes": {},
                "temporal_chain": graph.temporal_chain,
                "statistics": graph.get_graph_statistics()
            }
            
            # Serialize each node
            for node_id, node in graph.nodes.items():
                data["nodes"][node_id] = {
                    "mental_context": node.mental_context,
                    "action_taken": node.action_taken,
                    "predicted_sensory": node.predicted_sensory,
                    "actual_sensory": node.actual_sensory,
                    "prediction_error": node.prediction_error,
                    "node_id": node.node_id,
                    "strength": node.strength,
                    "timestamp": node.timestamp.isoformat(),
                    "temporal_predecessor": node.temporal_predecessor,
                    "temporal_successor": node.temporal_successor,
                    "prediction_sources": node.prediction_sources,
                    "similar_contexts": node.similar_contexts,
                    "times_accessed": node.times_accessed,
                    "last_accessed": node.last_accessed.isoformat(),
                    "merge_count": node.merge_count
                }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error serializing graph to JSON: {e}")
            return False
    
    @staticmethod
    def from_json(filepath: str) -> Optional[WorldGraph]:
        """
        Deserialize WorldGraph from JSON format.
        Returns WorldGraph instance or None if failed.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create new graph
            graph = WorldGraph()
            
            # Restore metadata
            if "metadata" in data:
                graph.total_nodes_created = data["metadata"].get("total_nodes", 0)
                graph.total_merges_performed = data["metadata"].get("total_merges", 0)
                graph.latest_node_id = data["metadata"].get("latest_node_id")
            
            # Restore nodes
            for node_id, node_data in data["nodes"].items():
                node = ExperienceNode(
                    mental_context=node_data["mental_context"],
                    action_taken=node_data["action_taken"],
                    predicted_sensory=node_data["predicted_sensory"],
                    actual_sensory=node_data["actual_sensory"],
                    prediction_error=node_data["prediction_error"],
                    node_id=node_data["node_id"],
                    strength=node_data["strength"],
                    timestamp=datetime.fromisoformat(node_data["timestamp"]),
                    temporal_predecessor=node_data["temporal_predecessor"],
                    temporal_successor=node_data["temporal_successor"],
                    prediction_sources=node_data["prediction_sources"],
                    similar_contexts=node_data["similar_contexts"],
                    times_accessed=node_data["times_accessed"],
                    last_accessed=datetime.fromisoformat(node_data["last_accessed"]),
                    merge_count=node_data["merge_count"]
                )
                
                # Add to graph (but don't auto-link temporally)
                graph.nodes[node_id] = node
                graph._update_strength_index(node_id, node.strength)
            
            # Restore temporal chain
            if "temporal_chain" in data:
                graph.temporal_chain = data["temporal_chain"]
            
            return graph
            
        except Exception as e:
            print(f"Error deserializing graph from JSON: {e}")
            return None
    
    @staticmethod
    def to_pickle(graph: WorldGraph, filepath: str) -> bool:
        """
        Serialize WorldGraph to pickle format (faster, binary).
        Returns True if successful, False otherwise.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
            return True
            
        except Exception as e:
            print(f"Error serializing graph to pickle: {e}")
            return False
    
    @staticmethod
    def from_pickle(filepath: str) -> Optional[WorldGraph]:
        """
        Deserialize WorldGraph from pickle format.
        Returns WorldGraph instance or None if failed.
        """
        try:
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            return graph
            
        except Exception as e:
            print(f"Error deserializing graph from pickle: {e}")
            return None
    
    @staticmethod
    def export_summary(graph: WorldGraph, filepath: str) -> bool:
        """
        Export a human-readable summary of the graph.
        Returns True if successful, False otherwise.
        """
        try:
            stats = graph.get_graph_statistics()
            
            summary = f"""WorldGraph Summary
================

Generated: {datetime.now().isoformat()}

Basic Statistics:
- Total nodes: {stats['total_nodes']}
- Total merges performed: {stats['total_merges']}
- Average strength: {stats['avg_strength']:.3f}
- Max strength: {stats['max_strength']:.3f}
- Min strength: {stats['min_strength']:.3f}
- Average context length: {stats['avg_context_length']:.1f}
- Temporal chain length: {stats['temporal_chain_length']}
- Average access count: {stats['avg_access_count']:.1f}
- Total accesses: {stats['total_accesses']}

Top 10 Strongest Nodes:
"""
            
            # Add top nodes info
            strongest_nodes = graph.get_strongest_nodes(10)
            for i, node in enumerate(strongest_nodes, 1):
                summary += f"{i:2d}. Node {node.node_id[:8]}... (strength: {node.strength:.3f}, accessed: {node.times_accessed}x)\n"
            
            summary += "\nWeakest 5 Nodes:\n"
            weakest_nodes = graph.get_weakest_nodes(5)
            for i, node in enumerate(weakest_nodes, 1):
                summary += f"{i:2d}. Node {node.node_id[:8]}... (strength: {node.strength:.3f}, accessed: {node.times_accessed}x)\n"
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(summary)
            
            return True
            
        except Exception as e:
            print(f"Error exporting graph summary: {e}")
            return False
    
    @staticmethod
    def export_node_data(graph: WorldGraph, filepath: str, include_context: bool = False) -> bool:
        """
        Export detailed node data as CSV for analysis.
        Returns True if successful, False otherwise.
        """
        try:
            import csv
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'node_id', 'strength', 'times_accessed', 'merge_count',
                    'prediction_error', 'context_length', 'action_count',
                    'has_predecessor', 'has_successor', 'similar_contexts_count'
                ]
                
                if include_context:
                    # Add context dimensions as separate columns
                    max_context_length = max(len(node.mental_context) for node in graph.nodes.values()) if graph.nodes else 0
                    fieldnames.extend([f'context_{i}' for i in range(max_context_length)])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for node in graph.nodes.values():
                    row = {
                        'node_id': node.node_id,
                        'strength': node.strength,
                        'times_accessed': node.times_accessed,
                        'merge_count': node.merge_count,
                        'prediction_error': node.prediction_error,
                        'context_length': len(node.mental_context),
                        'action_count': len(node.action_taken),
                        'has_predecessor': node.temporal_predecessor is not None,
                        'has_successor': node.temporal_successor is not None,
                        'similar_contexts_count': len(node.similar_contexts)
                    }
                    
                    if include_context:
                        for i, value in enumerate(node.mental_context):
                            row[f'context_{i}'] = value
                    
                    writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Error exporting node data: {e}")
            return False


def save_graph_debug_info(graph: WorldGraph, base_filename: str) -> Dict[str, bool]:
    """
    Save comprehensive debug information about a graph.
    Creates multiple files with different formats and levels of detail.
    """
    results = {}
    
    # Create output directory if it doesn't exist
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    # Full graph as JSON
    json_path = output_dir / f"{base_filename}_full.json"
    results['json'] = GraphSerializer.to_json(graph, str(json_path))
    
    # Full graph as pickle (faster for large graphs)
    pickle_path = output_dir / f"{base_filename}_full.pkl"
    results['pickle'] = GraphSerializer.to_pickle(graph, str(pickle_path))
    
    # Human-readable summary
    summary_path = output_dir / f"{base_filename}_summary.txt"
    results['summary'] = GraphSerializer.export_summary(graph, str(summary_path))
    
    # Node data as CSV
    csv_path = output_dir / f"{base_filename}_nodes.csv"
    results['csv'] = GraphSerializer.export_node_data(graph, str(csv_path))
    
    return results