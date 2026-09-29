"""Graph Neural Network: Heterogeneous GATv2 for patient embeddings.

This package provides the GNN component of the HealthRisk AI ensemble:
- PatientDiseaseDrugGraph: Constructs heterogeneous tripartite graph
- GATv2Heterogeneous: GATv2-based heterogeneous GNN
- GraphNeuralNetworkModel: High-level train/predict interface
"""

from __future__ import annotations

from healthrisk_ai.models.graph_network.model import (
    GATv2Heterogeneous,
    GraphNeuralNetworkModel,
    PatientDiseaseDrugGraph,
    build_graph_model,
)

__all__ = [
    "GATv2Heterogeneous",
    "GraphNeuralNetworkModel",
    "PatientDiseaseDrugGraph",
    "build_graph_model",
]
