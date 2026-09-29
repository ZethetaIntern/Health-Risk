"""Graph Neural Network: Heterogeneous GATv2 over Patient-Disease-Drug nodes.

This module implements a Heterogeneous Graph Attention Network (GATv2) using
PyTorch Geometric that operates over a tripartite graph:
- Patient nodes
- Disease nodes (ICD-10 codes)
- Drug nodes (medication codes)

The model produces 32-dimensional patient embeddings that capture:
- Patient-disease associations
- Patient-drug prescriptions
- Disease co-occurrence patterns
- Drug co-prescription patterns

The patient embeddings feed into the stacking ensemble as graph-based features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.utils import to_undirected

from healthrisk_ai.config import ModelConfig, build_config
from healthrisk_ai.exceptions import ModelError

logger = logging.getLogger("healthrisk_ai.models.graph")


class PatientDiseaseDrugGraph:
    """Constructs a heterogeneous graph from patient event data.

    Builds a tripartite graph with:
    - patient nodes (indexed by patient_id)
    - disease nodes (indexed by ICD-10 code)
    - drug nodes (indexed by medication code)

    Edges:
    - patient-disease: patient has diagnosis
    - patient-drug: patient prescribed medication
    - disease-disease: co-occurrence (optional)
    - drug-drug: co-prescription (optional)
    """

    def __init__(
        self,
        patient_ids: list[int],
        disease_codes: list[str],
        drug_codes: list[str],
        patient_disease_edges: list[tuple[int, str]] | None = None,
        patient_drug_edges: list[tuple[int, str]] | None = None,
    ) -> None:
        self.patient_ids = patient_ids
        # Sorted so index mappings are deterministic across runs.
        self.disease_codes = sorted(set(disease_codes))
        self.drug_codes = sorted(set(drug_codes))

        self.patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
        self.disease_code_to_idx = {code: i for i, code in enumerate(self.disease_codes)}
        self.drug_code_to_idx = {code: i for i, code in enumerate(self.drug_codes)}

        self._build_edge_index(patient_disease_edges or [], patient_drug_edges or [])

    def _build_edge_index(
        self,
        patient_disease: list[tuple[int, str]],
        patient_drug: list[tuple[int, str]],
    ) -> None:
        """Build edge indices for the heterogeneous graph."""
        num_patients = len(self.patient_ids)
        num_diseases = len(self.disease_codes)
        num_drugs = len(self.drug_codes)

        # Patient -> Disease edges.
        # NOTE: to_hetero_data() exposes per-node-type index spaces (patient
        # nodes 0..num_patients-1, disease nodes 0..num_diseases-1, etc.), so
        # edge endpoints must be local per-type indices without global offsets.
        patient_disease_src: list[int] = []
        patient_disease_dst: list[int] = []
        for pid, code in patient_disease:
            if pid in self.patient_id_to_idx and code in self.disease_code_to_idx:
                patient_disease_src.append(self.patient_id_to_idx[pid])
                patient_disease_dst.append(self.disease_code_to_idx[code])

        # Patient -> Drug edges (reversed to drug -> patient for message
        # passing, still using local per-type indices).
        patient_drug_src: list[int] = []
        patient_drug_dst: list[int] = []
        for pid, code in patient_drug:
            if pid in self.patient_id_to_idx and code in self.drug_code_to_idx:
                patient_drug_src.append(self.drug_code_to_idx[code])
                patient_drug_dst.append(self.patient_id_to_idx[pid])

        total_nodes = num_patients + num_diseases + num_drugs

        self.edge_index = {
            ("patient", "has_disease", "disease"): [
                patient_disease_src,
                patient_disease_dst,
            ],
            ("drug", "prescribed_to", "patient"): [
                patient_drug_src,
                patient_drug_dst,
            ],
        }

        self.total_nodes = total_nodes
        self.num_patients = num_patients
        self.num_diseases = num_diseases
        self.num_drugs = num_drugs

    def to_hetero_data(self) -> HeteroData:
        """Convert to PyG HeteroData object."""
        data = HeteroData()

        data["patient"].num_nodes = self.num_patients
        data["disease"].num_nodes = self.num_diseases
        data["drug"].num_nodes = self.num_drugs

        for edge_type, edges in self.edge_index.items():
            src_type, relation, dst_type = edge_type
            src_idx, dst_idx = edges
            if len(src_idx) > 0:
                data[(src_type, relation, dst_type)].edge_index = torch.tensor(
                    [src_idx, dst_idx], dtype=torch.long
                )

        return data

    def get_patient_embedding_indices(self) -> list[int]:
        """Return the node indices corresponding to patients."""
        return list(range(self.num_patients))


class GATv2Heterogeneous(nn.Module):
    """Heterogeneous GATv2 graph neural network for patient embeddings.

    Architecture:
        - Separate GATv2Conv for each edge type
        - Hidden layers with configurable heads
        - Final projection to patient embedding dimension

    Args:
        num_patient_nodes: Number of patient nodes.
        num_disease_nodes: Number of disease nodes.
        num_drug_nodes: Number of drug nodes.
        patient_embedding_dim: Output embedding dimension for patients.
        hidden_channels: Hidden channel dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        edge_types: List of edge types to process.
    """

    def __init__(
        self,
        num_patient_nodes: int,
        num_disease_nodes: int,
        num_drug_nodes: int,
        patient_embedding_dim: int = 32,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.patient_embedding_dim = patient_embedding_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        # Node feature projections (learnable embeddings for each node type)
        self.patient_emb = nn.Embedding(num_patient_nodes, hidden_channels)
        self.disease_emb = nn.Embedding(num_disease_nodes, hidden_channels)
        self.drug_emb = nn.Embedding(num_drug_nodes, hidden_channels)

        # Hetero convolution with GATv2 for each edge type
        self.conv1 = HeteroConv(
            {
                ("patient", "has_disease", "disease"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
                ("drug", "prescribed_to", "patient"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("patient", "has_disease", "disease"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
                ("drug", "prescribed_to", "patient"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )

        # Final projection to patient embeddings
        self.project_patient = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, patient_embedding_dim),
            nn.LayerNorm(patient_embedding_dim),
        )

    def forward(self, data: HeteroData) -> Tensor:
        """Forward pass through the GATv2 network.

        Args:
            data: HeteroData object with patient, disease, drug nodes.

        Returns:
            Patient embeddings [num_patients, patient_embedding_dim].
        """
        x_dict = {
            "patient": self.patient_emb.weight,
            "disease": self.disease_emb.weight,
            "drug": self.drug_emb.weight,
        }

        # First GAT layer
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            k: v.reshape(v.size(0), -1) for k, v in x_dict.items()
        }  # Concatenate heads
        x_dict = {k: torch.nn.functional.dropout(v, p=0.2, training=self.training)
                  for k, v in x_dict.items()}
        x_dict = {k: torch.nn.functional.relu(v) for k, v in x_dict.items()}

        # Second GAT layer
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {
            k: v.reshape(v.size(0), -1) for k, v in x_dict.items()
        }

        # Project patient embeddings
        patient_embeddings = self.project_patient(x_dict["patient"])
        return patient_embeddings

    def get_patient_embeddings(
        self,
        data: HeteroData,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Extract patient embeddings from the graph.

        Args:
            data: HeteroData object.
            device: Device to run on.

        Returns:
            Array of shape (num_patients, patient_embedding_dim).
        """
        self.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.to(device)
        data = data.to(device)

        with torch.no_grad():
            embeddings = self.forward(data)

        return embeddings.cpu().numpy()


class GraphNeuralNetworkModel:
    """High-level wrapper for the GATv2 heterogeneous graph model.

    Provides train/eval/predict interfaces suitable for the ensemble pipeline.
    """

    def __init__(
        self,
        patient_embedding_dim: int = 32,
        hidden_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.patient_embedding_dim = patient_embedding_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.model: GATv2Heterogeneous | None = None
        self.graph: PatientDiseaseDrugGraph | None = None
        self._is_trained = False

    def build_graph(
        self,
        patient_ids: list[int],
        disease_codes: list[str],
        drug_codes: list[str],
        patient_disease_edges: list[tuple[int, str]] | None = None,
        patient_drug_edges: list[tuple[int, str]] | None = None,
    ) -> None:
        """Build the heterogeneous graph from patient data.

        Args:
            patient_ids: List of patient IDs.
            disease_codes: List of all disease codes.
            drug_codes: List of all drug codes.
            patient_disease_edges: List of (patient_id, disease_code) tuples.
            patient_drug_edges: List of (patient_id, drug_code) tuples.
        """
        self.graph = PatientDiseaseDrugGraph(
            patient_ids=patient_ids,
            disease_codes=disease_codes,
            drug_codes=drug_codes,
            patient_disease_edges=patient_disease_edges or [],
            patient_drug_edges=patient_drug_edges or [],
        )

        self.model = GATv2Heterogeneous(
            num_patient_nodes=self.graph.num_patients,
            num_disease_nodes=self.graph.num_diseases,
            num_drug_nodes=self.graph.num_drugs,
            patient_embedding_dim=self.patient_embedding_dim,
            hidden_channels=self.hidden_channels,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        logger.info(
            "Built heterogeneous graph: %d patients, %d diseases, %d drugs",
            self.graph.num_patients,
            self.graph.num_diseases,
            self.graph.num_drugs,
        )

    def train(
        self,
        targets: np.ndarray | None = None,
        *,
        epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int | None = None,
        seed: int = 20260914,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the GNN on patient embeddings.

        Args:
            targets: Optional regression targets for supervised training.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            batch_size: Batch size (not used for full-batch GNN).
            seed: Random seed.
            verbose: Log progress.

        Returns:
            Training history.
        """
        if self.model is None or self.graph is None:
            msg = "Must build graph before training"
            raise ModelError(msg)

        torch.manual_seed(seed)
        np.random.seed(seed)

        data = self.graph.to_hetero_data().to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(epochs):
            optimizer.zero_grad()

            patient_embeddings = self.model(data)

            if targets is not None:
                # Supervised: reconstruct targets from embeddings
                loss = nn.functional.mse_loss(
                    patient_embeddings,
                    torch.tensor(targets, dtype=torch.float32, device=self.device),
                )
            else:
                # Self-supervised: variance regularization
                loss = -torch.var(patient_embeddings).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            history["loss"].append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, loss.item())

        self._is_trained = True
        logger.info("GNN training complete")
        return history

    def predict_embeddings(self) -> np.ndarray:
        """Extract patient embeddings from the graph.

        Returns:
            Array of shape (num_patients, patient_embedding_dim).
        """
        if self.model is None or self.graph is None:
            msg = "Must build graph before predicting"
            raise ModelError(msg)

        data = self.graph.to_hetero_data()
        return self.model.get_patient_embeddings(data, device=self.device)

    def save(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Directory to save to.
        """
        import os

        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "patient_embedding_dim": self.patient_embedding_dim,
                    "hidden_channels": self.hidden_channels,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                    "num_patients": self.graph.num_patients if self.graph else 0,
                    "num_diseases": self.graph.num_diseases if self.graph else 0,
                    "num_drugs": self.graph.num_drugs if self.graph else 0,
                },
                os.path.join(path, "gnn.pt"),
            )
            logger.info("Saved GNN model to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> GraphNeuralNetworkModel:
        """Load a saved model.

        Args:
            path: Directory containing saved model.
            device: Device to load on.

        Returns:
            Loaded GraphNeuralNetworkModel instance.
        """
        import os

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(os.path.join(path, "gnn.pt"), map_location=device)
        model = cls(
            patient_embedding_dim=checkpoint["patient_embedding_dim"],
            hidden_channels=checkpoint["hidden_channels"],
            num_heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        model.model = GATv2Heterogeneous(
            num_patient_nodes=checkpoint["num_patients"],
            num_disease_nodes=checkpoint["num_diseases"],
            num_drug_nodes=checkpoint["num_drugs"],
            patient_embedding_dim=checkpoint["patient_embedding_dim"],
            hidden_channels=checkpoint["hidden_channels"],
            num_heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        ).to(device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model._is_trained = True
        logger.info("Loaded GNN model from %s", path)
        return model

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained


def build_graph_model(config: ModelConfig | None = None) -> GraphNeuralNetworkModel:
    """Build a GraphNeuralNetworkModel from config.

    Args:
        config: Model config; uses default if None.

    Returns:
        Initialized GraphNeuralNetworkModel.
    """
    if config is None:
        _, _, _, _, _, config, _ = build_config()

    graph_cfg = config.graph
    return GraphNeuralNetworkModel(
        patient_embedding_dim=graph_cfg.patient_embedding_dim,
        hidden_channels=graph_cfg.patient_embedding_dim * 2,
        num_heads=graph_cfg.num_heads,
        dropout=graph_cfg.dropout,
    )
