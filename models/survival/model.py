"""Survival analysis engine: DeepSurv and Dynamic-DeepHit for time-to-event prediction.

This module implements survival models for predicting:
- Time to 30-day readmission
- Time to complications
- Time to mortality

Models:
- DeepSurv: Deep learning extension of Cox proportional hazards
- Dynamic-DeepHit: Discrete-time survival with competing risks

Both models output hazard rates / survival probabilities used by the ensemble.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from healthrisk_ai.config import ModelConfig, build_config
from healthrisk_ai.exceptions import ModelError
from healthrisk_ai.numerics import time_aware_train_test_split

logger = logging.getLogger("healthrisk_ai.models.survival")


class SurvivalDataset(Dataset):
    """PyTorch Dataset for survival data.

    Each sample contains:
    - features: covariate vector
    - time: follow-up time
    - event: event indicator (1=event, 0=censored)
    """

    def __init__(
        self,
        features: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "features": self.features[idx],
            "time": self.times[idx],
            "event": self.events[idx],
        }


class DeepSurv(nn.Module):
    """DeepSurv: Deep learning extension of Cox proportional hazards.

    Architecture:
        Input -> Hidden layers (configurable) -> Linear output (log hazard ratio)

    The model learns a risk score h(x) such that the hazard function is:
        λ(t|x) = λ₀(t) * exp(h(x))

    Loss: Partial likelihood (Cox loss)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [128, 64, 32],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Compute log hazard ratio for each sample.

        Args:
            x: Feature matrix [batch, input_dim].

        Returns:
            Log hazard ratios [batch, 1].
        """
        return self.network(x).squeeze(-1)

    def cox_partial_likelihood_loss(
        self,
        x: Tensor,
        times: Tensor,
        events: Tensor,
    ) -> Tensor:
        """Compute Cox partial likelihood loss.

        Args:
            x: Features [batch, input_dim].
            times: Event/censoring times [batch].
            events: Event indicators [batch].

        Returns:
            Negative partial likelihood (to minimize).
        """
        log_hazard = self.forward(x)

        # Sort by time (ascending)
        sorted_indices = torch.argsort(times)
        log_hazard = log_hazard[sorted_indices]
        times = times[sorted_indices]
        events = events[sorted_indices]

        # Risk set: all samples with time >= current time
        risk_set_sums = torch.cumsum(torch.exp(log_hazard), dim=0)

        # Loss contribution for each event
        loss = torch.tensor(0.0, device=x.device)
        event_count = 0

        for i in range(len(events)):
            if events[i] > 0.5:
                # Log hazard for this patient minus log of risk set sum
                loss = loss + log_hazard[i] - torch.log(risk_set_sums[i] + 1e-8)
                event_count += 1

        if event_count > 0:
            loss = -loss / event_count
        else:
            loss = torch.tensor(0.0, device=x.device)

        return loss


class DynamicDeepHit(nn.Module):
    """Dynamic-DeepHit: Discrete-time survival model with competing risks.

    Models the conditional probability of event at time t given survival up to t-1:
        P(T=t | T>=t, X) = softmax(shared_representation + time_specific_head)

    Supports a single event type (can be extended to competing risks).

    Args:
        input_dim: Feature dimension.
        max_time_steps: Maximum number of discrete time intervals.
        hidden_layers: Hidden layer dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        max_time_steps: int = 30,
        hidden_layers: list[int] = [128, 64, 32],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.max_time_steps = max_time_steps

        # Shared representation layers
        shared_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.shared_network = nn.Sequential(*shared_layers)

        # Time-specific output heads
        self.time_heads = nn.ModuleList([
            nn.Linear(prev_dim, 2) for _ in range(max_time_steps)
        ])

    def forward(
        self,
        x: Tensor,
        times: Tensor | None = None,
    ) -> Tensor:
        """Compute survival probabilities.

        Args:
            x: Feature matrix [batch, input_dim].
            times: Optional time indices for selection [batch].

        Returns:
            If times is None: probabilities [batch, max_time_steps, 2].
            If times is given: selected probabilities [batch, 2].
        """
        shared = self.shared_network(x)

        if times is None:
            # Return all time-step probabilities
            all_probs = []
            for t in range(self.max_time_steps):
                logits = self.time_heads[t](shared)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)
            return torch.stack(all_probs, dim=1)
        else:
            # Return probabilities for specific times
            batch_size = x.size(0)
            selected_probs = []
            for i in range(batch_size):
                t = min(int(times[i].item()), self.max_time_steps - 1)
                logits = self.time_heads[t](shared[i:i+1])
                probs = torch.softmax(logits, dim=-1)
                selected_probs.append(probs)
            return torch.cat(selected_probs, dim=0)


class SurvivalModel:
    """High-level wrapper for survival models.

    Supports both DeepSurv and Dynamic-DeepHit architectures.
    """

    def __init__(
        self,
        input_dim: int,
        model_type: str = "deepsurv",
        hidden_layers: list[int] = [128, 64, 32],
        dropout: float = 0.2,
        max_time_steps: int = 30,
        device: str = "auto",
        cindex_target: float = 0.70,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.model_type = model_type
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.max_time_steps = max_time_steps
        self.cindex_target = cindex_target

        self._build_model()
        self._is_trained = False

    def _build_model(self) -> None:
        """Build the appropriate survival model."""
        if self.model_type == "deepsurv":
            self.model = DeepSurv(
                input_dim=self.input_dim,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
            ).to(self.device)
        elif self.model_type == "deepdephit":
            self.model = DynamicDeepHit(
                input_dim=self.input_dim,
                max_time_steps=self.max_time_steps,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
            ).to(self.device)
        else:
            msg = f"Unknown survival model type: {self.model_type}"
            raise ModelError(msg)

    def train(
        self,
        features: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        validation_split: float = 0.2,
        seed: int = 20260914,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the survival model.

        Args:
            features: Feature matrix [n_samples, n_features].
            times: Follow-up times [n_samples].
            events: Event indicators [n_samples].
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            validation_split: Fraction for validation.
            seed: Random seed.
            verbose: Log progress.

        Returns:
            Training history with train/val loss and C-index.
        """
        if len(features) != len(times) or len(features) != len(events):
            msg = "features, times, and events must have same length"
            raise ModelError(msg)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split data
        n = len(features)
        indices = np.random.permutation(n)
        val_size = int(n * validation_split)

        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        train_features = features[train_idx]
        train_times = times[train_idx]
        train_events = events[train_idx]

        val_features = features[val_idx]
        val_times = times[val_idx]
        val_events = events[val_idx]

        # Create datasets
        train_dataset = SurvivalDataset(train_features, train_times, train_events)
        val_dataset = SurvivalDataset(val_features, val_times, val_events)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_cindex": [],
            "val_cindex": [],
        }

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training
            epoch_train_loss = 0.0
            for batch in train_loader:
                x = batch["features"].to(self.device)
                t = batch["time"].to(self.device)
                e = batch["event"].to(self.device)

                optimizer.zero_grad()
                loss = self.model.cox_partial_likelihood_loss(x, t, e)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["features"].to(self.device)
                    t = batch["time"].to(self.device)
                    e = batch["event"].to(self.device)
                    loss = self.model.cox_partial_likelihood_loss(x, t, e)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            # C-index
            train_cindex = self._compute_c_index(train_features, train_times, train_events)
            val_cindex = self._compute_c_index(val_features, val_times, val_events)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_cindex"].append(train_cindex)
            history["val_cindex"].append(val_cindex)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d, train_loss=%.4f, val_loss=%.4f, "
                    "train_cindex=%.4f, val_cindex=%.4f",
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    avg_val_loss,
                    train_cindex,
                    val_cindex,
                )

            self.model.train()

        self._is_trained = True
        logger.info(
            "Survival model training complete. Best val C-index: %.4f",
            max(history["val_cindex"]),
        )
        return history

    def _compute_c_index(
        self,
        features: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """Compute concordance index (C-index).

        Args:
            features: Feature matrix.
            times: Follow-up times.
            events: Event indicators.

        Returns:
            C-index value.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            risk_scores = self.model.forward(x).cpu().numpy()

        # Simple C-index implementation
        n = len(times)
        concordant = 0
        comparable = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Comparable pair: one has event before the other's time
                if events[i] > 0 and times[i] < times[j]:
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                elif events[j] > 0 and times[j] < times[i]:
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1

        if comparable == 0:
            return 0.5

        return concordant / comparable

    def predict_hazard(self, features: np.ndarray) -> np.ndarray:
        """Predict log hazard ratios for samples.

        Args:
            features: Feature matrix [n_samples, n_features].

        Returns:
            Log hazard ratios [n_samples].
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            hazards = self.model.forward(x)
        return hazards.cpu().numpy()

    def predict_survival_probability(
        self,
        features: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict survival probabilities.

        Args:
            features: Feature matrix.
            times: Time points to evaluate (for DeepHit).

        Returns:
            Survival probabilities [n_samples] or [n_samples, n_times].
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)

            if self.model_type == "deepdephit" and times is not None:
                t = torch.tensor(times, dtype=torch.float32).to(self.device)
                probs = self.model.forward(x, t)
                # Probability of event at the specified time
                return probs[:, 1].cpu().numpy()
            else:
                # For DeepSurv, convert log hazard to survival
                log_hazard = self.model.forward(x)
                # Simple exponential decay approximation
                baseline_hazard = 0.01
                survival = np.exp(-baseline_hazard * np.exp(log_hazard.numpy()))
                return survival

    def save(self, path: str) -> None:
        """Save model.

        Args:
            path: Directory to save to.
        """
        import os

        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "max_time_steps": self.max_time_steps,
                "cindex_target": self.cindex_target,
            },
            os.path.join(path, "survival.pt"),
        )
        logger.info("Saved survival model to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> SurvivalModel:
        """Load a saved model.

        Args:
            path: Directory containing saved model.
            device: Device to load on.

        Returns:
            Loaded SurvivalModel instance.
        """
        import os

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(os.path.join(path, "survival.pt"), map_location=device)
        model = cls(
            input_dim=checkpoint["input_dim"],
            model_type=checkpoint["model_type"],
            hidden_layers=checkpoint["hidden_layers"],
            dropout=checkpoint["dropout"],
            max_time_steps=checkpoint.get("max_time_steps", 30),
            device=device,
            cindex_target=checkpoint.get("cindex_target", 0.70),
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model._is_trained = True
        logger.info("Loaded survival model from %s", path)
        return model

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained


def build_survival_model(config: ModelConfig | None = None) -> SurvivalModel:
    """Build a SurvivalModel from config.

    Args:
        config: Model config; uses default if None.

    Returns:
        Initialized SurvivalModel.
    """
    if config is None:
        _, _, _, _, _, config, _ = build_config()

    survival_cfg = config.survival
    return SurvivalModel(
        input_dim=64 + 32,  # Combined NLP + GNN embedding dimension
        model_type="deepsurv",
        hidden_layers=survival_cfg.deep_surv_hidden,
        dropout=0.2,
        max_time_steps=30,
        device=survival_cfg.device if hasattr(survival_cfg, 'device') else "auto",
        cindex_target=survival_cfg.cindex_target,
    )
