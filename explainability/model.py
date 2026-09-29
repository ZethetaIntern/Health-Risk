"""Explainability module: SHAP attributions and model card generation.

This module provides:
- SHAP-based feature importance for all models
- Automated HTML/PDF model card generation
- Model documentation and ethical consideration reporting
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("healthrisk_ai.explainability")


class SHAPExplainer:
    """SHAP-based feature importance explainer.

    Wraps various model types and computes SHAP values for
    feature attribution and explanation.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        model_type: str = "auto",
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            model: Trained model instance.
            feature_names: Names of input features.
            model_type: Type hint for model ('xgboost', 'lightgbm', 'linear', 'auto').
        """
        self.model = model
        self.feature_names = feature_names or []
        self.model_type = model_type
        self._explainer: Any = None
        self._shap_values: Any = None
        self._is_fitted = False

        # Try to import shap lazily
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.shap = None
            logger.warning("SHAP not installed; explainer will be limited")

    def fit(
        self,
        background_data: np.ndarray | pd.DataFrame,
        *,
        n_samples: int = 100,
        **kwargs: Any,
    ) -> SHAPExplainer:
        """Fit the SHAP explainer on background data.

        Args:
            background_data: Representative data samples for SHAP.
            n_samples: Number of background samples to use.
            **kwargs: Additional arguments for SHAP explainer.

        Returns:
            Self for chaining.
        """
        if self.shap is None:
            logger.warning("SHAP not available; skipping explainer fit")
            self._is_fitted = True
            return self

        if isinstance(background_data, pd.DataFrame):
            background = background_data.values
            if self.feature_names is None or len(self.feature_names) == 0:
                self.feature_names = list(background_data.columns)
        else:
            background = background_data
            if self.feature_names is None or len(self.feature_names) == 0:
                self.feature_names = [f"feature_{i}" for i in range(background.shape[1])]

        # Sample background data
        if len(background) > n_samples:
            indices = np.random.choice(len(background), n_samples, replace=False)
            background = background[indices]

        # Create appropriate explainer
        if self.model_type == "auto":
            self.model_type = self._detect_model_type()

        if self.model_type in ("xgboost", "lightgbm"):
            try:
                self._explainer = self.shap.TreeExplainer(self.model)
            except Exception:
                self._explainer = self.shap.KernelExplainer(
                    self.model.predict,
                    background,
                )
        elif self.model_type == "linear":
            self._explainer = self.shap.LinearExplainer(
                self.model,
                background,
            )
        else:
            self._explainer = self.shap.KernelExplainer(
                self.model.predict,
                background,
            )

        self._is_fitted = True
        logger.info("SHAP explainer fitted for model type: %s", self.model_type)
        return self

    def _detect_model_type(self) -> str:
        """Detect model type from model object."""
        model_str = str(type(self.model))
        if "xgboost" in model_str.lower():
            return "xgboost"
        elif "lightgbm" in model_str.lower():
            return "lightgbm"
        elif "linear" in model_str.lower() or "ridge" in model_str.lower():
            return "linear"
        else:
            return "unknown"

    def explain(
        self,
        instances: np.ndarray | pd.DataFrame,
        *,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values for instances.

        Args:
            instances: Data to explain.
            max_samples: Max samples to explain (for performance).

        Returns:
            Dict with shap_values, base_values, and explanations.
        """
        if not self._is_fitted:
            msg = "Explainer must be fitted before explaining"
            raise ValueError(msg)

        if self.shap is None:
            return {
                "shap_values": None,
                "base_values": None,
                "feature_names": self.feature_names,
                "warning": "SHAP not available",
            }

        if isinstance(instances, pd.DataFrame):
            data = instances.values
            if self.feature_names is None:
                self.feature_names = list(instances.columns)
        else:
            data = instances

        if max_samples is not None and len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]

        # Compute SHAP values
        if self._explainer is not None:
            shap_values = self._explainer.shap_values(data)
        else:
            shap_values = None

        # Base value (expected model output)
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
        else:
            base_value = None

        # Handle multi-output SHAP values
        if isinstance(shap_values, list):
            # Multi-class case
            shap_summary = {
                f"class_{i}": vals.tolist() if hasattr(vals, 'tolist') else vals
                for i, vals in enumerate(shap_values)
            }
        else:
            shap_summary = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values

        return {
            "shap_values": shap_summary,
            "base_values": base_value.tolist() if hasattr(base_value, 'tolist') else base_value,
            "feature_names": self.feature_names,
        }

    def get_feature_importance(
        self,
        instances: np.ndarray | pd.DataFrame,
        *,
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """Get global feature importance from SHAP values.

        Args:
            instances: Data to compute importance on.
            n_samples: Number of samples to use.

        Returns:
            DataFrame with feature names and importance scores.
        """
        explanation = self.explain(instances, max_samples=n_samples)

        shap_vals = explanation.get("shap_values")
        if shap_vals is None:
            return pd.DataFrame({"feature": self.feature_names, "importance": 0.0})

        # Compute mean absolute SHAP value per feature
        if isinstance(shap_vals, dict):
            # Multi-class: average across classes
            values = np.array([v for v in shap_vals.values()])
            importance = np.mean(np.abs(values), axis=(0, 1)).tolist()
        else:
            importance = np.mean(np.abs(shap_vals), axis=0).tolist()

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def get_local_explanation(
        self,
        instance: np.ndarray | pd.Series,
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Get local explanation for a single instance.

        Args:
            instance: Single data point to explain.
            top_k: Number of top features to return.

        Returns:
            Dict with top contributing features and directions.
        """
        if isinstance(instance, pd.Series):
            instance = instance.values

        instance = instance.reshape(1, -1)
        explanation = self.explain(instance)

        shap_vals = explanation.get("shap_values")
        if shap_vals is None:
            return {"warning": "SHAP not available"}

        # Handle different SHAP value formats
        if isinstance(shap_vals, dict):
            # Multi-class: use first class
            vals = np.array(list(shap_vals.values())[0])
        else:
            vals = np.array(shap_vals)

        # Flatten if needed
        if vals.ndim > 1:
            vals = vals[0]

        # Create feature contribution list
        contributions = []
        for i, (name, val) in enumerate(zip(self.feature_names, vals)):
            contributions.append({
                "feature": name,
                "shap_value": float(val),
                "direction": "positive" if val > 0 else "negative",
                "abs_contribution": float(abs(val)),
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

        return {
            "instance_id": "single",
            "base_value": explanation.get("base_values"),
            "top_features": contributions[:top_k],
            "all_features": contributions,
        }


class ModelCardGenerator:
    """Generate model cards in HTML and PDF formats.

    Model cards document:
    - Model details and architecture
    - Intended use cases
    - Training data and preprocessing
    - Performance metrics
    - Limitations and ethical considerations
    - SHAP-based feature importance
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        version: str = "1.0.0",
        author: str = "HealthRisk AI Team",
    ) -> None:
        """Initialize model card generator.

        Args:
            model_name: Name of the model.
            model_type: Type of model (e.g. 'NLP', 'GNN', 'Survival', 'Tabular').
            version: Model version.
            author: Model author/developer.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.author = author
        self.metadata: dict[str, Any] = {}
        self.performance_metrics: dict[str, float] = {}
        self.ethical_considerations: list[str] = []
        self.limitations: list[str] = []
        self.intended_use_cases: list[str] = []
        self.out_of_scope_use_cases: list[str] = []

    def set_metadata(
        self,
        training_data: str | None = None,
        training_date: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> ModelCardGenerator:
        """Set model metadata.

        Args:
            training_data: Description of training data.
            training_date: Date of training.
            hyperparameters: Model hyperparameters.
            model_config: Model configuration.

        Returns:
            Self for chaining.
        """
        self.metadata = {
            "training_data": training_data or "Not specified",
            "training_date": training_date or datetime.now().isoformat(),
            "hyperparameters": hyperparameters or {},
            "model_config": model_config or {},
        }
        return self

    def set_performance_metrics(
        self,
        metrics: dict[str, float],
    ) -> ModelCardGenerator:
        """Set model performance metrics.

        Args:
            metrics: Dict of metric name -> value.

        Returns:
            Self for chaining.
        """
        self.performance_metrics = metrics
        return self

    def add_ethical_consideration(self, consideration: str) -> ModelCardGenerator:
        """Add an ethical consideration.

        Args:
            consideration: Description of ethical consideration.

        Returns:
            Self for chaining.
        """
        self.ethical_considerations.append(consideration)
        return self

    def add_limitation(self, limitation: str) -> ModelCardGenerator:
        """Add a model limitation.

        Args:
            limitation: Description of limitation.

        Returns:
            Self for chaining.
        """
        self.limitations.append(limitation)
        return self

    def set_intended_use(
        self,
        use_cases: list[str],
        out_of_scope: list[str] | None = None,
    ) -> ModelCardGenerator:
        """Set intended and out-of-scope use cases.

        Args:
            use_cases: List of intended use cases.
            out_of_scope: List of out-of-scope use cases.

        Returns:
            Self for chaining.
        """
        self.intended_use_cases = use_cases
        self.out_of_scope_use_cases = out_of_scope or []
        return self

    def generate_html(
        self,
        feature_importance: pd.DataFrame | None = None,
        include_shap_summary: bool = False,
    ) -> str:
        """Generate HTML model card.

        Args:
            feature_importance: Optional feature importance DataFrame.
            include_shap_summary: Whether to include SHAP summary plot placeholder.

        Returns:
            HTML string.
        """
        # Build HTML content
        html_parts = [
            f"<h1>{self.model_name} - Model Card</h1>",
            f"<p><strong>Version:</strong> {self.version}</p>",
            f"<p><strong>Type:</strong> {self.model_type}</p>",
            f"<p><strong>Author:</strong> {self.author}</p>",
            f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
        ]

        # Metadata section
        html_parts.append("<h2>Model Details</h2>")
        html_parts.append("<table class='model-card-table'>")
        html_parts.append(f"<tr><td>Training Data</td><td>{self.metadata.get('training_data', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Training Date</td><td>{self.metadata.get('training_date', 'N/A')}</td></tr>")
        html_parts.append("</table>")

        if self.metadata.get('hyperparameters'):
            html_parts.append("<h3>Hyperparameters</h3>")
            html_parts.append("<pre>")
            html_parts.append(json.dumps(self.metadata['hyperparameters'], indent=2))
            html_parts.append("</pre>")

        # Performance section
        html_parts.append("<h2>Performance Metrics</h2>")
        if self.performance_metrics:
            html_parts.append("<table class='model-card-table'>")
            for metric, value in self.performance_metrics.items():
                html_parts.append(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>")
            html_parts.append("</table>")
        else:
            html_parts.append("<p>No performance metrics available.</p>")

        # Intended use
        html_parts.append("<h2>Intended Use</h2>")
        html_parts.append("<h3>Intended Use Cases</h3>")
        for use_case in self.intended_use_cases:
            html_parts.append(f"<li>{use_case}</li>")

        if self.out_of_scope_use_cases:
            html_parts.append("<h3>Out-of-Scope Use Cases</h3>")
            for use_case in self.out_of_scope_use_cases:
                html_parts.append(f"<li>{use_case}</li>")

        # Limitations
        html_parts.append("<h2>Limitations</h2>")
        for limitation in self.limitations:
            html_parts.append(f"<li>{limitation}</li>")

        # Ethical considerations
        html_parts.append("<h2>Ethical Considerations</h2>")
        for consideration in self.ethical_considerations:
            html_parts.append(f"<li>{concern}</li>")

        # Feature importance
        if feature_importance is not None and not feature_importance.empty:
            html_parts.append("<h2>Feature Importance</h2>")
            html_parts.append("<table class='model-card-table'>")
            html_parts.append("<tr><th>Feature</th><th>Importance</th></tr>")
            for _, row in feature_importance.head(20).iterrows():
                html_parts.append(f"<tr><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>")
            html_parts.append("</table>")

        if include_shap_summary:
            html_parts.append("<h2>SHAP Summary</h2>")
            html_parts.append("<p><em>SHAP summary plot would be inserted here.</em></p>")

        # CSS styling
        css = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h3 { color: #7f8c8d; }
            .model-card-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            .model-card-table th, .model-card-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            .model-card-table th { background-color: #3498db; color: white; }
            .model-card-table tr:nth-child(even) { background-color: #f2f2f2; }
            ul { line-height: 1.6; }
        </style>
        """

        return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{''.join(html_parts)}</body></html>"

    def save_html(self, path: str | Path) -> None:
        """Save model card as HTML file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.generate_html()
        path.write_text(html, encoding='utf-8')
        logger.info("Saved model card HTML to %s", path)

    def save_yaml(self, path: str | Path) -> None:
        """Save model card metadata as YAML.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        card_data = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "author": self.author,
            "generated_date": datetime.now().isoformat(),
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "intended_use_cases": self.intended_use_cases,
            "out_of_scope_use_cases": self.out_of_scope_use_cases,
            "limitations": self.limitations,
            "ethical_considerations": self.ethical_considerations,
        }

        path.write_text(yaml.dump(card_data), encoding='utf-8')
        logger.info("Saved model card YAML to %s", path)


class ExplainabilityModule:
    """High-level explainability interface for the HealthRisk AI platform.

    Provides unified access to:
    - SHAP explanations for all models
    - Model card generation
    - Feature importance aggregation
    """

    def __init__(self) -> None:
        self.explainers: dict[str, SHAPExplainer] = {}
        self.model_cards: dict[str, ModelCardGenerator] = {}

    def register_model(
        self,
        model_name: str,
        model: Any,
        feature_names: list[str] | None = None,
        model_type: str = "auto",
    ) -> ExplainabilityModule:
        """Register a model for explainability.

        Args:
            model_name: Name to register model under.
            model: Trained model instance.
            feature_names: Feature names for the model.
            model_type: Model type hint.

        Returns:
            Self for chaining.
        """
        explainer = SHAPExplainer(model, feature_names, model_type)
        self.explainers[model_name] = explainer

        card = ModelCardGenerator(model_name, model_type)
        self.model_cards[model_name] = card

        logger.info("Registered model for explainability: %s", model_name)
        return self

    def explain_model(
        self,
        model_name: str,
        instances: np.ndarray | pd.DataFrame,
        *,
        n_background_samples: int = 100,
    ) -> dict[str, Any]:
        """Get SHAP explanation for a model.

        Args:
            model_name: Registered model name.
            instances: Data to explain.
            n_background_samples: Background samples for SHAP.

        Returns:
            Explanation dict.
        """
        if model_name not in self.explainers:
            msg = f"Model not registered: {model_name}"
            raise ValueError(msg)

        explainer = self.explainers[model_name]

        # Fit explainer if not already fitted
        if not explainer._is_fitted:
            explainer.fit(instances, n_samples=n_background_samples)

        return explainer.explain(instances)

    def get_feature_importance(
        self,
        model_name: str,
        instances: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Get feature importance for a model.

        Args:
            model_name: Registered model name.
            instances: Data to compute importance on.

        Returns:
            DataFrame with feature importance.
        """
        if model_name not in self.explainers:
            msg = f"Model not registered: {model_name}"
            raise ValueError(msg)

        explainer = self.explainers[model_name]

        if not explainer._is_fitted:
            explainer.fit(instances)

        return explainer.get_feature_importance(instances)

    def generate_model_card(
        self,
        model_name: str,
        *,
        feature_importance: pd.DataFrame | None = None,
        output_path: str | None = None,
    ) -> str:
        """Generate model card for a model.

        Args:
            model_name: Registered model name.
            feature_importance: Optional feature importance.
            output_path: Optional path to save HTML card.

        Returns:
            HTML string of model card.
        """
        if model_name not in self.model_cards:
            msg = f"Model not registered: {model_name}"
            raise ValueError(msg)

        card = self.model_cards[model_name]

        if feature_importance is not None:
            html = card.generate_html(feature_importance=feature_importance)
        else:
            html = card.generate_html()

        if output_path:
            card.save_html(output_path)

        return html

    def save_all_model_cards(self, output_dir: str | Path) -> None:
        """Save model cards for all registered models.

        Args:
            output_dir: Directory to save cards.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, card in self.model_cards.items():
            model_dir = output_path / model_name
            card.save_html(model_dir / "model_card.html")
            card.save_yaml(model_dir / "model_card.yaml")

        logger.info("Saved all model cards to %s", output_path)
