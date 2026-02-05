"""
Cluster Operator.

Groups similar data points using various clustering algorithms:
- K-Means: Centroid-based clustering
- DBSCAN: Density-based clustering
- HDBSCAN: Hierarchical density-based clustering
- Agglomerative: Hierarchical clustering
"""

from enum import Enum
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class ClusterMethod(str, Enum):
    """Supported clustering methods."""
    
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"
    MINIBATCH_KMEANS = "minibatch_kmeans"


class LinkageType(str, Enum):
    """Linkage types for hierarchical clustering."""
    
    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"
    SINGLE = "single"


class ClusterOperatorConfig(BaseModel):
    """Configuration for ClusterOperator."""
    
    # Feature columns to cluster on
    feature_columns: list[str] = Field(
        ...,
        min_length=1,
        description="Columns to use as features for clustering"
    )
    
    # Method selection
    method: ClusterMethod = Field(
        default=ClusterMethod.KMEANS,
        description="Clustering algorithm to use"
    )
    
    # K-Means / MiniBatch K-Means settings
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Number of clusters (for K-Means, Agglomerative)"
    )
    max_iter: int = Field(
        default=300,
        ge=10,
        description="Maximum iterations for K-Means"
    )
    n_init: int = Field(
        default=10,
        ge=1,
        description="Number of initializations for K-Means"
    )
    
    # DBSCAN settings
    eps: float = Field(
        default=0.5,
        gt=0,
        description="Maximum distance between samples (DBSCAN)"
    )
    min_samples: int = Field(
        default=5,
        ge=1,
        description="Minimum samples in neighborhood (DBSCAN/HDBSCAN)"
    )
    
    # HDBSCAN settings
    min_cluster_size: int = Field(
        default=5,
        ge=2,
        description="Minimum cluster size (HDBSCAN)"
    )
    cluster_selection_epsilon: float = Field(
        default=0.0,
        ge=0,
        description="Distance threshold for cluster selection (HDBSCAN)"
    )
    
    # Agglomerative settings
    linkage: LinkageType = Field(
        default=LinkageType.WARD,
        description="Linkage criterion for hierarchical clustering"
    )
    
    # Feature preprocessing
    normalize: bool = Field(
        default=True,
        description="Normalize features before clustering"
    )
    
    # Output settings
    cluster_column: str = Field(
        default="cluster",
        description="Name of the cluster label column"
    )
    include_distance: bool = Field(
        default=False,
        description="Include distance to cluster center (K-Means only)"
    )
    include_probability: bool = Field(
        default=False,
        description="Include cluster probability (HDBSCAN only)"
    )
    
    # Reproducibility
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    @field_validator("feature_columns")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one feature column required")
        return [col.strip() for col in v if col.strip()]
    
    @field_validator("cluster_column")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


@register_operator("ClusterOperator")
class ClusterOperator(Operator):
    """
    Groups data points into clusters using various algorithms.
    
    Supported Methods:
    - kmeans: Fast, requires specifying n_clusters
    - dbscan: Density-based, finds arbitrary shapes, handles noise
    - hdbscan: Hierarchical DBSCAN, auto-selects clusters
    - agglomerative: Hierarchical, various linkage options
    - minibatch_kmeans: Faster K-Means for large datasets
    
    Example Config:
        {
            "feature_columns": ["lat", "lon", "speed"],
            "method": "dbscan",
            "eps": 0.5,
            "min_samples": 5,
            "cluster_column": "cluster_id"
        }
    """
    
    name = "ClusterOperator"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="*", dtype=DataType.FLOAT, required=True),
        ],
        description="DataFrame with numeric feature columns"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="cluster", dtype=DataType.INTEGER, required=True),
        ],
        description="Original data with cluster labels"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[ClusterOperatorConfig] = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = ClusterOperatorConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid ClusterOperator config: {e}")
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Cluster the data points.
        
        Args:
            data: Input DataFrame with feature columns
            
        Returns:
            OperatorResult with cluster labels
        """
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        # Validate feature columns exist
        missing_cols = [c for c in config.feature_columns if c not in data.columns]
        if missing_cols:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Missing feature columns: {missing_cols}"
            )
        
        try:
            # Extract features
            features = data.select(config.feature_columns).to_numpy()
            
            # Normalize if requested
            if config.normalize:
                features = self._normalize_features(features)
            
            # Run clustering
            method_handlers = {
                ClusterMethod.KMEANS: self._cluster_kmeans,
                ClusterMethod.MINIBATCH_KMEANS: self._cluster_minibatch_kmeans,
                ClusterMethod.DBSCAN: self._cluster_dbscan,
                ClusterMethod.HDBSCAN: self._cluster_hdbscan,
                ClusterMethod.AGGLOMERATIVE: self._cluster_agglomerative,
            }
            
            handler = method_handlers.get(config.method)
            if not handler:
                raise ValueError(f"Unknown clustering method: {config.method}")
            
            labels, extras = handler(features, config)
            
            # Add cluster labels to data
            result = data.with_columns([
                pl.Series(config.cluster_column, labels),
            ])
            
            # Add optional columns
            if config.include_distance and "distances" in extras:
                result = result.with_columns([
                    pl.Series(f"{config.cluster_column}_distance", extras["distances"]),
                ])
            
            if config.include_probability and "probabilities" in extras:
                result = result.with_columns([
                    pl.Series(f"{config.cluster_column}_probability", extras["probabilities"]),
                ])
            
            # Calculate cluster statistics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = sum(1 for l in labels if l == -1)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "method": config.method.value,
                    "n_clusters": n_clusters,
                    "n_noise_points": n_noise,
                    "n_samples": len(labels),
                    "feature_columns": config.feature_columns,
                    **{k: v for k, v in extras.items() if k not in ["distances", "probabilities"]},
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Clustering failed: {e}"
            )
    
    def _normalize_features(self, features) -> Any:
        """Normalize features using StandardScaler."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(features)
        except ImportError:
            # Manual normalization
            import numpy as np
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (features - mean) / std
    
    def _cluster_kmeans(
        self,
        features: Any,
        config: ClusterOperatorConfig
    ) -> tuple[list[int], dict]:
        """K-Means clustering."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for K-Means. "
                "Install with: pip install scikit-learn"
            )
        
        kmeans = KMeans(
            n_clusters=config.n_clusters,
            max_iter=config.max_iter,
            n_init=config.n_init,
            random_state=config.random_state,
        )
        
        labels = kmeans.fit_predict(features)
        
        extras = {
            "inertia": kmeans.inertia_,
            "n_iter": kmeans.n_iter_,
        }
        
        if config.include_distance:
            # Distance to assigned cluster center
            import numpy as np
            distances = np.min(kmeans.transform(features), axis=1)
            extras["distances"] = distances.tolist()
        
        return labels.tolist(), extras
    
    def _cluster_minibatch_kmeans(
        self,
        features: Any,
        config: ClusterOperatorConfig
    ) -> tuple[list[int], dict]:
        """MiniBatch K-Means for large datasets."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for MiniBatch K-Means. "
                "Install with: pip install scikit-learn"
            )
        
        kmeans = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            max_iter=config.max_iter,
            n_init=config.n_init,
            random_state=config.random_state,
            batch_size=min(1024, len(features)),
        )
        
        labels = kmeans.fit_predict(features)
        
        extras = {
            "inertia": kmeans.inertia_,
            "n_iter": kmeans.n_iter_,
        }
        
        if config.include_distance:
            import numpy as np
            distances = np.min(kmeans.transform(features), axis=1)
            extras["distances"] = distances.tolist()
        
        return labels.tolist(), extras
    
    def _cluster_dbscan(
        self,
        features: Any,
        config: ClusterOperatorConfig
    ) -> tuple[list[int], dict]:
        """DBSCAN density-based clustering."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError(
                "scikit-learn is required for DBSCAN. "
                "Install with: pip install scikit-learn"
            )
        
        dbscan = DBSCAN(
            eps=config.eps,
            min_samples=config.min_samples,
            n_jobs=-1,
        )
        
        labels = dbscan.fit_predict(features)
        
        extras = {
            "core_sample_indices": len(dbscan.core_sample_indices_),
        }
        
        return labels.tolist(), extras
    
    def _cluster_hdbscan(
        self,
        features: Any,
        config: ClusterOperatorConfig
    ) -> tuple[list[int], dict]:
        """HDBSCAN hierarchical density-based clustering."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
        )
        
        labels = clusterer.fit_predict(features)
        
        extras = {}
        
        if config.include_probability:
            extras["probabilities"] = clusterer.probabilities_.tolist()
        
        return labels.tolist(), extras
    
    def _cluster_agglomerative(
        self,
        features: Any,
        config: ClusterOperatorConfig
    ) -> tuple[list[int], dict]:
        """Agglomerative hierarchical clustering."""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            raise ImportError(
                "scikit-learn is required for Agglomerative clustering. "
                "Install with: pip install scikit-learn"
            )
        
        agg = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            linkage=config.linkage.value,
        )
        
        labels = agg.fit_predict(features)
        
        extras = {
            "n_leaves": agg.n_leaves_,
            "n_connected_components": agg.n_connected_components_,
        }
        
        return labels.tolist(), extras
