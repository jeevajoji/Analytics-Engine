"""
Anomaly Detector Operator.

Detects anomalies using statistical and ML methods:
- Z-Score: Standard deviation based
- IQR: Interquartile range based
- Isolation Forest: ML-based outlier detection
- MAD: Median Absolute Deviation
"""

from enum import Enum
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, field_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class AnomalyMethod(str, Enum):
    """Supported anomaly detection methods."""
    
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    MAD = "mad"  # Median Absolute Deviation
    MODIFIED_ZSCORE = "modified_zscore"


class AnomalyDetectorConfig(BaseModel):
    """Configuration for AnomalyDetector operator."""
    
    column: str = Field(..., description="Column to analyze for anomalies")
    method: AnomalyMethod = Field(
        default=AnomalyMethod.ZSCORE,
        description="Anomaly detection method"
    )
    threshold: float = Field(
        default=3.0,
        ge=0.1,
        description="Threshold for anomaly detection (method-specific)"
    )
    
    # IQR-specific
    iqr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        description="IQR multiplier (1.5 = outliers, 3.0 = extreme outliers)"
    )
    
    # Isolation Forest specific
    contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Expected proportion of outliers (for Isolation Forest)"
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of trees in Isolation Forest"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    # Output options
    output_column: str = Field(
        default="is_anomaly",
        description="Name of the boolean anomaly flag column"
    )
    score_column: Optional[str] = Field(
        default="anomaly_score",
        description="Name of the anomaly score column (None to skip)"
    )
    include_bounds: bool = Field(
        default=False,
        description="Include upper/lower bound columns"
    )
    
    # Grouping (detect anomalies within groups)
    group_by: Optional[list[str]] = Field(
        default=None,
        description="Columns to group by before detecting anomalies"
    )
    
    @field_validator("column", "output_column")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


@register_operator("AnomalyDetector")
class AnomalyDetector(Operator):
    """
    Detects anomalies in data using various statistical and ML methods.
    
    Supported Methods:
    - zscore: Points beyond N standard deviations from mean
    - iqr: Points outside Q1 - k*IQR to Q3 + k*IQR range
    - isolation_forest: ML-based isolation scoring
    - mad: Median Absolute Deviation based
    - modified_zscore: Uses median instead of mean (robust)
    
    Example Config:
        {
            "column": "temperature",
            "method": "zscore",
            "threshold": 3.0,
            "output_column": "is_anomaly",
            "score_column": "anomaly_score"
        }
    """
    
    name = "AnomalyDetector"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="*", dtype=DataType.FLOAT, required=True),
        ],
        description="DataFrame with numeric column to analyze"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="is_anomaly", dtype=DataType.BOOLEAN, required=True),
            SchemaField(name="anomaly_score", dtype=DataType.FLOAT, required=False),
        ],
        description="Original data with anomaly flag and optional score"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[AnomalyDetectorConfig] = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = AnomalyDetectorConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid AnomalyDetector config: {e}")
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Detect anomalies in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            OperatorResult with anomaly flags and scores
        """
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        # Validate column exists
        if config.column not in data.columns:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Column '{config.column}' not found in data"
            )
        
        # Check column is numeric
        col_dtype = data[config.column].dtype
        if col_dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Column '{config.column}' must be numeric, got {col_dtype}"
            )
        
        try:
            if config.group_by:
                # Detect anomalies within each group
                result = self._detect_grouped(data, config)
            else:
                # Detect anomalies globally
                result = self._detect_global(data, config)
            
            # Count anomalies
            anomaly_count = result[config.output_column].sum()
            total_count = len(result)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "method": config.method.value,
                    "threshold": config.threshold,
                    "column": config.column,
                    "anomaly_count": anomaly_count,
                    "total_count": total_count,
                    "anomaly_rate": anomaly_count / total_count if total_count > 0 else 0,
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Anomaly detection failed: {e}"
            )
    
    def _detect_global(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Detect anomalies across the entire dataset."""
        
        method_handlers = {
            AnomalyMethod.ZSCORE: self._detect_zscore,
            AnomalyMethod.IQR: self._detect_iqr,
            AnomalyMethod.ISOLATION_FOREST: self._detect_isolation_forest,
            AnomalyMethod.MAD: self._detect_mad,
            AnomalyMethod.MODIFIED_ZSCORE: self._detect_modified_zscore,
        }
        
        handler = method_handlers.get(config.method)
        if not handler:
            raise ValueError(f"Unknown method: {config.method}")
        
        return handler(data, config)
    
    def _detect_grouped(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Detect anomalies within each group."""
        
        # Add row index for rejoining
        data = data.with_row_index("__row_idx__")
        
        results = []
        for group_data in data.group_by(config.group_by):
            group_df = group_data[1] if isinstance(group_data, tuple) else group_data
            
            # Create a config without grouping for per-group detection
            group_config = AnomalyDetectorConfig(
                column=config.column,
                method=config.method,
                threshold=config.threshold,
                iqr_multiplier=config.iqr_multiplier,
                contamination=config.contamination,
                n_estimators=config.n_estimators,
                random_state=config.random_state,
                output_column=config.output_column,
                score_column=config.score_column,
                include_bounds=config.include_bounds,
                group_by=None,
            )
            
            group_result = self._detect_global(group_df, group_config)
            results.append(group_result)
        
        # Combine and sort by original order
        combined = pl.concat(results)
        combined = combined.sort("__row_idx__").drop("__row_idx__")
        
        return combined
    
    def _detect_zscore(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Z-Score based anomaly detection."""
        
        col = config.column
        values = data[col]
        
        mean = values.mean()
        std = values.std()
        
        # Avoid division by zero
        if std == 0 or std is None:
            # All values are the same - no anomalies
            result = data.with_columns([
                pl.lit(False).alias(config.output_column),
            ])
            if config.score_column:
                result = result.with_columns([
                    pl.lit(0.0).alias(config.score_column),
                ])
            return result
        
        # Calculate Z-scores
        z_scores = ((pl.col(col) - mean) / std).abs()
        
        result = data.with_columns([
            (z_scores > config.threshold).alias(config.output_column),
        ])
        
        if config.score_column:
            result = result.with_columns([
                z_scores.alias(config.score_column),
            ])
        
        if config.include_bounds:
            lower_bound = mean - config.threshold * std
            upper_bound = mean + config.threshold * std
            result = result.with_columns([
                pl.lit(lower_bound).alias(f"{col}_lower_bound"),
                pl.lit(upper_bound).alias(f"{col}_upper_bound"),
            ])
        
        return result
    
    def _detect_iqr(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """IQR (Interquartile Range) based anomaly detection."""
        
        col = config.column
        values = data[col]
        
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        k = config.iqr_multiplier
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        # Distance from nearest bound (normalized by IQR)
        score_expr = pl.when(pl.col(col) < lower_bound).then(
            (lower_bound - pl.col(col)) / iqr if iqr > 0 else 0
        ).when(pl.col(col) > upper_bound).then(
            (pl.col(col) - upper_bound) / iqr if iqr > 0 else 0
        ).otherwise(0.0)
        
        result = data.with_columns([
            ((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).alias(config.output_column),
        ])
        
        if config.score_column:
            result = result.with_columns([
                score_expr.alias(config.score_column),
            ])
        
        if config.include_bounds:
            result = result.with_columns([
                pl.lit(lower_bound).alias(f"{col}_lower_bound"),
                pl.lit(upper_bound).alias(f"{col}_upper_bound"),
            ])
        
        return result
    
    def _detect_mad(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Median Absolute Deviation based anomaly detection."""
        
        col = config.column
        values = data[col]
        
        median = values.median()
        
        # Calculate MAD: median(|x - median(x)|)
        abs_deviations = (values - median).abs()
        mad = abs_deviations.median()
        
        # Consistency constant for normal distribution
        k = 1.4826
        
        if mad == 0 or mad is None:
            result = data.with_columns([
                pl.lit(False).alias(config.output_column),
            ])
            if config.score_column:
                result = result.with_columns([
                    pl.lit(0.0).alias(config.score_column),
                ])
            return result
        
        # Modified Z-score using MAD
        mad_scores = (pl.col(col) - median).abs() / (k * mad)
        
        result = data.with_columns([
            (mad_scores > config.threshold).alias(config.output_column),
        ])
        
        if config.score_column:
            result = result.with_columns([
                mad_scores.alias(config.score_column),
            ])
        
        if config.include_bounds:
            lower_bound = median - config.threshold * k * mad
            upper_bound = median + config.threshold * k * mad
            result = result.with_columns([
                pl.lit(lower_bound).alias(f"{col}_lower_bound"),
                pl.lit(upper_bound).alias(f"{col}_upper_bound"),
            ])
        
        return result
    
    def _detect_modified_zscore(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Modified Z-Score using median (more robust than standard Z-score)."""
        
        col = config.column
        values = data[col]
        
        median = values.median()
        
        # Use MAD for scale estimation
        abs_deviations = (values - median).abs()
        mad = abs_deviations.median()
        
        # Consistency constant
        k = 1.4826
        
        if mad == 0 or mad is None:
            result = data.with_columns([
                pl.lit(False).alias(config.output_column),
            ])
            if config.score_column:
                result = result.with_columns([
                    pl.lit(0.0).alias(config.score_column),
                ])
            return result
        
        # Modified Z-score: 0.6745 * (x - median) / MAD
        modified_z = (0.6745 * (pl.col(col) - median) / mad).abs()
        
        result = data.with_columns([
            (modified_z > config.threshold).alias(config.output_column),
        ])
        
        if config.score_column:
            result = result.with_columns([
                modified_z.alias(config.score_column),
            ])
        
        return result
    
    def _detect_isolation_forest(
        self,
        data: pl.DataFrame,
        config: AnomalyDetectorConfig
    ) -> pl.DataFrame:
        """Isolation Forest based anomaly detection."""
        
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "scikit-learn is required for Isolation Forest. "
                "Install with: pip install scikit-learn"
            )
        
        col = config.column
        values = data[col].to_numpy().reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=-1,
        )
        
        # Predict: -1 for anomalies, 1 for normal
        predictions = iso_forest.fit_predict(values)
        
        # Get anomaly scores (lower = more anomalous)
        scores = iso_forest.decision_function(values)
        
        # Normalize scores to [0, 1] where higher = more anomalous
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = 1 - (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = [0.0] * len(scores)
        
        result = data.with_columns([
            pl.Series(config.output_column, predictions == -1),
        ])
        
        if config.score_column:
            result = result.with_columns([
                pl.Series(config.score_column, normalized_scores),
            ])
        
        return result
