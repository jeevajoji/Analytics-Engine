"""
Regression Operator.

Runs regression ML models to predict continuous values:
- Supports scikit-learn models
- Supports ONNX models for portable inference
- Supports pre-trained model loading
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, field_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class RegressionModelType(str, Enum):
    """Supported regression model types."""
    
    SKLEARN = "sklearn"  # scikit-learn model
    ONNX = "onnx"  # ONNX model
    LIGHTGBM = "lightgbm"  # LightGBM regressor
    XGBOOST = "xgboost"  # XGBoost regressor


class RegressionOperatorConfig(BaseModel):
    """Configuration for RegressionOperator."""
    
    # Model configuration
    model_path: Optional[str] = Field(
        default=None,
        description="Path to saved model file"
    )
    model_type: RegressionModelType = Field(
        default=RegressionModelType.SKLEARN,
        description="Type of model to load"
    )
    
    # Feature columns
    feature_columns: list[str] = Field(
        ...,
        min_length=1,
        description="Columns to use as input features"
    )
    
    # Output settings
    prediction_column: str = Field(
        default="prediction",
        description="Name of the prediction output column"
    )
    include_confidence: bool = Field(
        default=False,
        description="Include prediction intervals (if model supports)"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )
    
    # Multi-output support
    target_names: Optional[list[str]] = Field(
        default=None,
        description="Names for multiple output targets"
    )
    
    # Preprocessing
    normalize_features: bool = Field(
        default=False,
        description="Normalize features before prediction"
    )
    handle_missing: str = Field(
        default="error",
        description="How to handle missing values: 'error', 'drop', 'fill_zero', 'fill_mean'"
    )
    
    # Post-processing
    clip_min: Optional[float] = Field(
        default=None,
        description="Minimum value to clip predictions"
    )
    clip_max: Optional[float] = Field(
        default=None,
        description="Maximum value to clip predictions"
    )
    round_decimals: Optional[int] = Field(
        default=None,
        description="Round predictions to N decimal places"
    )
    
    # ONNX specific
    onnx_input_name: str = Field(
        default="input",
        description="Input tensor name for ONNX model"
    )
    
    @field_validator("feature_columns")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one feature column required")
        return [col.strip() for col in v if col.strip()]
    
    @field_validator("prediction_column")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


@register_operator("RegressionOperator")
class RegressionOperator(Operator):
    """
    Runs regression models to predict continuous values.
    
    Supported Model Types:
    - sklearn: scikit-learn models (joblib/pickle)
    - onnx: ONNX format models
    - lightgbm: LightGBM regressors
    - xgboost: XGBoost regressors
    
    Features:
    - Pre-trained model loading
    - Multi-output regression support
    - Prediction clipping and rounding
    - Feature normalization
    
    Example Config:
        {
            "model_path": "/models/regressor.joblib",
            "model_type": "sklearn",
            "feature_columns": ["feature1", "feature2"],
            "prediction_column": "predicted_value",
            "clip_min": 0,
            "clip_max": 100
        }
    """
    
    name = "RegressionOperator"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="*", dtype=DataType.FLOAT, required=True),
        ],
        description="DataFrame with feature columns"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="prediction", dtype=DataType.FLOAT, required=True),
        ],
        description="Original data with predictions"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[RegressionOperatorConfig] = None
        self._model: Any = None
        self._scaler: Any = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = RegressionOperatorConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid RegressionOperator config: {e}")
    
    def load_model(self, model: Any = None) -> None:
        """
        Load the regression model.
        
        Args:
            model: Pre-loaded model object (optional)
        """
        if model is not None:
            self._model = model
            return
        
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        if not config.model_path:
            raise ValueError("No model path specified and no model provided")
        
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if config.model_type == RegressionModelType.SKLEARN:
            self._model = self._load_sklearn(model_path)
        elif config.model_type == RegressionModelType.ONNX:
            self._model = self._load_onnx(model_path)
        elif config.model_type == RegressionModelType.LIGHTGBM:
            self._model = self._load_lightgbm(model_path)
        elif config.model_type == RegressionModelType.XGBOOST:
            self._model = self._load_xgboost(model_path)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def _load_sklearn(self, path: Path) -> Any:
        """Load scikit-learn model."""
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def _load_onnx(self, path: Path) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(str(path))
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime"
            )
    
    def _load_lightgbm(self, path: Path) -> Any:
        """Load LightGBM model."""
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        except ImportError:
            raise ImportError(
                "lightgbm is required for LightGBM models. "
                "Install with: pip install lightgbm"
            )
    
    def _load_xgboost(self, path: Path) -> Any:
        """Load XGBoost model."""
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            return model
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoost models. "
                "Install with: pip install xgboost"
            )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Run regression on the input data.
        
        Args:
            data: Input DataFrame with feature columns
            
        Returns:
            OperatorResult with predictions
        """
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        # Validate feature columns
        missing = [c for c in config.feature_columns if c not in data.columns]
        if missing:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Missing feature columns: {missing}"
            )
        
        # Load model if not already loaded
        if self._model is None:
            try:
                self.load_model()
            except Exception as e:
                return OperatorResult(
                    success=False,
                    data=data,
                    error=f"Failed to load model: {e}"
                )
        
        try:
            # Extract features
            features = data.select(config.feature_columns)
            
            # Handle missing values
            features = self._handle_missing(features, config)
            
            # Convert to numpy
            X = features.to_numpy()
            
            # Normalize if requested
            if config.normalize_features:
                X = self._normalize(X)
            
            # Run prediction
            predictions = self._predict(X, config)
            
            # Post-process predictions
            predictions = self._post_process(predictions, config)
            
            # Build result
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                # Single output
                preds = predictions.flatten().tolist()
                result = data.with_columns([
                    pl.Series(config.prediction_column, preds),
                ])
            else:
                # Multi-output
                result = data
                n_outputs = predictions.shape[1]
                target_names = config.target_names or [
                    f"{config.prediction_column}_{i}" for i in range(n_outputs)
                ]
                for i, name in enumerate(target_names[:n_outputs]):
                    result = result.with_columns([
                        pl.Series(name, predictions[:, i].tolist()),
                    ])
            
            # Calculate prediction stats
            import numpy as np
            pred_flat = predictions.flatten()
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "model_type": config.model_type.value,
                    "n_samples": len(pred_flat) // (predictions.shape[1] if len(predictions.shape) > 1 else 1),
                    "n_features": len(config.feature_columns),
                    "prediction_mean": float(np.mean(pred_flat)),
                    "prediction_std": float(np.std(pred_flat)),
                    "prediction_min": float(np.min(pred_flat)),
                    "prediction_max": float(np.max(pred_flat)),
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Regression failed: {e}"
            )
    
    def _handle_missing(
        self,
        data: pl.DataFrame,
        config: RegressionOperatorConfig
    ) -> pl.DataFrame:
        """Handle missing values in features."""
        
        if config.handle_missing == "error":
            if data.null_count().sum_horizontal()[0] > 0:
                raise ValueError("Missing values found in features")
            return data
        
        elif config.handle_missing == "drop":
            return data.drop_nulls()
        
        elif config.handle_missing == "fill_zero":
            return data.fill_null(0)
        
        elif config.handle_missing == "fill_mean":
            return data.fill_null(strategy="mean")
        
        return data
    
    def _normalize(self, X: Any) -> Any:
        """Normalize features using StandardScaler."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            if self._scaler is None:
                self._scaler = StandardScaler()
                return self._scaler.fit_transform(X)
            return self._scaler.transform(X)
        except ImportError:
            import numpy as np
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1
            return (X - mean) / std
    
    def _predict(self, X: Any, config: RegressionOperatorConfig) -> Any:
        """Run prediction using the loaded model."""
        
        import numpy as np
        
        if config.model_type == RegressionModelType.ONNX:
            input_name = config.onnx_input_name
            inputs = {input_name: X.astype(np.float32)}
            outputs = self._model.run(None, inputs)
            return np.array(outputs[0])
        
        elif config.model_type == RegressionModelType.LIGHTGBM:
            return self._model.predict(X)
        
        else:
            # sklearn / XGBoost
            return np.array(self._model.predict(X))
    
    def _post_process(self, predictions: Any, config: RegressionOperatorConfig) -> Any:
        """Apply post-processing to predictions."""
        
        import numpy as np
        
        # Clip values
        if config.clip_min is not None or config.clip_max is not None:
            predictions = np.clip(
                predictions,
                config.clip_min if config.clip_min is not None else -np.inf,
                config.clip_max if config.clip_max is not None else np.inf
            )
        
        # Round values
        if config.round_decimals is not None:
            predictions = np.round(predictions, config.round_decimals)
        
        return predictions


def regress(
    data: pl.DataFrame,
    model: Any,
    feature_columns: list[str],
    prediction_column: str = "prediction",
    **kwargs
) -> pl.DataFrame:
    """
    Convenience function to run regression.
    
    Args:
        data: DataFrame with features
        model: Pre-loaded model object
        feature_columns: List of feature column names
        prediction_column: Name for prediction output
        **kwargs: Additional config options
        
    Returns:
        DataFrame with predictions
    """
    config = OperatorConfig(
        params={
            "feature_columns": feature_columns,
            "prediction_column": prediction_column,
            **kwargs,
        }
    )
    
    operator = RegressionOperator(config)
    operator.load_model(model)
    result = operator.process(data)
    
    if not result.success:
        raise ValueError(result.error)
    
    return result.data
