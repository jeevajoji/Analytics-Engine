"""
Classification Operator.

Runs classification ML models to predict categories/labels:
- Supports scikit-learn models
- Supports ONNX models for portable inference
- Supports pre-trained model loading
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
from pydantic import BaseModel, Field, field_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class ModelType(str, Enum):
    """Supported model types."""
    
    SKLEARN = "sklearn"  # scikit-learn model
    ONNX = "onnx"  # ONNX model
    LIGHTGBM = "lightgbm"  # LightGBM model
    XGBOOST = "xgboost"  # XGBoost model


class ClassificationOperatorConfig(BaseModel):
    """Configuration for ClassificationOperator."""
    
    # Model configuration
    model_path: Optional[str] = Field(
        default=None,
        description="Path to saved model file"
    )
    model_type: ModelType = Field(
        default=ModelType.SKLEARN,
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
    include_probabilities: bool = Field(
        default=False,
        description="Include class probabilities in output"
    )
    probability_prefix: str = Field(
        default="prob_",
        description="Prefix for probability columns"
    )
    
    # Label mapping
    label_mapping: Optional[dict[int, str]] = Field(
        default=None,
        description="Mapping from numeric labels to string labels"
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


@register_operator("ClassificationOperator")
class ClassificationOperator(Operator):
    """
    Runs classification models to predict categories/labels.
    
    Supported Model Types:
    - sklearn: scikit-learn models (joblib/pickle)
    - onnx: ONNX format models
    - lightgbm: LightGBM models
    - xgboost: XGBoost models
    
    Features:
    - Loads pre-trained models from disk
    - Optional probability outputs
    - Label mapping support
    - Feature normalization
    
    Example Config:
        {
            "model_path": "/models/classifier.joblib",
            "model_type": "sklearn",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "prediction_column": "predicted_class",
            "include_probabilities": true
        }
    """
    
    name = "ClassificationOperator"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="*", dtype=DataType.FLOAT, required=True),
        ],
        description="DataFrame with feature columns"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="prediction", dtype=DataType.STRING, required=True),
        ],
        description="Original data with predictions"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[ClassificationOperatorConfig] = None
        self._model: Any = None
        self._scaler: Any = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = ClassificationOperatorConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid ClassificationOperator config: {e}")
    
    def load_model(self, model: Any = None) -> None:
        """
        Load the classification model.
        
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
        
        if config.model_type == ModelType.SKLEARN:
            self._model = self._load_sklearn(model_path)
        elif config.model_type == ModelType.ONNX:
            self._model = self._load_onnx(model_path)
        elif config.model_type == ModelType.LIGHTGBM:
            self._model = self._load_lightgbm(model_path)
        elif config.model_type == ModelType.XGBOOST:
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
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            return model
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoost models. "
                "Install with: pip install xgboost"
            )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Run classification on the input data.
        
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
            predictions, probabilities = self._predict(X, config)
            
            # Apply label mapping
            if config.label_mapping:
                predictions = [
                    config.label_mapping.get(p, str(p))
                    for p in predictions
                ]
            
            # Build result
            result = data.with_columns([
                pl.Series(config.prediction_column, predictions),
            ])
            
            # Add probabilities if requested
            if config.include_probabilities and probabilities is not None:
                n_classes = probabilities.shape[1]
                for i in range(n_classes):
                    class_label = config.label_mapping.get(i, str(i)) if config.label_mapping else str(i)
                    col_name = f"{config.probability_prefix}{class_label}"
                    result = result.with_columns([
                        pl.Series(col_name, probabilities[:, i]),
                    ])
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "model_type": config.model_type.value,
                    "n_samples": len(predictions),
                    "n_features": len(config.feature_columns),
                    "unique_predictions": len(set(predictions)),
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Classification failed: {e}"
            )
    
    def _handle_missing(
        self,
        data: pl.DataFrame,
        config: ClassificationOperatorConfig
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
            # Manual normalization
            import numpy as np
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1
            return (X - mean) / std
    
    def _predict(
        self,
        X: Any,
        config: ClassificationOperatorConfig
    ) -> tuple[list, Optional[Any]]:
        """Run prediction using the loaded model."""
        
        import numpy as np
        
        if config.model_type == ModelType.ONNX:
            # ONNX inference
            input_name = config.onnx_input_name
            inputs = {input_name: X.astype(np.float32)}
            outputs = self._model.run(None, inputs)
            
            predictions = outputs[0].tolist()
            probabilities = outputs[1] if len(outputs) > 1 else None
            
        elif config.model_type == ModelType.LIGHTGBM:
            # LightGBM prediction
            probabilities = self._model.predict(X)
            if len(probabilities.shape) == 1:
                # Binary classification
                predictions = (probabilities > 0.5).astype(int).tolist()
                probabilities = np.column_stack([1 - probabilities, probabilities])
            else:
                predictions = np.argmax(probabilities, axis=1).tolist()
            
        else:
            # sklearn / XGBoost
            predictions = self._model.predict(X).tolist()
            
            if config.include_probabilities and hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba(X)
            else:
                probabilities = None
        
        return predictions, probabilities


def classify(
    data: pl.DataFrame,
    model: Any,
    feature_columns: list[str],
    prediction_column: str = "prediction",
    include_probabilities: bool = False,
    **kwargs
) -> pl.DataFrame:
    """
    Convenience function to run classification.
    
    Args:
        data: DataFrame with features
        model: Pre-loaded model object
        feature_columns: List of feature column names
        prediction_column: Name for prediction output
        include_probabilities: Include class probabilities
        **kwargs: Additional config options
        
    Returns:
        DataFrame with predictions
    """
    config = OperatorConfig(
        params={
            "feature_columns": feature_columns,
            "prediction_column": prediction_column,
            "include_probabilities": include_probabilities,
            **kwargs,
        }
    )
    
    operator = ClassificationOperator(config)
    operator.load_model(model)
    result = operator.process(data)
    
    if not result.success:
        raise ValueError(result.error)
    
    return result.data
