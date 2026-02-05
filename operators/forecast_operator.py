"""
Forecast Operator.

Time-series forecasting using various methods:
- Moving Average: Simple/Exponential moving averages
- Exponential Smoothing: Single, Double, Triple (Holt-Winters)
- ARIMA: Autoregressive Integrated Moving Average
- Prophet: Facebook's forecasting library (optional)
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class ForecastMethod(str, Enum):
    """Supported forecasting methods."""
    
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DOUBLE_EXPONENTIAL = "double_exponential"  # Holt's method (trend)
    TRIPLE_EXPONENTIAL = "triple_exponential"  # Holt-Winters (trend + seasonality)
    ARIMA = "arima"
    PROPHET = "prophet"
    LINEAR_TREND = "linear_trend"


class SeasonalityType(str, Enum):
    """Seasonality type for Holt-Winters."""
    
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class ForecastOperatorConfig(BaseModel):
    """Configuration for ForecastOperator."""
    
    # Column configuration
    value_column: str = Field(..., description="Column containing values to forecast")
    timestamp_column: str = Field(
        default="timestamp",
        description="Column containing timestamps"
    )
    
    # Forecast settings
    method: ForecastMethod = Field(
        default=ForecastMethod.EXPONENTIAL_SMOOTHING,
        description="Forecasting method to use"
    )
    horizon: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of periods to forecast"
    )
    
    # Moving Average settings
    window_size: int = Field(
        default=7,
        ge=2,
        description="Window size for moving average"
    )
    
    # Exponential Smoothing settings
    alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for level (0-1)"
    )
    beta: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for trend (Double/Triple exponential)"
    )
    gamma: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for seasonality (Triple exponential)"
    )
    seasonal_period: int = Field(
        default=7,
        ge=2,
        description="Length of seasonal cycle"
    )
    seasonality_type: SeasonalityType = Field(
        default=SeasonalityType.ADDITIVE,
        description="Type of seasonality (additive or multiplicative)"
    )
    
    # ARIMA settings
    arima_order: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="ARIMA (p, d, q) order"
    )
    seasonal_order: Optional[tuple[int, int, int, int]] = Field(
        default=None,
        description="Seasonal ARIMA (P, D, Q, s) order"
    )
    
    # Output settings
    forecast_column: str = Field(
        default="forecast",
        description="Name of forecast output column"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence interval columns"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )
    include_fitted: bool = Field(
        default=False,
        description="Include fitted values for historical data"
    )
    
    # Grouping
    group_by: Optional[list[str]] = Field(
        default=None,
        description="Columns to group by (forecast per group)"
    )
    
    @field_validator("value_column", "timestamp_column", "forecast_column")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_method_params(self):
        """Validate method-specific parameters."""
        if self.method == ForecastMethod.TRIPLE_EXPONENTIAL:
            if self.gamma is None:
                self.gamma = 0.1
        return self


@register_operator("ForecastOperator")
class ForecastOperator(Operator):
    """
    Time-series forecasting operator.
    
    Supported Methods:
    - moving_average: Simple moving average projection
    - exponential_smoothing: Single exponential smoothing
    - double_exponential: Holt's method (level + trend)
    - triple_exponential: Holt-Winters (level + trend + seasonality)
    - arima: ARIMA/SARIMA models
    - prophet: Facebook Prophet (requires fbprophet)
    - linear_trend: Simple linear regression trend
    
    Example Config:
        {
            "value_column": "temperature",
            "timestamp_column": "ts",
            "method": "exponential_smoothing",
            "horizon": 24,
            "alpha": 0.3
        }
    """
    
    name = "ForecastOperator"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="timestamp", dtype=DataType.DATETIME, required=True),
            SchemaField(name="value", dtype=DataType.FLOAT, required=True),
        ],
        description="Time-series data with timestamp and value columns"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="timestamp", dtype=DataType.DATETIME, required=True),
            SchemaField(name="forecast", dtype=DataType.FLOAT, required=True),
            SchemaField(name="forecast_lower", dtype=DataType.FLOAT, required=False),
            SchemaField(name="forecast_upper", dtype=DataType.FLOAT, required=False),
        ],
        description="Forecast values with optional confidence intervals"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[ForecastOperatorConfig] = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = ForecastOperatorConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid ForecastOperator config: {e}")
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Generate forecasts for the time-series data.
        
        Args:
            data: Input DataFrame with timestamp and value columns
            
        Returns:
            OperatorResult with forecast data
        """
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        # Validate columns exist
        for col in [config.value_column, config.timestamp_column]:
            if col not in data.columns:
                return OperatorResult(
                    success=False,
                    data=data,
                    error=f"Column '{col}' not found in data"
                )
        
        try:
            if config.group_by:
                result = self._forecast_grouped(data, config)
            else:
                result = self._forecast_single(data, config)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "method": config.method.value,
                    "horizon": config.horizon,
                    "value_column": config.value_column,
                    "forecast_count": config.horizon,
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Forecasting failed: {e}"
            )
    
    def _forecast_single(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Generate forecast for a single time-series."""
        
        # Sort by timestamp
        data = data.sort(config.timestamp_column)
        
        method_handlers = {
            ForecastMethod.MOVING_AVERAGE: self._forecast_moving_average,
            ForecastMethod.EXPONENTIAL_SMOOTHING: self._forecast_exponential,
            ForecastMethod.DOUBLE_EXPONENTIAL: self._forecast_double_exponential,
            ForecastMethod.TRIPLE_EXPONENTIAL: self._forecast_triple_exponential,
            ForecastMethod.LINEAR_TREND: self._forecast_linear_trend,
            ForecastMethod.ARIMA: self._forecast_arima,
            ForecastMethod.PROPHET: self._forecast_prophet,
        }
        
        handler = method_handlers.get(config.method)
        if not handler:
            raise ValueError(f"Unknown forecast method: {config.method}")
        
        return handler(data, config)
    
    def _forecast_grouped(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Generate forecasts for each group."""
        
        results = []
        
        for group_keys, group_data in data.group_by(config.group_by):
            # Forecast for this group
            group_config = ForecastOperatorConfig(
                **{**config.model_dump(), "group_by": None}
            )
            group_forecast = self._forecast_single(group_data, group_config)
            
            # Add group keys to forecast
            if isinstance(group_keys, tuple):
                for i, key_col in enumerate(config.group_by):
                    group_forecast = group_forecast.with_columns([
                        pl.lit(group_keys[i]).alias(key_col)
                    ])
            else:
                group_forecast = group_forecast.with_columns([
                    pl.lit(group_keys).alias(config.group_by[0])
                ])
            
            results.append(group_forecast)
        
        return pl.concat(results) if results else data.clear()
    
    def _forecast_moving_average(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Simple Moving Average forecast."""
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        if len(values) < config.window_size:
            raise ValueError(
                f"Not enough data points ({len(values)}) for "
                f"window size ({config.window_size})"
            )
        
        # Calculate moving average of last window
        last_window = values[-config.window_size:]
        ma = sum(last_window) / len(last_window)
        
        # Generate future timestamps
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = []
        future_vals = []
        
        for i in range(1, config.horizon + 1):
            future_ts.append(last_ts + avg_interval * i)
            future_vals.append(ma)
        
        # Build forecast dataframe
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: future_vals,
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            # Estimate std from historical data
            std = pl.Series(values).std() or 0
            z = 1.96 if config.confidence_level == 0.95 else 1.645
            
            forecast_data[f"{config.forecast_column}_lower"] = [ma - z * std] * config.horizon
            forecast_data[f"{config.forecast_column}_upper"] = [ma + z * std] * config.horizon
        
        forecast_df = pl.DataFrame(forecast_data)
        
        # Optionally include fitted values
        if config.include_fitted:
            fitted_vals = self._compute_rolling_mean(values, config.window_size)
            fitted_data = {
                config.timestamp_column: timestamps,
                config.forecast_column: fitted_vals,
                "is_forecast": [False] * len(timestamps),
            }
            if config.include_confidence:
                fitted_data[f"{config.forecast_column}_lower"] = fitted_vals
                fitted_data[f"{config.forecast_column}_upper"] = fitted_vals
            
            fitted_df = pl.DataFrame(fitted_data)
            return pl.concat([fitted_df, forecast_df])
        
        return forecast_df
    
    def _forecast_exponential(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Simple Exponential Smoothing forecast."""
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        alpha = config.alpha
        
        # Calculate smoothed values
        smoothed = [values[0]]
        for i in range(1, len(values)):
            s = alpha * values[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s)
        
        # Forecast is just the last smoothed value
        forecast_value = smoothed[-1]
        
        # Generate future timestamps
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = []
        future_vals = []
        
        for i in range(1, config.horizon + 1):
            future_ts.append(last_ts + avg_interval * i)
            future_vals.append(forecast_value)
        
        # Build forecast dataframe
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: future_vals,
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            residuals = [values[i] - smoothed[i] for i in range(len(values))]
            std = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5
            z = 1.96 if config.confidence_level == 0.95 else 1.645
            
            # Confidence interval widens with horizon
            lower = []
            upper = []
            for h in range(1, config.horizon + 1):
                width = z * std * (h ** 0.5)
                lower.append(forecast_value - width)
                upper.append(forecast_value + width)
            
            forecast_data[f"{config.forecast_column}_lower"] = lower
            forecast_data[f"{config.forecast_column}_upper"] = upper
        
        forecast_df = pl.DataFrame(forecast_data)
        
        if config.include_fitted:
            fitted_data = {
                config.timestamp_column: timestamps,
                config.forecast_column: smoothed,
                "is_forecast": [False] * len(timestamps),
            }
            if config.include_confidence:
                fitted_data[f"{config.forecast_column}_lower"] = smoothed
                fitted_data[f"{config.forecast_column}_upper"] = smoothed
            
            fitted_df = pl.DataFrame(fitted_data)
            return pl.concat([fitted_df, forecast_df])
        
        return forecast_df
    
    def _forecast_double_exponential(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Holt's Double Exponential Smoothing (level + trend)."""
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        alpha = config.alpha
        beta = config.beta or 0.1
        
        # Initialize
        level = [values[0]]
        trend = [values[1] - values[0] if len(values) > 1 else 0]
        
        # Calculate level and trend
        for i in range(1, len(values)):
            new_level = alpha * values[i] + (1 - alpha) * (level[-1] + trend[-1])
            new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
            level.append(new_level)
            trend.append(new_trend)
        
        # Forecast
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = []
        future_vals = []
        
        for h in range(1, config.horizon + 1):
            future_ts.append(last_ts + avg_interval * h)
            forecast = level[-1] + h * trend[-1]
            future_vals.append(forecast)
        
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: future_vals,
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            fitted = [level[i] + trend[i] for i in range(len(level))]
            residuals = [values[i] - fitted[i] for i in range(len(values))]
            std = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5
            z = 1.96 if config.confidence_level == 0.95 else 1.645
            
            lower = []
            upper = []
            for h in range(1, config.horizon + 1):
                width = z * std * (h ** 0.5)
                forecast = level[-1] + h * trend[-1]
                lower.append(forecast - width)
                upper.append(forecast + width)
            
            forecast_data[f"{config.forecast_column}_lower"] = lower
            forecast_data[f"{config.forecast_column}_upper"] = upper
        
        return pl.DataFrame(forecast_data)
    
    def _forecast_triple_exponential(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Holt-Winters Triple Exponential Smoothing."""
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        m = config.seasonal_period
        
        if len(values) < 2 * m:
            raise ValueError(
                f"Need at least {2 * m} data points for seasonal period {m}"
            )
        
        alpha = config.alpha
        beta = config.beta or 0.1
        gamma = config.gamma or 0.1
        additive = config.seasonality_type == SeasonalityType.ADDITIVE
        
        # Initialize seasonality
        season = []
        for i in range(m):
            if additive:
                season.append(values[i] - sum(values[:m]) / m)
            else:
                avg = sum(values[:m]) / m
                season.append(values[i] / avg if avg != 0 else 1)
        
        # Initialize level and trend
        level = [sum(values[:m]) / m]
        trend = [(sum(values[m:2*m]) - sum(values[:m])) / (m * m)]
        
        # Fit the model
        for i in range(len(values)):
            season_idx = i % m
            
            if additive:
                new_level = alpha * (values[i] - season[season_idx]) + (1 - alpha) * (level[-1] + trend[-1])
                new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
                new_season = gamma * (values[i] - new_level) + (1 - gamma) * season[season_idx]
            else:
                new_level = alpha * (values[i] / season[season_idx]) + (1 - alpha) * (level[-1] + trend[-1])
                new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
                new_season = gamma * (values[i] / new_level) + (1 - gamma) * season[season_idx]
            
            level.append(new_level)
            trend.append(new_trend)
            season.append(new_season)
        
        # Forecast
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = []
        future_vals = []
        
        for h in range(1, config.horizon + 1):
            future_ts.append(last_ts + avg_interval * h)
            
            season_idx = (len(values) + h - 1) % m + len(values)
            if season_idx >= len(season):
                season_idx = (len(values) + h - 1) % m
            
            if additive:
                forecast = level[-1] + h * trend[-1] + season[season_idx]
            else:
                forecast = (level[-1] + h * trend[-1]) * season[season_idx]
            
            future_vals.append(forecast)
        
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: future_vals,
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            # Simple confidence interval estimation
            std = pl.Series(values).std() or 0
            z = 1.96 if config.confidence_level == 0.95 else 1.645
            
            lower = [v - z * std for v in future_vals]
            upper = [v + z * std for v in future_vals]
            
            forecast_data[f"{config.forecast_column}_lower"] = lower
            forecast_data[f"{config.forecast_column}_upper"] = upper
        
        return pl.DataFrame(forecast_data)
    
    def _forecast_linear_trend(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Simple linear regression trend forecast."""
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        n = len(values)
        x = list(range(n))
        
        # Calculate linear regression coefficients
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
            intercept = y_mean
        else:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        
        # Generate forecast
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = []
        future_vals = []
        
        for h in range(1, config.horizon + 1):
            future_ts.append(last_ts + avg_interval * h)
            forecast = intercept + slope * (n - 1 + h)
            future_vals.append(forecast)
        
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: future_vals,
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            fitted = [intercept + slope * i for i in range(n)]
            residuals = [values[i] - fitted[i] for i in range(n)]
            std = (sum(r**2 for r in residuals) / n) ** 0.5
            z = 1.96 if config.confidence_level == 0.95 else 1.645
            
            lower = [v - z * std for v in future_vals]
            upper = [v + z * std for v in future_vals]
            
            forecast_data[f"{config.forecast_column}_lower"] = lower
            forecast_data[f"{config.forecast_column}_upper"] = upper
        
        return pl.DataFrame(forecast_data)
    
    def _forecast_arima(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """ARIMA/SARIMA forecast using statsmodels."""
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError(
                "statsmodels is required for ARIMA. "
                "Install with: pip install statsmodels"
            )
        
        values = data[config.value_column].to_list()
        timestamps = data[config.timestamp_column].to_list()
        
        # Fit ARIMA model
        model = ARIMA(values, order=config.arima_order)
        fitted_model = model.fit()
        
        # Forecast
        forecast_result = fitted_model.forecast(steps=config.horizon)
        
        if config.include_confidence:
            conf_int = fitted_model.get_forecast(steps=config.horizon).conf_int(
                alpha=1 - config.confidence_level
            )
        
        # Generate timestamps
        last_ts = timestamps[-1]
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        else:
            avg_interval = timedelta(hours=1)
        
        future_ts = [last_ts + avg_interval * h for h in range(1, config.horizon + 1)]
        
        forecast_data = {
            config.timestamp_column: future_ts,
            config.forecast_column: list(forecast_result),
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            forecast_data[f"{config.forecast_column}_lower"] = list(conf_int.iloc[:, 0])
            forecast_data[f"{config.forecast_column}_upper"] = list(conf_int.iloc[:, 1])
        
        return pl.DataFrame(forecast_data)
    
    def _forecast_prophet(
        self,
        data: pl.DataFrame,
        config: ForecastOperatorConfig
    ) -> pl.DataFrame:
        """Prophet forecast using Facebook Prophet."""
        
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is required for this method. "
                "Install with: pip install prophet"
            )
        
        # Prophet requires 'ds' and 'y' columns
        prophet_df = data.select([
            pl.col(config.timestamp_column).alias("ds"),
            pl.col(config.value_column).alias("y"),
        ]).to_pandas()
        
        # Fit model
        model = Prophet(interval_width=config.confidence_level)
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=config.horizon)
        
        # Predict
        forecast = model.predict(future)
        
        # Get only the forecast portion
        forecast_portion = forecast.tail(config.horizon)
        
        forecast_data = {
            config.timestamp_column: list(forecast_portion["ds"]),
            config.forecast_column: list(forecast_portion["yhat"]),
            "is_forecast": [True] * config.horizon,
        }
        
        if config.include_confidence:
            forecast_data[f"{config.forecast_column}_lower"] = list(forecast_portion["yhat_lower"])
            forecast_data[f"{config.forecast_column}_upper"] = list(forecast_portion["yhat_upper"])
        
        return pl.DataFrame(forecast_data)
    
    def _compute_rolling_mean(self, values: list, window: int) -> list:
        """Compute rolling mean for fitted values."""
        result = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(sum(values[:i+1]) / (i + 1))
            else:
                result.append(sum(values[i-window+1:i+1]) / window)
        return result
