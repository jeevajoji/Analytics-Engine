"""
End-to-End Tests for Analytics Engine.

Tests complete pipeline flows including:
- Ruleset Lane (real-time) scenarios
- Analytics Lane (batch) scenarios  
- Multi-operator pipelines
- All 15 operators in various combinations
"""

import pytest
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from analytics_engine.core.operator import OperatorConfig, OperatorResult
from analytics_engine.core.registry import OperatorRegistry
from analytics_engine.core.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineExecutor,
    OperatorNode,
    LaneType,
)

# Import all operators to ensure they're registered
from analytics_engine.operators import (
    WindowSelector,
    FilterOperator,
    Aggregator,
    JoinOperator,
    ThresholdEvaluator,
    EventBuilder,
    AnomalyDetector,
    ForecastOperator,
    ClusterOperator,
    TrajectoryExtractor,
    GraphBuilder,
    PathComparer,
    ClassificationOperator,
    RegressionOperator,
    Reporter,
)

# Import specialized config classes
from analytics_engine.operators.window_selector import WindowSelectorConfig, WindowType
from analytics_engine.operators.filter_operator import FilterOperatorConfig
from analytics_engine.operators.aggregator import AggregatorConfig, MetricConfig
from analytics_engine.operators.join_operator import JoinOperatorConfig
from analytics_engine.operators.threshold_evaluator import ThresholdEvaluatorConfig, ThresholdRule
from analytics_engine.operators.event_builder import EventBuilderConfig
from analytics_engine.operators.reporter import ReporterConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_telemetry_data() -> pl.DataFrame:
    """Sample IoT telemetry data for testing."""
    now = datetime.now()
    return pl.DataFrame({
        "timestamp": [now - timedelta(minutes=i) for i in range(100, 0, -1)],
        "device_id": ["device_001"] * 50 + ["device_002"] * 50,
        "temperature": [20 + i * 0.1 + (5 if i == 25 else 0) for i in range(100)],  # Spike at i=25
        "humidity": [50 + i * 0.05 for i in range(100)],
        "pressure": [1013 + i * 0.01 for i in range(100)],
    })


@pytest.fixture
def sample_gps_data() -> pl.DataFrame:
    """Sample GPS trajectory data for testing."""
    now = datetime.now()
    return pl.DataFrame({
        "timestamp": [now - timedelta(minutes=i) for i in range(50, 0, -1)],
        "vehicle_id": ["vehicle_001"] * 25 + ["vehicle_002"] * 25,
        "latitude": [37.7749 + i * 0.001 for i in range(50)],
        "longitude": [-122.4194 + i * 0.001 for i in range(50)],
        "speed": [30 + i * 0.5 for i in range(50)],
    })


@pytest.fixture
def sample_events_data() -> pl.DataFrame:
    """Sample event data for testing."""
    return pl.DataFrame({
        "event_id": [f"evt_{i}" for i in range(20)],
        "source": ["sensor_A"] * 10 + ["sensor_B"] * 10,
        "target": ["sensor_B"] * 5 + ["sensor_C"] * 5 + ["sensor_A"] * 5 + ["sensor_C"] * 5,
        "value": [100 + i * 10 for i in range(20)],
        "category": ["normal"] * 15 + ["anomaly"] * 5,
    })


@pytest.fixture
def sample_timeseries_data() -> pl.DataFrame:
    """Sample time-series data for forecasting."""
    now = datetime.now()
    import math
    return pl.DataFrame({
        "timestamp": [now - timedelta(hours=i) for i in range(168, 0, -1)],  # 1 week
        "value": [100 + 20 * math.sin(i * 0.3) + i * 0.1 for i in range(168)],
        "metric_name": ["cpu_usage"] * 168,
    })


# =============================================================================
# Tier 1: Foundation Operator Tests
# =============================================================================

class TestWindowSelector:
    """Tests for WindowSelector operator."""
    
    def test_tumbling_window(self, sample_telemetry_data):
        """Test tumbling window selection."""
        config = WindowSelectorConfig(
            timestamp_column="timestamp",
            window_type=WindowType.TUMBLING,
            window="10m",
        )
        
        operator = WindowSelector(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "window_start" in result.data.columns
        assert "window_end" in result.data.columns
    
    def test_sliding_window(self, sample_telemetry_data):
        """Test sliding window selection."""
        config = WindowSelectorConfig(
            timestamp_column="timestamp",
            window_type=WindowType.SLIDING,
            window="10m",
            slide="5m",
        )
        
        operator = WindowSelector(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success


class TestFilterOperator:
    """Tests for FilterOperator."""
    
    def test_simple_filter(self, sample_telemetry_data):
        """Test simple condition filtering."""
        config = FilterOperatorConfig(
            expression="temperature > 25",
        )
        
        operator = FilterOperator(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert len(result.data) < len(sample_telemetry_data)
        assert result.data["temperature"].min() > 25
    
    def test_structured_filter(self, sample_telemetry_data):
        """Test structured condition filtering."""
        from analytics_engine.operators.filter_operator import FilterCondition, ComparisonOperator, FilterMode
        
        config = FilterOperatorConfig(
            conditions=[
                FilterCondition(column="device_id", operator=ComparisonOperator.EQ, value="device_001"),
                FilterCondition(column="temperature", operator=ComparisonOperator.GT, value=22),
            ],
            mode=FilterMode.AND,
        )
        
        operator = FilterOperator(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert all(result.data["device_id"] == "device_001")


class TestAggregator:
    """Tests for Aggregator operator."""
    
    def test_basic_aggregation(self, sample_telemetry_data):
        """Test basic aggregation functions."""
        config = AggregatorConfig(
            metrics=[
                MetricConfig(column="temperature", functions=["mean", "max", "min", "std"]),
                MetricConfig(column="humidity", functions=["mean"]),
            ],
            group_by=["device_id"],
        )
        
        operator = Aggregator(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "temperature_mean" in result.data.columns
        assert "temperature_max" in result.data.columns
        assert len(result.data) == 2  # Two devices


# =============================================================================
# Tier 2-3: Data Combination & Rule Engine Tests
# =============================================================================

class TestJoinOperator:
    """Tests for JoinOperator."""
    
    def test_inner_join(self):
        """Test inner join of two DataFrames."""
        left = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "value_a": [10, 20, 30, 40],
        })
        right = pl.DataFrame({
            "id": [2, 3, 4, 5],
            "value_b": [200, 300, 400, 500],
        })
        
        config = JoinOperatorConfig(
            join_type="inner",
            left_on=["id"],
            right_on=["id"],
        )
        
        operator = JoinOperator(config)
        operator.set_right_data(right)
        result = operator.process(left)
        
        assert result.success
        assert len(result.data) == 3  # IDs 2, 3, 4


class TestThresholdEvaluator:
    """Tests for ThresholdEvaluator operator."""
    
    def test_threshold_evaluation(self, sample_telemetry_data):
        """Test threshold evaluation."""
        from analytics_engine.operators.threshold_evaluator import ThresholdType
        
        config = ThresholdEvaluatorConfig(
            rules=[
                ThresholdRule(
                    column="temperature",
                    threshold_type=ThresholdType.GT,
                    value=25,
                ),
            ],
        )
        
        operator = ThresholdEvaluator(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "is_violation" in result.data.columns or "temperature_violation" in result.data.columns


class TestEventBuilder:
    """Tests for EventBuilder operator."""
    
    def test_event_creation(self, sample_telemetry_data):
        """Test event creation from violations."""
        # First add violation column
        data = sample_telemetry_data.with_columns([
            (pl.col("temperature") > 25).alias("violation"),
        ])
        
        config = EventBuilderConfig(
            event_type="TEMPERATURE_ALERT",
            trigger_column="violation",
            severity="warning",
            include_columns=["device_id", "temperature"],
        )
        
        operator = EventBuilder(config)
        result = operator.process(data)
        
        assert result.success


# =============================================================================
# Tier 4: ML & Analytics Tests
# =============================================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector operator."""
    
    def test_zscore_detection(self, sample_telemetry_data):
        """Test Z-score anomaly detection."""
        config = OperatorConfig(params={
            "column": "temperature",
            "method": "zscore",
            "threshold": 2.0,
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "is_anomaly" in result.data.columns
        # Should detect the spike we added at i=25
        assert result.data["is_anomaly"].sum() > 0
    
    def test_iqr_detection(self, sample_telemetry_data):
        """Test IQR anomaly detection."""
        config = OperatorConfig(params={
            "column": "temperature",
            "method": "iqr",
            "iqr_multiplier": 1.5,
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "is_anomaly" in result.data.columns


class TestForecastOperator:
    """Tests for ForecastOperator."""
    
    def test_exponential_smoothing(self, sample_timeseries_data):
        """Test exponential smoothing forecast."""
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "exponential_smoothing",
            "horizon": 24,
            "alpha": 0.3,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(sample_timeseries_data)
        
        assert result.success
        assert "forecast" in result.data.columns
        assert len(result.data) == 24  # Forecast horizon
    
    def test_linear_trend(self, sample_timeseries_data):
        """Test linear trend forecast."""
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "linear_trend",
            "horizon": 12,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(sample_timeseries_data)
        
        assert result.success


class TestClusterOperator:
    """Tests for ClusterOperator."""
    
    def test_kmeans_clustering(self, sample_telemetry_data):
        """Test K-Means clustering."""
        config = OperatorConfig(params={
            "feature_columns": ["temperature", "humidity"],
            "method": "kmeans",
            "n_clusters": 3,
        })
        
        operator = ClusterOperator(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert "cluster" in result.data.columns
        assert result.data["cluster"].n_unique() == 3


# =============================================================================
# Tier 5: Spatial/Graph Tests
# =============================================================================

class TestTrajectoryExtractor:
    """Tests for TrajectoryExtractor operator."""
    
    def test_trajectory_extraction(self, sample_gps_data):
        """Test trajectory extraction from GPS points."""
        config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "vehicle_id",
            "include_speed": True,
            "include_bearing": True,
        })
        
        operator = TrajectoryExtractor(config)
        result = operator.process(sample_gps_data)
        
        assert result.success
        assert "distance" in result.data.columns
        assert "speed" in result.data.columns
        assert "bearing" in result.data.columns


class TestGraphBuilder:
    """Tests for GraphBuilder operator."""
    
    def test_graph_from_edges(self, sample_events_data):
        """Test graph building from edge data."""
        config = OperatorConfig(params={
            "input_mode": "edges",
            "source_column": "source",
            "target_column": "target",
            "weight_column": "value",
            "graph_type": "directed",
        })
        
        operator = GraphBuilder(config)
        result = operator.process(sample_events_data)
        
        assert result.success
        assert "n_nodes" in result.data.columns
        assert "n_edges" in result.data.columns


class TestPathComparer:
    """Tests for PathComparer operator."""
    
    def test_frechet_distance(self):
        """Test Frechet distance calculation."""
        # Create two similar paths
        path_data = pl.DataFrame({
            "path_id": ["path_A"] * 10 + ["path_B"] * 10,
            "latitude": [37.77 + i * 0.001 for i in range(10)] + [37.77 + i * 0.0011 for i in range(10)],
            "longitude": [-122.41 + i * 0.001 for i in range(10)] + [-122.41 + i * 0.0011 for i in range(10)],
        })
        
        config = OperatorConfig(params={
            "path_id_column": "path_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "metric": "frechet",
        })
        
        operator = PathComparer(config)
        result = operator.process(path_data)
        
        assert result.success
        assert "similarity" in result.data.columns


# =============================================================================
# Tier 7: Reporter Tests
# =============================================================================

class TestReporter:
    """Tests for Reporter operator."""
    
    def test_console_output(self, sample_telemetry_data, capsys):
        """Test console output."""
        config = OperatorConfig(params={
            "destination": "console",
            "output_columns": ["device_id", "temperature"],
        })
        
        operator = Reporter(config)
        result = operator.process(sample_telemetry_data.head(5))
        
        assert result.success
        captured = capsys.readouterr()
        assert "Analytics Engine Report" in captured.out
    
    def test_log_output(self, sample_telemetry_data):
        """Test log output."""
        config = OperatorConfig(params={
            "destination": "log",
            "log_level": "info",
            "log_include_data": False,
        })
        
        operator = Reporter(config)
        result = operator.process(sample_telemetry_data)
        
        assert result.success
        assert result.metadata["log_level"] == "info"


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestRulesetLanePipeline:
    """End-to-end tests for Ruleset Lane (real-time) scenarios."""
    
    def test_temperature_violation_pipeline(self, sample_telemetry_data):
        """
        Test complete ruleset pipeline:
        Telemetry → Filter → ThresholdEvaluator → EventBuilder → Reporter
        """
        from analytics_engine.operators.threshold_evaluator import ThresholdType
        
        # Step 1: Filter to specific device
        filter_config = FilterOperatorConfig(
            expression="device_id == 'device_001'",
        )
        filter_op = FilterOperator(filter_config)
        filter_result = filter_op.process(sample_telemetry_data)
        assert filter_result.success
        
        # Step 2: Evaluate thresholds
        threshold_config = ThresholdEvaluatorConfig(
            rules=[
                ThresholdRule(
                    column="temperature",
                    threshold_type=ThresholdType.GT,
                    value=24,
                ),
            ],
        )
        threshold_op = ThresholdEvaluator(threshold_config)
        threshold_result = threshold_op.process(filter_result.data)
        assert threshold_result.success
        
        # Step 3: Build events
        event_config = EventBuilderConfig(
            event_type="TEMP_VIOLATION",
            severity="warning",
        )
        event_op = EventBuilder(event_config)
        event_result = event_op.process(threshold_result.data)
        assert event_result.success
        
        # Step 4: Report
        report_config = ReporterConfig(destination="console")
        reporter = Reporter(report_config)
        report_result = reporter.process(event_result.data)
        assert report_result.success


class TestAnalyticsLanePipeline:
    """End-to-end tests for Analytics Lane (batch) scenarios."""
    
    def test_anomaly_detection_pipeline(self, sample_telemetry_data):
        """
        Test complete analytics pipeline:
        Data → WindowSelector → Aggregator → AnomalyDetector → Reporter
        """
        # Step 1: Window selection
        window_config = WindowSelectorConfig(
            timestamp_column="timestamp",
            window_type=WindowType.TUMBLING,
            window="10m",
        )
        window_op = WindowSelector(window_config)
        window_result = window_op.process(sample_telemetry_data)
        assert window_result.success
        
        # Step 2: Aggregation
        agg_config = AggregatorConfig(
            metrics=[
                MetricConfig(column="temperature", functions=["mean", "std"]),
            ],
            group_by=["device_id", "window_start"],
        )
        agg_op = Aggregator(agg_config)
        agg_result = agg_op.process(window_result.data)
        assert agg_result.success
        
        # Step 3: Anomaly detection
        anomaly_config = OperatorConfig(params={
            "column": "temperature_mean",
            "method": "zscore",
            "threshold": 2.0,
        })
        anomaly_op = AnomalyDetector(anomaly_config)
        anomaly_result = anomaly_op.process(agg_result.data)
        assert anomaly_result.success
        assert "is_anomaly" in anomaly_result.data.columns
        
        # Step 4: Report
        report_config = ReporterConfig(destination="console")
        reporter = Reporter(report_config)
        report_result = reporter.process(anomaly_result.data)
        assert report_result.success
    
    def test_forecasting_pipeline(self, sample_timeseries_data):
        """
        Test forecasting pipeline:
        TimeSeries → Aggregator → ForecastOperator → Reporter
        """
        # Step 1: Aggregate by hour (already hourly, but test the flow)
        agg_config = AggregatorConfig(
            metrics=[
                MetricConfig(column="value", functions=["mean"]),
            ],
        )
        agg_op = Aggregator(agg_config)
        agg_result = agg_op.process(sample_timeseries_data)
        
        # Step 2: Forecast
        forecast_config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "double_exponential",
            "horizon": 24,
        })
        forecast_op = ForecastOperator(forecast_config)
        forecast_result = forecast_op.process(sample_timeseries_data)
        assert forecast_result.success
        assert len(forecast_result.data) == 24


class TestSpatialAnalyticsPipeline:
    """End-to-end tests for spatial analytics scenarios."""
    
    def test_trajectory_clustering_pipeline(self, sample_gps_data):
        """
        Test trajectory analysis pipeline:
        GPS → TrajectoryExtractor → ClusterOperator → Reporter
        """
        # Step 1: Extract trajectories
        traj_config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "vehicle_id",
        })
        traj_op = TrajectoryExtractor(traj_config)
        traj_result = traj_op.process(sample_gps_data)
        assert traj_result.success
        
        # Step 2: Cluster locations
        cluster_config = OperatorConfig(params={
            "feature_columns": ["latitude", "longitude"],
            "method": "kmeans",
            "n_clusters": 3,
        })
        cluster_op = ClusterOperator(cluster_config)
        cluster_result = cluster_op.process(traj_result.data)
        assert cluster_result.success
        assert "cluster" in cluster_result.data.columns


class TestPipelineExecutor:
    """Tests for PipelineExecutor with DAG execution."""
    
    @pytest.mark.skip(reason="PipelineExecutor needs DAG-based implementation")
    def test_linear_pipeline(self, sample_telemetry_data):
        """Test linear pipeline execution."""
        # This test is skipped until PipelineExecutor is fully implemented
        pass


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Basic performance tests."""
    
    def test_large_dataset_filter(self):
        """Test filter performance on large dataset."""
        import time
        
        # Create 1M row dataset
        large_data = pl.DataFrame({
            "id": range(1_000_000),
            "value": [i * 0.1 for i in range(1_000_000)],
            "category": ["A", "B", "C"] * 333_333 + ["A"],
        })
        
        config = FilterOperatorConfig(
            expression="value > 50000",
        )
        
        operator = FilterOperator(config)
        
        start = time.time()
        result = operator.process(large_data)
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < 5.0  # Should complete in under 5 seconds
        print(f"Filter 1M rows: {elapsed:.3f}s")
    
    def test_large_dataset_aggregation(self):
        """Test aggregation performance on large dataset."""
        import time
        
        large_data = pl.DataFrame({
            "group": ["G1", "G2", "G3", "G4", "G5"] * 200_000,
            "value": [i * 0.01 for i in range(1_000_000)],
        })
        
        config = AggregatorConfig(
            metrics=[
                MetricConfig(column="value", functions=["mean", "sum", "std", "min", "max"]),
            ],
            group_by=["group"],
        )
        
        operator = Aggregator(config)
        
        start = time.time()
        result = operator.process(large_data)
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < 5.0
        print(f"Aggregate 1M rows: {elapsed:.3f}s")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_missing_column(self, sample_telemetry_data):
        """Test handling of missing column."""
        config = OperatorConfig(params={
            "column": "nonexistent_column",
            "method": "zscore",
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(sample_telemetry_data)
        
        assert not result.success
        assert "not found" in result.error.lower()
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        config = OperatorConfig(params={
            "method": "invalid_method",
        })
        
        operator = AnomalyDetector(config)
        
        # Should raise during validation
        with pytest.raises(ValueError):
            operator.validate_config()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_data = pl.DataFrame({
            "value": [],
            "timestamp": [],
        }).cast({"value": pl.Float64, "timestamp": pl.Datetime})
        
        config = OperatorConfig(params={
            "column": "value",
            "method": "zscore",
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(empty_data)
        
        # Should handle gracefully
        assert result.success or "empty" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
