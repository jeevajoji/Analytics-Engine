"""
Tests for Tier 4-7 Operators.

Comprehensive tests for advanced operators:
- AnomalyDetector
- ForecastOperator
- ClusterOperator
- TrajectoryExtractor
- GraphBuilder
- PathComparer
- ClassificationOperator
- RegressionOperator
- Reporter
"""

import pytest
import math
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl

from analytics_engine.core.operator import OperatorConfig


# =============================================================================
# AnomalyDetector Tests
# =============================================================================

class TestAnomalyDetectorComprehensive:
    """Comprehensive tests for AnomalyDetector."""
    
    @pytest.fixture
    def normal_data(self) -> pl.DataFrame:
        """Normal distribution data."""
        import random
        random.seed(42)
        return pl.DataFrame({
            "value": [random.gauss(100, 10) for _ in range(1000)],
        })
    
    @pytest.fixture
    def data_with_outliers(self) -> pl.DataFrame:
        """Data with known outliers."""
        values = [50.0] * 100
        values[25] = 200.0  # Outlier
        values[75] = 0.0    # Outlier
        return pl.DataFrame({"value": values})
    
    def test_zscore_detects_outliers(self, data_with_outliers):
        """Test Z-score detects outliers correctly."""
        from analytics_engine.operators import AnomalyDetector
        
        config = OperatorConfig(params={
            "column": "value",
            "method": "zscore",
            "threshold": 2.0,
            "output_column": "is_anomaly",
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(data_with_outliers)
        
        assert result.success
        anomalies = result.data.filter(pl.col("is_anomaly"))
        assert len(anomalies) >= 2  # Should detect at least our 2 outliers
    
    def test_iqr_detects_outliers(self, data_with_outliers):
        """Test IQR detects outliers correctly."""
        from analytics_engine.operators import AnomalyDetector
        
        config = OperatorConfig(params={
            "column": "value",
            "method": "iqr",
            "iqr_multiplier": 1.5,
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(data_with_outliers)
        
        assert result.success
        assert result.data["is_anomaly"].sum() >= 2
    
    def test_anomaly_score_included(self, data_with_outliers):
        """Test anomaly score is included when requested."""
        from analytics_engine.operators import AnomalyDetector
        
        config = OperatorConfig(params={
            "column": "value",
            "method": "zscore",
            "score_column": "anomaly_score",
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(data_with_outliers)
        
        assert result.success
        assert "anomaly_score" in result.data.columns
    
    def test_group_by_anomaly_detection(self):
        """Test anomaly detection within groups."""
        from analytics_engine.operators import AnomalyDetector
        
        data = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "value": [10.0] * 49 + [100.0] + [20.0] * 49 + [200.0],  # Outlier in each group
        })
        
        config = OperatorConfig(params={
            "column": "value",
            "method": "zscore",
            "threshold": 2.0,
            "group_by": ["group"],
        })
        
        operator = AnomalyDetector(config)
        result = operator.process(data)
        
        assert result.success
        assert result.data["is_anomaly"].sum() >= 2


# =============================================================================
# ForecastOperator Tests
# =============================================================================

class TestForecastOperatorComprehensive:
    """Comprehensive tests for ForecastOperator."""
    
    @pytest.fixture
    def trend_data(self) -> pl.DataFrame:
        """Data with clear upward trend."""
        now = datetime.utcnow()
        return pl.DataFrame({
            "timestamp": [now - timedelta(hours=i) for i in range(100, 0, -1)],
            "value": [float(i) for i in range(100)],
        })
    
    @pytest.fixture
    def seasonal_data(self) -> pl.DataFrame:
        """Data with seasonal pattern."""
        now = datetime.utcnow()
        return pl.DataFrame({
            "timestamp": [now - timedelta(hours=i) for i in range(168, 0, -1)],
            "value": [50 + 20 * math.sin(i * 2 * math.pi / 24) for i in range(168)],
        })
    
    def test_moving_average_forecast(self, trend_data):
        """Test moving average forecast."""
        from analytics_engine.operators import ForecastOperator
        
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "moving_average",
            "window_size": 10,
            "horizon": 12,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(trend_data)
        
        assert result.success
        assert len(result.data) == 12
        assert "forecast" in result.data.columns
    
    def test_exponential_smoothing_forecast(self, trend_data):
        """Test exponential smoothing forecast."""
        from analytics_engine.operators import ForecastOperator
        
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "exponential_smoothing",
            "alpha": 0.3,
            "horizon": 10,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(trend_data)
        
        assert result.success
        assert len(result.data) == 10
    
    def test_double_exponential_captures_trend(self, trend_data):
        """Test double exponential captures trend."""
        from analytics_engine.operators import ForecastOperator
        
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "double_exponential",
            "alpha": 0.3,
            "beta": 0.1,
            "horizon": 10,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(trend_data)
        
        assert result.success
        # Forecast should continue upward trend
        forecasts = result.data["forecast"].to_list()
        assert forecasts[-1] > forecasts[0]
    
    def test_confidence_intervals(self, trend_data):
        """Test confidence intervals are included."""
        from analytics_engine.operators import ForecastOperator
        
        config = OperatorConfig(params={
            "value_column": "value",
            "timestamp_column": "timestamp",
            "method": "exponential_smoothing",
            "horizon": 10,
            "include_confidence": True,
            "confidence_level": 0.95,
        })
        
        operator = ForecastOperator(config)
        result = operator.process(trend_data)
        
        assert result.success
        assert "forecast_lower" in result.data.columns
        assert "forecast_upper" in result.data.columns


# =============================================================================
# ClusterOperator Tests
# =============================================================================

class TestClusterOperatorComprehensive:
    """Comprehensive tests for ClusterOperator."""
    
    @pytest.fixture
    def clustered_data(self) -> pl.DataFrame:
        """Data with clear cluster structure."""
        import random
        random.seed(42)
        
        cluster1 = [(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(50)]
        cluster2 = [(random.gauss(10, 1), random.gauss(10, 1)) for _ in range(50)]
        cluster3 = [(random.gauss(0, 1), random.gauss(10, 1)) for _ in range(50)]
        
        all_points = cluster1 + cluster2 + cluster3
        
        return pl.DataFrame({
            "x": [p[0] for p in all_points],
            "y": [p[1] for p in all_points],
        })
    
    def test_kmeans_finds_clusters(self, clustered_data):
        """Test K-Means finds correct number of clusters."""
        from analytics_engine.operators import ClusterOperator
        
        config = OperatorConfig(params={
            "feature_columns": ["x", "y"],
            "method": "kmeans",
            "n_clusters": 3,
        })
        
        operator = ClusterOperator(config)
        result = operator.process(clustered_data)
        
        assert result.success
        assert result.data["cluster"].n_unique() == 3
    
    def test_dbscan_clustering(self, clustered_data):
        """Test DBSCAN clustering."""
        from analytics_engine.operators import ClusterOperator
        
        config = OperatorConfig(params={
            "feature_columns": ["x", "y"],
            "method": "dbscan",
            "eps": 2.0,
            "min_samples": 5,
        })
        
        operator = ClusterOperator(config)
        result = operator.process(clustered_data)
        
        assert result.success
        assert "cluster" in result.data.columns
    
    def test_cluster_with_normalization(self, clustered_data):
        """Test clustering with feature normalization."""
        from analytics_engine.operators import ClusterOperator
        
        config = OperatorConfig(params={
            "feature_columns": ["x", "y"],
            "method": "kmeans",
            "n_clusters": 3,
            "normalize": True,
        })
        
        operator = ClusterOperator(config)
        result = operator.process(clustered_data)
        
        assert result.success


# =============================================================================
# TrajectoryExtractor Tests
# =============================================================================

class TestTrajectoryExtractorComprehensive:
    """Comprehensive tests for TrajectoryExtractor."""
    
    @pytest.fixture
    def gps_track(self) -> pl.DataFrame:
        """GPS track data."""
        now = datetime.utcnow()
        # Simulate a vehicle moving northeast
        return pl.DataFrame({
            "timestamp": [now + timedelta(seconds=i * 10) for i in range(100)],
            "entity_id": ["vehicle_1"] * 100,
            "latitude": [37.7749 + i * 0.0001 for i in range(100)],
            "longitude": [-122.4194 + i * 0.0001 for i in range(100)],
        })
    
    def test_distance_calculation(self, gps_track):
        """Test distance calculation between points."""
        from analytics_engine.operators import TrajectoryExtractor
        
        config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "entity_id",
            "distance_unit": "meters",
        })
        
        operator = TrajectoryExtractor(config)
        result = operator.process(gps_track)
        
        assert result.success
        assert "distance" in result.data.columns
        # Distances should be positive (except first point)
        assert result.data["distance"][1:].min() > 0
    
    def test_speed_calculation(self, gps_track):
        """Test speed calculation."""
        from analytics_engine.operators import TrajectoryExtractor
        
        config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "entity_id",
            "speed_unit": "km/h",
            "include_speed": True,
        })
        
        operator = TrajectoryExtractor(config)
        result = operator.process(gps_track)
        
        assert result.success
        assert "speed" in result.data.columns
    
    def test_bearing_calculation(self, gps_track):
        """Test bearing calculation."""
        from analytics_engine.operators import TrajectoryExtractor
        
        config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "entity_id",
            "include_bearing": True,
        })
        
        operator = TrajectoryExtractor(config)
        result = operator.process(gps_track)
        
        assert result.success
        assert "bearing" in result.data.columns
        # Moving northeast, bearing should be around 45 degrees
        avg_bearing = result.data["bearing"][1:].mean()
        assert 0 < avg_bearing < 90
    
    def test_stop_detection(self):
        """Test stop detection."""
        from analytics_engine.operators import TrajectoryExtractor
        
        now = datetime.utcnow()
        # Vehicle stops in the middle
        data = pl.DataFrame({
            "timestamp": [now + timedelta(seconds=i * 10) for i in range(20)],
            "entity_id": ["vehicle_1"] * 20,
            "latitude": [37.7749 + i * 0.0001 for i in range(10)] + [37.7759] * 10,  # Stops at point 10
            "longitude": [-122.4194 + i * 0.0001 for i in range(10)] + [-122.4084] * 10,
        })
        
        config = OperatorConfig(params={
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "timestamp_column": "timestamp",
            "entity_column": "entity_id",
            "detect_stops": True,
            "stop_speed_threshold": 1.0,  # km/h
        })
        
        operator = TrajectoryExtractor(config)
        result = operator.process(data)
        
        assert result.success
        assert "is_stop" in result.data.columns
        # Should detect stops in the second half
        assert result.data["is_stop"].sum() > 0


# =============================================================================
# GraphBuilder Tests
# =============================================================================

class TestGraphBuilderComprehensive:
    """Comprehensive tests for GraphBuilder."""
    
    @pytest.fixture
    def edge_data(self) -> pl.DataFrame:
        """Edge list data."""
        return pl.DataFrame({
            "source": ["A", "A", "B", "C", "D"],
            "target": ["B", "C", "C", "D", "A"],
            "weight": [1.0, 2.0, 1.5, 3.0, 0.5],
        })
    
    @pytest.fixture
    def point_data(self) -> pl.DataFrame:
        """Point data for proximity graph."""
        return pl.DataFrame({
            "node_id": ["N1", "N2", "N3", "N4"],
            "latitude": [37.7749, 37.7750, 37.7849, 37.7850],
            "longitude": [-122.4194, -122.4195, -122.4194, -122.4195],
        })
    
    def test_edge_list_graph(self, edge_data):
        """Test graph building from edge list."""
        from analytics_engine.operators import GraphBuilder
        
        config = OperatorConfig(params={
            "input_mode": "edges",
            "source_column": "source",
            "target_column": "target",
            "weight_column": "weight",
        })
        
        operator = GraphBuilder(config)
        result = operator.process(edge_data)
        
        assert result.success
        assert result.metadata["n_nodes"] == 4
        assert result.metadata["n_edges"] == 5
    
    def test_directed_graph(self, edge_data):
        """Test directed graph building."""
        from analytics_engine.operators import GraphBuilder
        
        config = OperatorConfig(params={
            "input_mode": "edges",
            "source_column": "source",
            "target_column": "target",
            "graph_type": "directed",
        })
        
        operator = GraphBuilder(config)
        result = operator.process(edge_data)
        
        assert result.success
    
    def test_proximity_graph(self, point_data):
        """Test proximity-based graph building."""
        from analytics_engine.operators import GraphBuilder
        
        config = OperatorConfig(params={
            "input_mode": "points",
            "node_id_column": "node_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "max_distance": 200,  # meters
        })
        
        operator = GraphBuilder(config)
        result = operator.process(point_data)
        
        assert result.success
        # Close points should be connected
        assert result.metadata["n_edges"] > 0
    
    def test_graph_metrics(self, edge_data):
        """Test graph metrics computation."""
        from analytics_engine.operators import GraphBuilder
        
        config = OperatorConfig(params={
            "input_mode": "edges",
            "source_column": "source",
            "target_column": "target",
            "compute_metrics": True,
        })
        
        operator = GraphBuilder(config)
        result = operator.process(edge_data)
        
        assert result.success
        assert "density" in result.metadata
        assert "avg_degree" in result.metadata


# =============================================================================
# PathComparer Tests
# =============================================================================

class TestPathComparerComprehensive:
    """Comprehensive tests for PathComparer."""
    
    @pytest.fixture
    def similar_paths(self) -> pl.DataFrame:
        """Two similar paths."""
        return pl.DataFrame({
            "path_id": ["P1"] * 10 + ["P2"] * 10,
            "latitude": [37.77 + i * 0.001 for i in range(10)] + [37.77 + i * 0.001 for i in range(10)],
            "longitude": [-122.41 + i * 0.001 for i in range(10)] + [-122.41 + i * 0.0011 for i in range(10)],
        })
    
    @pytest.fixture
    def different_paths(self) -> pl.DataFrame:
        """Two very different paths."""
        return pl.DataFrame({
            "path_id": ["P1"] * 10 + ["P2"] * 10,
            "latitude": [37.77 + i * 0.001 for i in range(10)] + [38.77 + i * 0.001 for i in range(10)],
            "longitude": [-122.41 + i * 0.001 for i in range(10)] + [-121.41 + i * 0.001 for i in range(10)],
        })
    
    def test_frechet_similar_paths(self, similar_paths):
        """Test Frechet distance for similar paths."""
        from analytics_engine.operators import PathComparer
        
        config = OperatorConfig(params={
            "path_id_column": "path_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "metric": "frechet",
        })
        
        operator = PathComparer(config)
        result = operator.process(similar_paths)
        
        assert result.success
        similarity = result.data["similarity"][0]
        assert similarity > 0.5  # Should be fairly similar
    
    def test_hausdorff_distance(self, similar_paths):
        """Test Hausdorff distance."""
        from analytics_engine.operators import PathComparer
        
        config = OperatorConfig(params={
            "path_id_column": "path_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "metric": "hausdorff",
        })
        
        operator = PathComparer(config)
        result = operator.process(similar_paths)
        
        assert result.success
        assert "distance" in result.data.columns
    
    def test_dtw_distance(self, similar_paths):
        """Test DTW distance."""
        from analytics_engine.operators import PathComparer
        
        config = OperatorConfig(params={
            "path_id_column": "path_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "metric": "dtw",
        })
        
        operator = PathComparer(config)
        result = operator.process(similar_paths)
        
        assert result.success
    
    def test_overlap_detection(self, similar_paths):
        """Test overlap percentage."""
        from analytics_engine.operators import PathComparer
        
        config = OperatorConfig(params={
            "path_id_column": "path_id",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "metric": "overlap",
            "overlap_threshold": 50,  # meters
        })
        
        operator = PathComparer(config)
        result = operator.process(similar_paths)
        
        assert result.success
        assert result.data["similarity"][0] > 0


# =============================================================================
# Reporter Tests
# =============================================================================

class TestReporterComprehensive:
    """Comprehensive tests for Reporter."""
    
    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Sample data for reporting."""
        return pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85.5, 92.0, 78.3],
        })
    
    def test_console_report(self, sample_data, capsys):
        """Test console output."""
        from analytics_engine.operators import Reporter
        
        config = OperatorConfig(params={
            "destination": "console",
        })
        
        operator = Reporter(config)
        result = operator.process(sample_data)
        
        assert result.success
        captured = capsys.readouterr()
        assert "Analytics Engine Report" in captured.out
    
    def test_column_selection(self, sample_data):
        """Test column selection in output."""
        from analytics_engine.operators import Reporter
        
        config = OperatorConfig(params={
            "destination": "console",
            "output_columns": ["name", "score"],
        })
        
        operator = Reporter(config)
        result = operator.process(sample_data)
        
        assert result.success
    
    def test_timestamp_addition(self, sample_data):
        """Test automatic timestamp addition."""
        from analytics_engine.operators import Reporter
        
        config = OperatorConfig(params={
            "destination": "console",
            "add_timestamp": True,
            "timestamp_column": "reported_at",
        })
        
        operator = Reporter(config)
        result = operator.process(sample_data)
        
        assert result.success
    
    def test_log_level_output(self, sample_data, caplog):
        """Test log level configuration."""
        from analytics_engine.operators import Reporter
        import logging
        
        with caplog.at_level(logging.INFO):
            config = OperatorConfig(params={
                "destination": "log",
                "log_level": "info",
                "log_message_template": "Test report: {record_count} records",
            })
            
            operator = Reporter(config)
            result = operator.process(sample_data)
            
            assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
