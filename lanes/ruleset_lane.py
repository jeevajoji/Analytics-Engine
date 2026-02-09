"""
Ruleset Lane - Enhanced Real-time Processing Lane.

Handles per-event evaluation for safety violations, compliance rules,
and immediate incident generation with full I/O adapter support.
"""

import asyncio
from typing import Any, Dict, Iterator, Optional, Callable, List, Union
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import polars as pl

from ..core.pipeline import Pipeline, PipelineExecutor, PipelineExecutionResult, LaneType
from ..core.registry import OperatorRegistry, get_registry


logger = logging.getLogger(__name__)


@dataclass
class RulesetMetrics:
    """Metrics for Ruleset Lane performance monitoring."""
    records_processed: int = 0
    events_generated: int = 0
    incidents_generated: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    errors: int = 0
    started_at: Optional[datetime] = None
    
    @property
    def avg_latency_ms(self) -> float:
        if self.records_processed == 0:
            return 0.0
        return self.total_latency_ms / self.records_processed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "records_processed": self.records_processed,
            "events_generated": self.events_generated,
            "incidents_generated": self.incidents_generated,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


class InputAdapter(ABC):
    """Abstract base class for input adapters."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    def consume(self):
        """Consume records from the data source (async generator)."""
        pass


class OutputAdapter(ABC):
    """Abstract base class for output adapters."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data destination."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data destination."""
        pass
    
    @abstractmethod
    async def write(self, record: Dict[str, Any]) -> None:
        """Write a single record to the destination."""
        pass
    
    @abstractmethod
    async def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records to the destination."""
        pass


class RulesetLane:
    """
    Enhanced real-time processing lane for immediate event evaluation.
    
    Characteristics:
    - Ultra-low latency (<100ms target)
    - Per-event processing (no batching)
    - In-memory only
    - Deterministic rule evaluation
    - Full async support
    - I/O adapter integration
    
    Typical Flow:
        Redis/MQTT → FilterOperator → ThresholdEvaluator → EventBuilder → Incident
    
    Example:
        lane = RulesetLane()
        lane.register_pipeline(my_pipeline)
        lane.add_output_adapter(postgres_adapter)
        
        # Sync processing
        for record in incoming_stream:
            results = lane.process_record(record)
        
        # Async processing
        await lane.start_async(redis_adapter)
    """
    
    def __init__(
        self,
        registry: Optional[OperatorRegistry] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_incident: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
        latency_threshold_ms: float = 100.0,
    ):
        """
        Initialize the Ruleset Lane.
        
        Args:
            registry: Operator registry to use
            on_event: Callback when an event is generated
            on_incident: Callback when an incident is generated
            on_error: Callback when an error occurs
            latency_threshold_ms: Log warning if latency exceeds this
        """
        self.registry = registry or get_registry()
        self.executor = PipelineExecutor(self.registry)
        self.on_event = on_event
        self.on_incident = on_incident
        self.on_error = on_error
        self.latency_threshold_ms = latency_threshold_ms
        
        self._active_pipelines: Dict[str, Pipeline] = {}
        self._input_adapter: Optional[InputAdapter] = None
        self._output_adapters: List[OutputAdapter] = []
        self._running = False
        self._metrics = RulesetMetrics()
    
    def register_pipeline(self, pipeline: Pipeline) -> None:
        """
        Register a pipeline for real-time processing.
        
        Args:
            pipeline: Pipeline to register (must be REALTIME lane)
        """
        if pipeline.lane != LaneType.REALTIME:
            raise ValueError(f"Pipeline '{pipeline.name}' must be REALTIME lane")
        
        is_valid, errors = self.executor.validate_pipeline(pipeline)
        if not is_valid:
            raise ValueError(f"Invalid pipeline: {errors}")
        
        self._active_pipelines[pipeline.name] = pipeline
        logger.info(f"Registered pipeline '{pipeline.name}' for real-time processing")
    
    def unregister_pipeline(self, name: str) -> None:
        """Remove a pipeline from real-time processing."""
        if name in self._active_pipelines:
            del self._active_pipelines[name]
            logger.info(f"Unregistered pipeline '{name}'")
    
    def set_input_adapter(self, adapter: InputAdapter) -> None:
        """Set the input adapter for async processing."""
        self._input_adapter = adapter
    
    def add_output_adapter(self, adapter: OutputAdapter) -> None:
        """Add an output adapter for writing results."""
        self._output_adapters.append(adapter)
    
    def process_record(
        self, 
        record: Dict[str, Any],
        pipeline_name: Optional[str] = None,
    ) -> Dict[str, PipelineExecutionResult]:
        """
        Process a single record through pipelines.
        
        Args:
            record: Input data record
            pipeline_name: Specific pipeline to run (or all if None)
            
        Returns:
            Dict of pipeline name to execution result
        """
        start_time = datetime.now()
        results = {}
        df = pl.DataFrame([record])
        
        pipelines = self._active_pipelines
        if pipeline_name:
            if pipeline_name not in self._active_pipelines:
                raise ValueError(f"Pipeline '{pipeline_name}' not registered")
            pipelines = {pipeline_name: self._active_pipelines[pipeline_name]}
        
        for name, pipeline in pipelines.items():
            try:
                result = self.executor.execute(pipeline, df)
                results[name] = result
                
                if result.success and result.output_data is not None:
                    self._handle_output(result.output_data)
                    
            except Exception as e:
                logger.error(f"Pipeline '{name}' failed: {e}")
                self._metrics.errors += 1
                if self.on_error:
                    self.on_error(e, record)
        
        # Update metrics
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._metrics.records_processed += 1
        self._metrics.total_latency_ms += elapsed_ms
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, elapsed_ms)
        
        if elapsed_ms > self.latency_threshold_ms:
            logger.warning(f"High latency: {elapsed_ms:.2f}ms (threshold: {self.latency_threshold_ms}ms)")
        
        return results
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        pipeline_name: Optional[str] = None,
    ) -> List[Dict[str, PipelineExecutionResult]]:
        """
        Process multiple records (still per-record, but in sequence).
        
        For true batch processing, use AnalyticsLane.
        """
        return [self.process_record(record, pipeline_name) for record in records]
    
    def process_stream(
        self,
        stream: Iterator[Dict[str, Any]],
        pipeline_name: Optional[str] = None,
    ) -> Iterator[Dict[str, PipelineExecutionResult]]:
        """
        Process a stream of records.
        
        Args:
            stream: Iterator of data records
            pipeline_name: Specific pipeline to run
            
        Yields:
            Dict of pipeline name to execution result for each record
        """
        self._running = True
        self._metrics = RulesetMetrics()
        self._metrics.started_at = datetime.now()
        
        try:
            for record in stream:
                if not self._running:
                    break
                yield self.process_record(record, pipeline_name)
        finally:
            self._running = False
    
    async def start_async(
        self,
        input_adapter: Optional[InputAdapter] = None,
    ) -> None:
        """
        Start async processing from input adapter.
        
        Args:
            input_adapter: Input adapter to use (or use set adapter)
        """
        adapter = input_adapter or self._input_adapter
        if not adapter:
            raise ValueError("No input adapter configured")
        
        self._running = True
        self._metrics = RulesetMetrics()
        self._metrics.started_at = datetime.now()
        
        # Connect adapters
        await adapter.connect()
        for output in self._output_adapters:
            await output.connect()
        
        logger.info("Ruleset Lane started in async mode")
        
        try:
            async for record in adapter.consume():
                if not self._running:
                    break
                
                # Process record (sync, as operators are sync)
                results = self.process_record(record)
                
                # Write to output adapters async
                await self._write_outputs_async(results)
                
        finally:
            await adapter.disconnect()
            for output in self._output_adapters:
                await output.disconnect()
            self._running = False
            logger.info("Ruleset Lane stopped")
    
    async def _write_outputs_async(
        self,
        results: Dict[str, PipelineExecutionResult],
    ) -> None:
        """Write results to output adapters asynchronously."""
        for name, result in results.items():
            if result.success and result.output_data is not None:
                records = result.output_data.to_dicts()
                for adapter in self._output_adapters:
                    try:
                        await adapter.write_batch(records)
                    except Exception as e:
                        logger.error(f"Output adapter failed: {e}")
    
    def _handle_output(self, output_data: pl.DataFrame) -> None:
        """Handle pipeline output, triggering callbacks."""
        if output_data is None or len(output_data) == 0:
            return
        
        records = output_data.to_dicts()
        
        for record in records:
            # Check for events
            if "event_type" in record:
                self._metrics.events_generated += 1
                if self.on_event:
                    self.on_event(record)
            
            # Check for incidents
            if "event_id" in record or "incident_id" in record:
                self._metrics.incidents_generated += 1
                if self.on_incident:
                    self.on_incident(record)
    
    def stop(self) -> None:
        """Stop processing."""
        self._running = False
        logger.info("Ruleset Lane stop requested")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def pipeline_count(self) -> int:
        return len(self._active_pipelines)
    
    @property
    def metrics(self) -> RulesetMetrics:
        return self._metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = RulesetMetrics()
'''
This is the real-time lane for per-event evaluation (low latency).

Purpose
Designed for single-record processing with immediate results.
Typical use: safety violations, compliance, incidents.

Flow
Converts each record to a one-row DataFrame.
Executes a pipeline via PipelineExecutor.
Emits results via callbacks and/or output adapters.

Key features
Callbacks: on_event, on_incident, on_error.
Adapters:
InputAdapter (async data source)
OutputAdapter (async sink)
Metrics: tracks latency, counts, and errors.

Usage pattern
♦ Register real-time pipelines.
♦ Process incoming events one by one.
♦ Optionally run in async mode with adapters.'''