"""
Analytics Lane - Enhanced Batch Processing Lane with Scheduling.

Handles window-based processing for trends, insights, predictions,
and anomaly detection with full scheduler and I/O support.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import polars as pl

from ..core.pipeline import Pipeline, PipelineExecutor, PipelineExecutionResult, LaneType
from ..core.registry import OperatorRegistry, get_registry


logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of an analytics job."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority levels for job scheduling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AnalyticsJob:
    """Represents a scheduled analytics job."""
    pipeline: Pipeline
    job_id: str = ""
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[PipelineExecutionResult] = None
    error: Optional[str] = None
    input_rows: int = 0
    output_rows: int = 0
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"{self.pipeline.name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pipeline_name": self.pipeline.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "error": self.error,
            "retries": self.retries,
        }


@dataclass
class ScheduleConfig:
    """Configuration for scheduled pipeline execution."""
    pipeline_name: str
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    enabled: bool = True
    data_fetcher: Optional[Callable[[], pl.DataFrame]] = None
    window: Optional[str] = None
    priority: JobPriority = JobPriority.NORMAL
    
    def __post_init__(self):
        if not self.cron and not self.interval_seconds:
            raise ValueError("Either 'cron' or 'interval_seconds' must be set")


class DataFetcher(ABC):
    """Abstract base class for fetching input data."""
    
    @abstractmethod
    async def fetch(
        self, 
        pipeline_name: str, 
        window: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Fetch data for a pipeline."""
        pass


class InsightWriter(ABC):
    """Abstract base class for writing insights/results."""
    
    @abstractmethod
    async def write(self, job: "AnalyticsJob") -> None:
        """Write job results to destination."""
        pass


@dataclass
class AnalyticsMetrics:
    """Metrics for Analytics Lane performance."""
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_rows_processed: int = 0
    total_duration_seconds: float = 0.0
    last_run_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "total_rows_processed": self.total_rows_processed,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
        }


class AnalyticsLane:
    """
    Enhanced batch processing lane with scheduling.
    
    Characteristics:
    - Window-based processing (hours/days/weeks)
    - Scheduled execution (cron or interval)
    - Supports heavier ML models
    - Chunked processing for large datasets
    - Job queue with priorities
    - Retry on failure
    
    Typical Flow:
        InfluxDB → WindowSelector → Aggregator → AnomalyDetector → Reporter → Insights
    
    Example:
        lane = AnalyticsLane()
        lane.register_pipeline(my_pipeline)
        
        # Schedule hourly execution
        lane.schedule(ScheduleConfig(
            pipeline_name="my_pipeline",
            cron="0 * * * *",
            window="1h"
        ))
        
        # Start scheduler
        lane.start_scheduler()
        
        # Or run manually
        job = lane.run_pipeline("my_pipeline", data)
    """
    
    def __init__(
        self,
        registry: Optional[OperatorRegistry] = None,
        max_workers: int = 4,
        chunk_size: int = 10000,
        on_insight: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_job_complete: Optional[Callable[[AnalyticsJob], None]] = None,
        on_job_failed: Optional[Callable[[AnalyticsJob], None]] = None,
    ):
        """
        Initialize the Analytics Lane.
        
        Args:
            registry: Operator registry
            max_workers: Max concurrent jobs
            chunk_size: Default chunk size for processing
            on_insight: Callback for insights
            on_job_complete: Callback for completed jobs
            on_job_failed: Callback for failed jobs
        """
        self.registry = registry or get_registry()
        self.executor = PipelineExecutor(self.registry)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.on_insight = on_insight
        self.on_job_complete = on_job_complete
        self.on_job_failed = on_job_failed
        
        self._registered_pipelines: Dict[str, Pipeline] = {}
        self._schedules: Dict[str, ScheduleConfig] = {}
        self._job_queue: List[AnalyticsJob] = []
        self._job_history: List[AnalyticsJob] = []
        self._max_history: int = 1000
        
        self._data_fetcher: Optional[DataFetcher] = None
        self._insight_writer: Optional[InsightWriter] = None
        
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._executor_pool: Optional[ThreadPoolExecutor] = None
        self._metrics = AnalyticsMetrics()
        self._lock = threading.Lock()
    
    def register_pipeline(self, pipeline: Pipeline) -> None:
        """Register a pipeline for batch processing."""
        if pipeline.lane != LaneType.BATCH:
            raise ValueError(f"Pipeline '{pipeline.name}' must be BATCH lane")
        
        is_valid, errors = self.executor.validate_pipeline(pipeline)
        if not is_valid:
            raise ValueError(f"Invalid pipeline: {errors}")
        
        self._registered_pipelines[pipeline.name] = pipeline
        logger.info(f"Registered pipeline '{pipeline.name}' for batch processing")
    
    def unregister_pipeline(self, name: str) -> None:
        """Remove a pipeline."""
        if name in self._registered_pipelines:
            del self._registered_pipelines[name]
        if name in self._schedules:
            del self._schedules[name]
    
    def set_data_fetcher(self, fetcher: DataFetcher) -> None:
        """Set the data fetcher for scheduled jobs."""
        self._data_fetcher = fetcher
    
    def set_insight_writer(self, writer: InsightWriter) -> None:
        """Set the insight writer for job output."""
        self._insight_writer = writer
    
    def schedule(self, config: ScheduleConfig) -> None:
        """Schedule a pipeline for periodic execution."""
        if config.pipeline_name not in self._registered_pipelines:
            raise ValueError(f"Pipeline '{config.pipeline_name}' not registered")
        
        self._schedules[config.pipeline_name] = config
        logger.info(f"Scheduled pipeline '{config.pipeline_name}'")
    
    def unschedule(self, pipeline_name: str) -> None:
        """Remove a scheduled pipeline."""
        if pipeline_name in self._schedules:
            del self._schedules[pipeline_name]
    
    def run_pipeline(
        self,
        pipeline_name: str,
        input_data: pl.DataFrame,
        chunked: bool = True,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> AnalyticsJob:
        """
        Run a pipeline immediately on input data.
        
        Args:
            pipeline_name: Name of the pipeline
            input_data: Input DataFrame
            chunked: Whether to process in chunks
            priority: Job priority
            
        Returns:
            AnalyticsJob with results
        """
        if pipeline_name not in self._registered_pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not registered")
        
        pipeline = self._registered_pipelines[pipeline_name]
        job = AnalyticsJob(pipeline=pipeline, priority=priority)
        job.input_rows = len(input_data)
        
        self._execute_job(job, input_data, chunked)
        return job
    
    def queue_job(
        self,
        pipeline_name: str,
        input_data: pl.DataFrame,
        priority: JobPriority = JobPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
    ) -> AnalyticsJob:
        """Queue a job for execution."""
        if pipeline_name not in self._registered_pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not registered")
        
        pipeline = self._registered_pipelines[pipeline_name]
        job = AnalyticsJob(
            pipeline=pipeline,
            priority=priority,
            status=JobStatus.SCHEDULED,
            scheduled_at=scheduled_at or datetime.now(),
        )
        job.input_rows = len(input_data)
        job.metadata["input_data"] = input_data
        
        with self._lock:
            self._job_queue.append(job)
            self._job_queue.sort(
                key=lambda j: (-j.priority.value, j.scheduled_at or datetime.min)
            )
        
        logger.info(f"Queued job '{job.job_id}' for pipeline '{pipeline_name}'")
        return job
    
    def _execute_job(
        self,
        job: AnalyticsJob,
        input_data: pl.DataFrame,
        chunked: bool = True,
    ) -> None:
        """Execute a single job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            pipeline = job.pipeline
            
            if chunked and len(input_data) > self.chunk_size:
                all_outputs = []
                for chunk_result in self.executor.execute_chunked(
                    pipeline, 
                    input_data,
                    chunk_size=self.chunk_size
                ):
                    if chunk_result.success and chunk_result.output_data is not None:
                        all_outputs.append(chunk_result.output_data)
                    elif not chunk_result.success:
                        raise Exception(chunk_result.error)
                
                if all_outputs:
                    combined = pl.concat(all_outputs)
                    job.result = PipelineExecutionResult(
                        pipeline_name=pipeline.name,
                        success=True,
                        started_at=job.started_at,
                        completed_at=datetime.now(),
                        output_data=combined,
                        rows_processed=len(input_data),
                    )
                    job.output_rows = len(combined)
                else:
                    job.result = PipelineExecutionResult(
                        pipeline_name=pipeline.name,
                        success=True,
                        started_at=job.started_at,
                        completed_at=datetime.now(),
                        rows_processed=len(input_data),
                    )
            else:
                job.result = self.executor.execute(pipeline, input_data)
                if job.result.output_data is not None:
                    job.output_rows = len(job.result.output_data)
            
            if job.result.success:
                job.status = JobStatus.COMPLETED
                self._handle_output(job)
                self._metrics.jobs_completed += 1
                
                if self.on_job_complete:
                    self.on_job_complete(job)
            else:
                raise Exception(job.result.error)
                
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self._metrics.jobs_failed += 1
            logger.error(f"Job '{job.job_id}' failed: {e}")
            
            if job.retries < job.max_retries:
                job.retries += 1
                job.status = JobStatus.SCHEDULED
                job.scheduled_at = datetime.now() + timedelta(minutes=job.retries * 5)
                with self._lock:
                    self._job_queue.append(job)
                logger.info(f"Retry {job.retries}/{job.max_retries} scheduled for '{job.job_id}'")
            elif self.on_job_failed:
                self.on_job_failed(job)
        
        job.completed_at = datetime.now()
        self._metrics.total_rows_processed += job.input_rows
        if job.duration_seconds:
            self._metrics.total_duration_seconds += job.duration_seconds
        self._metrics.last_run_at = datetime.now()
        
        with self._lock:
            self._job_history.append(job)
            if len(self._job_history) > self._max_history:
                self._job_history = self._job_history[-self._max_history:]
    
    def _handle_output(self, job: AnalyticsJob) -> None:
        """Handle job output."""
        if job.result is None or job.result.output_data is None:
            return
        
        output_data = job.result.output_data
        if len(output_data) == 0:
            return
        
        if self.on_insight:
            for record in output_data.to_dicts():
                self.on_insight(record)
    
    def start_scheduler(self) -> None:
        """Start the background scheduler."""
        if self._running:
            return
        
        self._running = True
        self._executor_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Analytics Lane scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self._executor_pool:
            self._executor_pool.shutdown(wait=True)
        logger.info("Analytics Lane scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        last_check: Dict[str, datetime] = {}
        
        while self._running:
            now = datetime.now()
            
            for name, config in self._schedules.items():
                if not config.enabled:
                    continue
                
                should_run = False
                
                if config.interval_seconds:
                    last = last_check.get(name)
                    if last is None or (now - last).total_seconds() >= config.interval_seconds:
                        should_run = True
                        last_check[name] = now
                
                elif config.cron:
                    should_run = self._check_cron(config.cron, now, last_check.get(name))
                    if should_run:
                        last_check[name] = now
                
                if should_run:
                    self._trigger_scheduled_job(name, config)
            
            self._process_queue()
            
            import time
            time.sleep(1)
    
    def _check_cron(
        self, 
        cron: str, 
        now: datetime, 
        last_run: Optional[datetime]
    ) -> bool:
        """Simple cron check (minute-level granularity)."""
        parts = cron.split()
        if len(parts) != 5:
            return False
        
        minute, hour, day, month, weekday = parts
        
        def matches(value: str, current: int) -> bool:
            if value == "*":
                return True
            if value.startswith("*/"):
                step = int(value[2:])
                return current % step == 0
            return int(value) == current
        
        if not all([
            matches(minute, now.minute),
            matches(hour, now.hour),
            matches(day, now.day),
            matches(month, now.month),
            matches(weekday, now.weekday()),
        ]):
            return False
        
        if last_run and last_run.replace(second=0, microsecond=0) == now.replace(second=0, microsecond=0):
            return False
        
        return True
    
    def _trigger_scheduled_job(self, name: str, config: ScheduleConfig) -> None:
        """Trigger a scheduled job."""
        logger.info(f"Triggering scheduled job for '{name}'")
        
        try:
            if config.data_fetcher:
                input_data = config.data_fetcher()
            elif self._data_fetcher:
                loop = asyncio.new_event_loop()
                try:
                    input_data = loop.run_until_complete(
                        self._data_fetcher.fetch(name, window=config.window)
                    )
                finally:
                    loop.close()
            else:
                logger.warning(f"No data fetcher for scheduled job '{name}'")
                return
            
            self.queue_job(name, input_data, priority=config.priority)
            
        except Exception as e:
            logger.error(f"Failed to trigger scheduled job '{name}': {e}")
    
    def _process_queue(self) -> None:
        """Process pending jobs from queue."""
        now = datetime.now()
        jobs_to_run = []
        
        with self._lock:
            for job in self._job_queue[:]:
                if job.scheduled_at and job.scheduled_at <= now:
                    jobs_to_run.append(job)
                    self._job_queue.remove(job)
        
        for job in jobs_to_run:
            input_data = job.metadata.get("input_data")
            if input_data is not None:
                if self._executor_pool:
                    self._executor_pool.submit(self._execute_job, job, input_data, True)
                else:
                    self._execute_job(job, input_data, True)
    
    def run_all_pipelines(
        self,
        data_fetcher: Callable[[str], pl.DataFrame],
    ) -> List[AnalyticsJob]:
        """Run all registered pipelines once."""
        jobs = []
        for name in self._registered_pipelines:
            input_data = data_fetcher(name)
            job = self.run_pipeline(name, input_data)
            jobs.append(job)
        return jobs
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def pipeline_count(self) -> int:
        return len(self._registered_pipelines)
    
    @property
    def schedule_count(self) -> int:
        return len(self._schedules)
    
    @property
    def queue_length(self) -> int:
        return len(self._job_queue)
    
    @property
    def metrics(self) -> AnalyticsMetrics:
        return self._metrics
    
    def get_job_history(
        self, 
        limit: int = 100,
        status: Optional[JobStatus] = None,
    ) -> List[AnalyticsJob]:
        """Get job history."""
        history = self._job_history[-limit:]
        if status:
            history = [j for j in history if j.status == status]
        return history
    
    def get_job(self, job_id: str) -> Optional[AnalyticsJob]:
        """Get a specific job by ID."""
        for job in self._job_history:
            if job.job_id == job_id:
                return job
        for job in self._job_queue:
            if job.job_id == job_id:
                return job
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        with self._lock:
            for job in self._job_queue:
                if job.job_id == job_id:
                    job.status = JobStatus.CANCELLED
                    self._job_queue.remove(job)
                    self._job_history.append(job)
                    return True
        return False
'''
This is the batch/scheduled lane for windowed analytics.

Purpose :Runs pipelines on larger datasets (hourly/daily windows).
Supports scheduling, job queue, and retries.

Key features
Scheduling: cron or interval via ScheduleConfig.
Job system:
Analytics Job holds execution metadata.
Priority queue + retry logic.
Concurrency: worker thread pool for running jobs.
Chunking: splits large datasets for memory efficiency.
Metrics: tracks throughput and job success/failure.

Usage pattern -
♦ Register batch pipelines.
♦ Schedule them or run on demand.'''