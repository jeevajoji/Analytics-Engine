"""
Pipeline Versioning Module.

Provides version management for pipelines including:
- Semantic versioning
- Schema evolution tracking
- A/B testing support
- Rollback capabilities
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import copy


logger = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """Status of a pipeline version."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChangeType(str, Enum):
    """Types of changes between versions."""
    OPERATOR_ADDED = "operator_added"
    OPERATOR_REMOVED = "operator_removed"
    OPERATOR_MODIFIED = "operator_modified"
    PARAM_CHANGED = "param_changed"
    ORDER_CHANGED = "order_changed"
    SCHEMA_CHANGED = "schema_changed"


@dataclass
class SemanticVersion:
    """
    Semantic version representation (MAJOR.MINOR.PATCH).
    
    - MAJOR: Breaking changes (schema incompatible)
    - MINOR: New features (backward compatible)
    - PATCH: Bug fixes (backward compatible)
    """
    major: int = 1
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version
    
    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))
    
    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        prerelease = None
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)
        
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            prerelease=prerelease,
        )
    
    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class VersionChange:
    """Record of a change between versions."""
    change_type: ChangeType
    description: str
    path: str  # JSON path to changed element
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    breaking: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "description": self.description,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "breaking": self.breaking,
        }


@dataclass
class PipelineVersion:
    """
    A versioned pipeline definition.
    
    Attributes:
        pipeline_name: Name of the pipeline
        version: Semantic version
        definition: Full pipeline definition (dict)
        schema_hash: Hash of the output schema
        created_at: When version was created
        created_by: Who created the version
        status: Version status
        changelog: List of changes from previous version
        metadata: Additional metadata
    """
    pipeline_name: str
    version: SemanticVersion
    definition: Dict[str, Any]
    schema_hash: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    status: VersionStatus = VersionStatus.DRAFT
    changelog: List[VersionChange] = field(default_factory=list)
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.schema_hash:
            self.schema_hash = self._compute_schema_hash()
    
    def _compute_schema_hash(self) -> str:
        """Compute hash of the pipeline definition."""
        content = json.dumps(self.definition, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def version_id(self) -> str:
        """Unique version identifier."""
        return f"{self.pipeline_name}@{self.version}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "version": str(self.version),
            "definition": self.definition,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "changelog": [c.to_dict() for c in self.changelog],
            "parent_version": self.parent_version,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineVersion":
        return cls(
            pipeline_name=data["pipeline_name"],
            version=SemanticVersion.parse(data["version"]),
            definition=data["definition"],
            schema_hash=data.get("schema_hash", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "system"),
            status=VersionStatus(data.get("status", "draft")),
            changelog=[],  # Skip changelog parsing for simplicity
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {}),
        )


class VersionStore(ABC):
    """Abstract base class for version storage."""
    
    @abstractmethod
    async def save(self, version: PipelineVersion) -> None:
        """Save a pipeline version."""
        pass
    
    @abstractmethod
    async def get(self, pipeline_name: str, version: str) -> Optional[PipelineVersion]:
        """Get a specific version."""
        pass
    
    @abstractmethod
    async def get_latest(self, pipeline_name: str, status: Optional[VersionStatus] = None) -> Optional[PipelineVersion]:
        """Get the latest version of a pipeline."""
        pass
    
    @abstractmethod
    async def list_versions(self, pipeline_name: str) -> List[PipelineVersion]:
        """List all versions of a pipeline."""
        pass
    
    @abstractmethod
    async def delete(self, pipeline_name: str, version: str) -> bool:
        """Delete a version."""
        pass


class InMemoryVersionStore(VersionStore):
    """In-memory version store for development/testing."""
    
    def __init__(self):
        self._versions: Dict[str, Dict[str, PipelineVersion]] = {}
        self._lock = threading.Lock()
    
    async def save(self, version: PipelineVersion) -> None:
        with self._lock:
            if version.pipeline_name not in self._versions:
                self._versions[version.pipeline_name] = {}
            self._versions[version.pipeline_name][str(version.version)] = version
    
    async def get(self, pipeline_name: str, version: str) -> Optional[PipelineVersion]:
        with self._lock:
            if pipeline_name not in self._versions:
                return None
            return self._versions[pipeline_name].get(version)
    
    async def get_latest(self, pipeline_name: str, status: Optional[VersionStatus] = None) -> Optional[PipelineVersion]:
        with self._lock:
            if pipeline_name not in self._versions:
                return None
            
            versions = list(self._versions[pipeline_name].values())
            if status:
                versions = [v for v in versions if v.status == status]
            
            if not versions:
                return None
            
            return max(versions, key=lambda v: v.version)
    
    async def list_versions(self, pipeline_name: str) -> List[PipelineVersion]:
        with self._lock:
            if pipeline_name not in self._versions:
                return []
            return sorted(
                self._versions[pipeline_name].values(),
                key=lambda v: v.version,
                reverse=True,
            )
    
    async def delete(self, pipeline_name: str, version: str) -> bool:
        with self._lock:
            if pipeline_name in self._versions:
                if version in self._versions[pipeline_name]:
                    del self._versions[pipeline_name][version]
                    return True
            return False


class VersionComparator:
    """Compares two pipeline versions to detect changes."""
    
    def compare(
        self,
        old_version: PipelineVersion,
        new_version: PipelineVersion,
    ) -> List[VersionChange]:
        """
        Compare two versions and return list of changes.
        
        Args:
            old_version: Previous version
            new_version: New version
            
        Returns:
            List of detected changes
        """
        changes = []
        
        old_def = old_version.definition
        new_def = new_version.definition
        
        # Compare operators
        old_ops = self._get_operators(old_def)
        new_ops = self._get_operators(new_def)
        
        old_op_names = set(old_ops.keys())
        new_op_names = set(new_ops.keys())
        
        # Added operators
        for name in new_op_names - old_op_names:
            changes.append(VersionChange(
                change_type=ChangeType.OPERATOR_ADDED,
                description=f"Operator '{name}' added",
                path=f"operators.{name}",
                new_value=new_ops[name],
                breaking=False,
            ))
        
        # Removed operators
        for name in old_op_names - new_op_names:
            changes.append(VersionChange(
                change_type=ChangeType.OPERATOR_REMOVED,
                description=f"Operator '{name}' removed",
                path=f"operators.{name}",
                old_value=old_ops[name],
                breaking=True,
            ))
        
        # Modified operators
        for name in old_op_names & new_op_names:
            op_changes = self._compare_operators(
                old_ops[name],
                new_ops[name],
                f"operators.{name}",
            )
            changes.extend(op_changes)
        
        # Compare config
        config_changes = self._compare_config(
            old_def.get("config", {}),
            new_def.get("config", {}),
        )
        changes.extend(config_changes)
        
        return changes
    
    def _get_operators(self, definition: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract operators from definition."""
        operators = {}
        nodes = definition.get("nodes", [])
        
        for node in nodes:
            node_id = node.get("id", node.get("operator_name", "unknown"))
            operators[node_id] = node
        
        return operators
    
    def _compare_operators(
        self,
        old_op: Dict,
        new_op: Dict,
        path: str,
    ) -> List[VersionChange]:
        """Compare two operator definitions."""
        changes = []
        
        # Compare operator type
        if old_op.get("operator_name") != new_op.get("operator_name"):
            changes.append(VersionChange(
                change_type=ChangeType.OPERATOR_MODIFIED,
                description=f"Operator type changed",
                path=f"{path}.operator_name",
                old_value=old_op.get("operator_name"),
                new_value=new_op.get("operator_name"),
                breaking=True,
            ))
        
        # Compare params
        old_params = old_op.get("params", {})
        new_params = new_op.get("params", {})
        
        all_keys = set(old_params.keys()) | set(new_params.keys())
        
        for key in all_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            
            if old_val != new_val:
                changes.append(VersionChange(
                    change_type=ChangeType.PARAM_CHANGED,
                    description=f"Parameter '{key}' changed",
                    path=f"{path}.params.{key}",
                    old_value=old_val,
                    new_value=new_val,
                    breaking=False,
                ))
        
        return changes
    
    def _compare_config(
        self,
        old_config: Dict,
        new_config: Dict,
    ) -> List[VersionChange]:
        """Compare pipeline configurations."""
        changes = []
        
        # Lane change is breaking
        if old_config.get("lane") != new_config.get("lane"):
            changes.append(VersionChange(
                change_type=ChangeType.SCHEMA_CHANGED,
                description="Pipeline lane changed",
                path="config.lane",
                old_value=old_config.get("lane"),
                new_value=new_config.get("lane"),
                breaking=True,
            ))
        
        return changes
    
    def has_breaking_changes(self, changes: List[VersionChange]) -> bool:
        """Check if any changes are breaking."""
        return any(c.breaking for c in changes)


class PipelineVersionManager:
    """
    Manages pipeline versions with support for:
    - Version creation and tracking
    - Schema evolution
    - A/B testing (traffic splitting)
    - Rollback
    
    Example:
        manager = PipelineVersionManager()
        
        # Create new version
        v1 = await manager.create_version(
            pipeline_name="temperature_alerts",
            definition={...},
            created_by="admin",
        )
        
        # Activate version
        await manager.activate_version("temperature_alerts", "1.0.0")
        
        # Get active version
        active = await manager.get_active_version("temperature_alerts")
        
        # Rollback
        await manager.rollback("temperature_alerts", "1.0.0")
    """
    
    def __init__(
        self,
        store: Optional[VersionStore] = None,
    ):
        """
        Initialize version manager.
        
        Args:
            store: Version storage backend (defaults to in-memory)
        """
        self.store = store or InMemoryVersionStore()
        self.comparator = VersionComparator()
        
        # A/B testing configuration
        self._traffic_splits: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    async def create_version(
        self,
        pipeline_name: str,
        definition: Dict[str, Any],
        version: Optional[str] = None,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineVersion:
        """
        Create a new pipeline version.
        
        Args:
            pipeline_name: Name of the pipeline
            definition: Pipeline definition
            version: Version string (auto-incremented if not provided)
            created_by: Creator identifier
            metadata: Additional metadata
            
        Returns:
            Created PipelineVersion
        """
        # Get latest version for comparison
        latest = await self.store.get_latest(pipeline_name)
        
        # Determine version number
        if version:
            sem_version = SemanticVersion.parse(version)
        elif latest:
            # Auto-increment based on changes
            changes = []
            if latest:
                temp_version = PipelineVersion(
                    pipeline_name=pipeline_name,
                    version=latest.version,
                    definition=definition,
                )
                changes = self.comparator.compare(latest, temp_version)
            
            if self.comparator.has_breaking_changes(changes):
                sem_version = latest.version.bump_major()
            elif changes:
                sem_version = latest.version.bump_minor()
            else:
                sem_version = latest.version.bump_patch()
        else:
            sem_version = SemanticVersion(1, 0, 0)
        
        # Create version
        new_version = PipelineVersion(
            pipeline_name=pipeline_name,
            version=sem_version,
            definition=definition,
            created_by=created_by,
            status=VersionStatus.DRAFT,
            parent_version=str(latest.version) if latest else None,
            metadata=metadata or {},
        )
        
        # Compute changelog
        if latest:
            new_version.changelog = self.comparator.compare(latest, new_version)
        
        await self.store.save(new_version)
        
        logger.info(
            f"Created pipeline version {new_version.version_id} "
            f"(changes: {len(new_version.changelog)})"
        )
        
        return new_version
    
    async def get_version(
        self,
        pipeline_name: str,
        version: str,
    ) -> Optional[PipelineVersion]:
        """Get a specific pipeline version."""
        return await self.store.get(pipeline_name, version)
    
    async def get_active_version(
        self,
        pipeline_name: str,
    ) -> Optional[PipelineVersion]:
        """Get the active version of a pipeline."""
        return await self.store.get_latest(pipeline_name, VersionStatus.ACTIVE)
    
    async def get_latest_version(
        self,
        pipeline_name: str,
    ) -> Optional[PipelineVersion]:
        """Get the latest version regardless of status."""
        return await self.store.get_latest(pipeline_name)
    
    async def list_versions(
        self,
        pipeline_name: str,
    ) -> List[PipelineVersion]:
        """List all versions of a pipeline."""
        return await self.store.list_versions(pipeline_name)
    
    async def activate_version(
        self,
        pipeline_name: str,
        version: str,
    ) -> bool:
        """
        Activate a pipeline version (make it the current active version).
        
        Args:
            pipeline_name: Name of the pipeline
            version: Version to activate
            
        Returns:
            True if successful
        """
        # Get the version to activate
        target = await self.store.get(pipeline_name, version)
        if not target:
            logger.error(f"Version {pipeline_name}@{version} not found")
            return False
        
        # Deactivate current active version
        current_active = await self.store.get_latest(pipeline_name, VersionStatus.ACTIVE)
        if current_active and str(current_active.version) != version:
            current_active.status = VersionStatus.DEPRECATED
            await self.store.save(current_active)
        
        # Activate target version
        target.status = VersionStatus.ACTIVE
        await self.store.save(target)
        
        logger.info(f"Activated pipeline version {target.version_id}")
        return True
    
    async def deprecate_version(
        self,
        pipeline_name: str,
        version: str,
    ) -> bool:
        """Mark a version as deprecated."""
        target = await self.store.get(pipeline_name, version)
        if not target:
            return False
        
        target.status = VersionStatus.DEPRECATED
        await self.store.save(target)
        return True
    
    async def archive_version(
        self,
        pipeline_name: str,
        version: str,
    ) -> bool:
        """Archive a version (soft delete)."""
        target = await self.store.get(pipeline_name, version)
        if not target:
            return False
        
        target.status = VersionStatus.ARCHIVED
        await self.store.save(target)
        return True
    
    async def rollback(
        self,
        pipeline_name: str,
        target_version: str,
    ) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            pipeline_name: Name of the pipeline
            target_version: Version to rollback to
            
        Returns:
            True if successful
        """
        target = await self.store.get(pipeline_name, target_version)
        if not target:
            logger.error(f"Rollback target {pipeline_name}@{target_version} not found")
            return False
        
        # Create a new version based on the rollback target
        new_version = await self.create_version(
            pipeline_name=pipeline_name,
            definition=copy.deepcopy(target.definition),
            created_by="system:rollback",
            metadata={"rolled_back_from": target_version},
        )
        
        # Activate the new version
        await self.activate_version(pipeline_name, str(new_version.version))
        
        logger.info(
            f"Rolled back {pipeline_name} to {target_version} "
            f"(new version: {new_version.version})"
        )
        
        return True
    
    # -------------------------------------------------------------------------
    # A/B Testing
    # -------------------------------------------------------------------------
    
    def set_traffic_split(
        self,
        pipeline_name: str,
        splits: Dict[str, float],
    ) -> None:
        """
        Configure traffic split for A/B testing.
        
        Args:
            pipeline_name: Name of the pipeline
            splits: Dict of version -> percentage (must sum to 1.0)
            
        Example:
            manager.set_traffic_split("my_pipeline", {
                "1.0.0": 0.8,  # 80% traffic
                "2.0.0": 0.2,  # 20% traffic
            })
        """
        total = sum(splits.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Traffic splits must sum to 1.0, got {total}")
        
        with self._lock:
            self._traffic_splits[pipeline_name] = splits
        
        logger.info(f"Set traffic split for {pipeline_name}: {splits}")
    
    def get_version_for_request(
        self,
        pipeline_name: str,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get version to use for a request based on traffic split.
        
        Uses consistent hashing so same request_id always gets same version.
        
        Args:
            pipeline_name: Name of the pipeline
            request_id: Unique request identifier (for consistent routing)
            
        Returns:
            Version string or None if no split configured
        """
        with self._lock:
            splits = self._traffic_splits.get(pipeline_name)
        
        if not splits:
            return None
        
        # Use hash for consistent routing
        if request_id:
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            normalized = (hash_val % 1000) / 1000.0
        else:
            import random
            normalized = random.random()
        
        # Find version based on cumulative probability
        cumulative = 0.0
        for version, percentage in sorted(splits.items()):
            cumulative += percentage
            if normalized <= cumulative:
                return version
        
        # Fallback to first version
        return next(iter(splits.keys()))
    
    def clear_traffic_split(self, pipeline_name: str) -> None:
        """Remove traffic split configuration."""
        with self._lock:
            self._traffic_splits.pop(pipeline_name, None)
    
    # -------------------------------------------------------------------------
    # Compatibility Checking
    # -------------------------------------------------------------------------
    
    async def check_compatibility(
        self,
        pipeline_name: str,
        old_version: str,
        new_version: str,
    ) -> Tuple[bool, List[VersionChange]]:
        """
        Check if two versions are compatible.
        
        Returns:
            Tuple of (is_compatible, list of changes)
        """
        old = await self.store.get(pipeline_name, old_version)
        new = await self.store.get(pipeline_name, new_version)
        
        if not old or not new:
            return False, []
        
        changes = self.comparator.compare(old, new)
        is_compatible = not self.comparator.has_breaking_changes(changes)
        
        return is_compatible, changes


# -------------------------------------------------------------------------
# Global Instance
# -------------------------------------------------------------------------

_version_manager: Optional[PipelineVersionManager] = None


def get_version_manager() -> PipelineVersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = PipelineVersionManager()
    return _version_manager


def set_version_manager(manager: PipelineVersionManager) -> None:
    """Set the global version manager instance."""
    global _version_manager
    _version_manager = manager
