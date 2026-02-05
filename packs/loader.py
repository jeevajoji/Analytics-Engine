"""
Pack Loader Module.

Handles loading packs from various sources:
- Local directories
- Archives (ZIP)
- Remote URLs (marketplace)
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
import zipfile
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod


from analytics_engine.packs.models import (
    Pack,
    PackManifest,
    PackType,
    OperatorDefinition,
    ConnectorDefinition,
)
from analytics_engine.core.exceptions import PackLoadError


logger = logging.getLogger(__name__)


class PackSource(ABC):
    """Abstract base class for pack sources."""
    
    @abstractmethod
    def load_manifest(self) -> PackManifest:
        """Load and parse the pack manifest."""
        pass
    
    @abstractmethod
    def get_base_path(self) -> str:
        """Get the base path for the pack."""
        pass
    
    @abstractmethod
    def load_module(self, module_path: str) -> Any:
        """Load a Python module from the pack."""
        pass
    
    @abstractmethod
    def load_file(self, file_path: str) -> str:
        """Load a file's contents."""
        pass


class DirectoryPackSource(PackSource):
    """Load pack from a local directory."""
    
    MANIFEST_FILENAMES = ["pack.json", "manifest.json", "package.json"]
    
    def __init__(self, directory: str):
        self.directory = Path(directory).resolve()
        
        if not self.directory.exists():
            raise PackLoadError(f"Pack directory not found: {directory}")
        
        self._manifest_path = self._find_manifest()
        if not self._manifest_path:
            raise PackLoadError(
                f"No manifest found in {directory}. "
                f"Expected one of: {self.MANIFEST_FILENAMES}"
            )
    
    def _find_manifest(self) -> Optional[Path]:
        """Find the manifest file in the directory."""
        for filename in self.MANIFEST_FILENAMES:
            path = self.directory / filename
            if path.exists():
                return path
        return None
    
    def load_manifest(self) -> PackManifest:
        """Load and parse the pack manifest."""
        with open(self._manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PackManifest.from_dict(data)
    
    def get_base_path(self) -> str:
        """Get the base path for the pack."""
        return str(self.directory)
    
    def load_module(self, module_path: str) -> Any:
        """Load a Python module from the pack."""
        # Convert module path to file path
        parts = module_path.split(".")
        
        # Try as a package (directory with __init__.py)
        package_path = self.directory / "/".join(parts)
        if package_path.exists() and (package_path / "__init__.py").exists():
            module_file = package_path / "__init__.py"
        else:
            # Try as a module file
            module_file = self.directory / "/".join(parts[:-1]) / f"{parts[-1]}.py"
            if not module_file.exists():
                module_file = self.directory / f"{'/'.join(parts)}.py"
        
        if not module_file.exists():
            raise PackLoadError(f"Module not found: {module_path}")
        
        # Add pack directory to path if not already there
        pack_dir = str(self.directory)
        if pack_dir not in sys.path:
            sys.path.insert(0, pack_dir)
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_path, module_file)
        if spec is None or spec.loader is None:
            raise PackLoadError(f"Could not load module spec: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)
        
        return module
    
    def load_file(self, file_path: str) -> str:
        """Load a file's contents."""
        full_path = self.directory / file_path
        if not full_path.exists():
            raise PackLoadError(f"File not found: {file_path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()


class ZipPackSource(PackSource):
    """Load pack from a ZIP archive."""
    
    def __init__(self, zip_path: str, cleanup_on_del: bool = True):
        self.zip_path = Path(zip_path).resolve()
        self.cleanup_on_del = cleanup_on_del
        
        if not self.zip_path.exists():
            raise PackLoadError(f"ZIP file not found: {zip_path}")
        
        # Extract to temp directory
        self._temp_dir = tempfile.mkdtemp(prefix="ae_pack_")
        
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                zf.extractall(self._temp_dir)
        except zipfile.BadZipFile as e:
            shutil.rmtree(self._temp_dir)
            raise PackLoadError(f"Invalid ZIP file: {e}")
        
        # Create directory source from extracted content
        self._dir_source = DirectoryPackSource(self._temp_dir)
    
    def __del__(self):
        if self.cleanup_on_del and hasattr(self, "_temp_dir"):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    def load_manifest(self) -> PackManifest:
        return self._dir_source.load_manifest()
    
    def get_base_path(self) -> str:
        return self._dir_source.get_base_path()
    
    def load_module(self, module_path: str) -> Any:
        return self._dir_source.load_module(module_path)
    
    def load_file(self, file_path: str) -> str:
        return self._dir_source.load_file(file_path)


class PackLoader:
    """
    Loads packs from various sources and prepares them for use.
    
    Example:
        loader = PackLoader()
        
        # Load from directory
        pack = loader.load_from_directory("/path/to/pack")
        
        # Load from ZIP
        pack = loader.load_from_zip("/path/to/pack.zip")
        
        # Access operators
        for op_name in pack.list_operators():
            op_class = pack.get_operator(op_name)
    """
    
    def __init__(
        self,
        validate: bool = True,
        auto_register: bool = False,
    ):
        """
        Initialize pack loader.
        
        Args:
            validate: Whether to validate packs during loading
            auto_register: Whether to auto-register operators with the registry
        """
        self.validate = validate
        self.auto_register = auto_register
        
        # Callbacks for post-load processing
        self._post_load_callbacks: List[Callable[[Pack], None]] = []
    
    def add_post_load_callback(self, callback: Callable[[Pack], None]) -> None:
        """Add a callback to be called after pack loading."""
        self._post_load_callbacks.append(callback)
    
    def load_from_directory(self, directory: str) -> Pack:
        """
        Load a pack from a local directory.
        
        Args:
            directory: Path to pack directory
            
        Returns:
            Loaded Pack
        """
        source = DirectoryPackSource(directory)
        return self._load_from_source(source)
    
    def load_from_zip(self, zip_path: str) -> Pack:
        """
        Load a pack from a ZIP archive.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            Loaded Pack
        """
        source = ZipPackSource(zip_path)
        return self._load_from_source(source)
    
    def _load_from_source(self, source: PackSource) -> Pack:
        """Load pack from any source."""
        # Load manifest
        manifest = source.load_manifest()
        
        logger.info(f"Loading pack: {manifest.name} v{manifest.version}")
        
        # Validate if enabled
        if self.validate:
            from analytics_engine.packs.validator import PackValidator
            validator = PackValidator()
            is_valid, errors = validator.validate_manifest(manifest)
            
            if not is_valid:
                raise PackLoadError(
                    f"Pack validation failed: {'; '.join(errors)}"
                )
        
        # Create pack instance
        pack = Pack(
            manifest=manifest,
            base_path=source.get_base_path(),
        )
        
        # Load operators
        for op_def in manifest.operators:
            try:
                op_class = self._load_operator(source, op_def)
                pack.loaded_operators[op_def.name] = op_class
                
                # Auto-register with operator registry
                if self.auto_register:
                    self._register_operator(op_def.name, op_class)
                
            except Exception as e:
                logger.warning(f"Failed to load operator {op_def.name}: {e}")
        
        # Load connectors
        for conn_def in manifest.connectors:
            try:
                conn_class = self._load_connector(source, conn_def)
                pack.loaded_connectors[conn_def.name] = conn_class
                
            except Exception as e:
                logger.warning(f"Failed to load connector {conn_def.name}: {e}")
        
        # Load schemas
        for schema_path in manifest.schemas:
            try:
                schema_content = source.load_file(schema_path)
                schema_data = json.loads(schema_content)
                schema_name = Path(schema_path).stem
                pack.loaded_schemas[schema_name] = schema_data
                
            except Exception as e:
                logger.warning(f"Failed to load schema {schema_path}: {e}")
        
        # Run post-load callbacks
        for callback in self._post_load_callbacks:
            try:
                callback(pack)
            except Exception as e:
                logger.warning(f"Post-load callback failed: {e}")
        
        logger.info(
            f"Pack loaded: {manifest.name} "
            f"(operators: {len(pack.loaded_operators)}, "
            f"connectors: {len(pack.loaded_connectors)}, "
            f"schemas: {len(pack.loaded_schemas)})"
        )
        
        return pack
    
    def _load_operator(
        self,
        source: PackSource,
        op_def: OperatorDefinition,
    ) -> type:
        """Load an operator class from a pack."""
        module = source.load_module(op_def.module_path)
        
        if not hasattr(module, op_def.class_name):
            raise PackLoadError(
                f"Class {op_def.class_name} not found in {op_def.module_path}"
            )
        
        return getattr(module, op_def.class_name)
    
    def _load_connector(
        self,
        source: PackSource,
        conn_def: ConnectorDefinition,
    ) -> type:
        """Load a connector class from a pack."""
        module = source.load_module(conn_def.module_path)
        
        if not hasattr(module, conn_def.class_name):
            raise PackLoadError(
                f"Class {conn_def.class_name} not found in {conn_def.module_path}"
            )
        
        return getattr(module, conn_def.class_name)
    
    def _register_operator(self, name: str, operator_class: type) -> None:
        """Register an operator with the global registry."""
        try:
            from analytics_engine.core.registry import OperatorRegistry
            registry = OperatorRegistry()
            registry.register(name, operator_class)
        except Exception as e:
            logger.warning(f"Failed to register operator {name}: {e}")


class PackManager:
    """
    High-level pack management.
    
    Manages loading, installation, and lifecycle of packs.
    
    Example:
        manager = PackManager()
        
        # Load all packs from a directory
        manager.load_packs_from_directory("/path/to/packs")
        
        # Get a specific operator
        op_class = manager.get_operator("my_pack", "MyOperator")
    """
    
    def __init__(
        self,
        packs_directory: Optional[str] = None,
        auto_load: bool = False,
    ):
        """
        Initialize pack manager.
        
        Args:
            packs_directory: Directory containing packs
            auto_load: Whether to auto-load packs from directory
        """
        self.packs_directory = Path(packs_directory) if packs_directory else None
        self.loader = PackLoader(validate=True, auto_register=True)
        
        self._loaded_packs: Dict[str, Pack] = {}
        
        if auto_load and self.packs_directory:
            self.load_packs_from_directory(str(self.packs_directory))
    
    def load_pack(self, path: str) -> Pack:
        """
        Load a pack from path (directory or ZIP).
        
        Args:
            path: Path to pack
            
        Returns:
            Loaded Pack
        """
        path_obj = Path(path)
        
        if path_obj.is_dir():
            pack = self.loader.load_from_directory(path)
        elif path_obj.suffix.lower() == ".zip":
            pack = self.loader.load_from_zip(path)
        else:
            raise PackLoadError(f"Unknown pack format: {path}")
        
        self._loaded_packs[pack.pack_id] = pack
        return pack
    
    def load_packs_from_directory(self, directory: str) -> List[Pack]:
        """
        Load all packs from a directory.
        
        Args:
            directory: Directory containing pack subdirectories/zips
            
        Returns:
            List of loaded packs
        """
        packs = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Packs directory not found: {directory}")
            return packs
        
        for item in dir_path.iterdir():
            try:
                if item.is_dir():
                    # Check if it's a pack (has manifest)
                    for manifest_name in DirectoryPackSource.MANIFEST_FILENAMES:
                        if (item / manifest_name).exists():
                            pack = self.load_pack(str(item))
                            packs.append(pack)
                            break
                elif item.suffix.lower() == ".zip":
                    pack = self.load_pack(str(item))
                    packs.append(pack)
                    
            except Exception as e:
                logger.warning(f"Failed to load pack from {item}: {e}")
        
        return packs
    
    def unload_pack(self, pack_id: str) -> bool:
        """
        Unload a pack.
        
        Args:
            pack_id: Pack identifier
            
        Returns:
            True if unloaded
        """
        if pack_id in self._loaded_packs:
            del self._loaded_packs[pack_id]
            return True
        return False
    
    def get_pack(self, pack_id: str) -> Optional[Pack]:
        """Get a loaded pack by ID."""
        return self._loaded_packs.get(pack_id)
    
    def list_packs(self) -> List[Pack]:
        """List all loaded packs."""
        return list(self._loaded_packs.values())
    
    def get_operator(self, pack_id: str, operator_name: str) -> Optional[type]:
        """Get an operator class from a pack."""
        pack = self._loaded_packs.get(pack_id)
        if pack:
            return pack.get_operator(operator_name)
        return None
    
    def get_connector(self, pack_id: str, connector_name: str) -> Optional[type]:
        """Get a connector class from a pack."""
        pack = self._loaded_packs.get(pack_id)
        if pack:
            return pack.get_connector(connector_name)
        return None
    
    def search_operators(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search for operators across all loaded packs.
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of matching operators with pack info
        """
        results = []
        keyword_lower = keyword.lower()
        
        for pack in self._loaded_packs.values():
            for op_def in pack.manifest.operators:
                if (
                    keyword_lower in op_def.name.lower() or
                    keyword_lower in op_def.description.lower() or
                    any(keyword_lower in tag.lower() for tag in op_def.tags)
                ):
                    results.append({
                        "pack_id": pack.pack_id,
                        "pack_name": pack.name,
                        "operator_name": op_def.name,
                        "description": op_def.description,
                        "tags": op_def.tags,
                    })
        
        return results
