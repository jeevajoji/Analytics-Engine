"""
Pack Registry Module.

Central registry for installed and available packs.
Supports marketplace integration.
"""

import os
import json
import logging
import hashlib
import threading
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod


from analytics_engine.packs.models import (
    Pack,
    PackManifest,
    PackType,
    PackVersion,
    PackStatus,
)
from analytics_engine.packs.loader import PackLoader, PackManager


logger = logging.getLogger(__name__)


@dataclass
class PackInfo:
    """Summary information about a pack."""
    pack_id: str
    name: str
    version: PackVersion
    pack_type: PackType
    description: str = ""
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    status: PackStatus = PackStatus.PUBLISHED
    downloads: int = 0
    rating: float = 0.0
    source: str = "local"  # local, marketplace, git
    source_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "version": str(self.version),
            "pack_type": self.pack_type.value,
            "description": self.description,
            "authors": self.authors,
            "keywords": self.keywords,
            "status": self.status.value,
            "downloads": self.downloads,
            "rating": self.rating,
            "source": self.source,
            "source_url": self.source_url,
        }
    
    @classmethod
    def from_manifest(cls, manifest: PackManifest, source: str = "local") -> "PackInfo":
        return cls(
            pack_id=manifest.pack_id,
            name=manifest.name,
            version=manifest.version,
            pack_type=manifest.pack_type,
            description=manifest.description,
            authors=[a.name for a in manifest.authors],
            keywords=manifest.keywords,
            status=manifest.status,
            source=source,
        )


class PackCatalog(ABC):
    """Abstract base for pack catalogs (local, marketplace, etc.)."""
    
    @abstractmethod
    async def search(
        self,
        query: Optional[str] = None,
        pack_type: Optional[PackType] = None,
        keywords: Optional[List[str]] = None,
    ) -> List[PackInfo]:
        """Search for packs."""
        pass
    
    @abstractmethod
    async def get_info(self, pack_id: str) -> Optional[PackInfo]:
        """Get pack info by ID."""
        pass
    
    @abstractmethod
    async def get_versions(self, pack_id: str) -> List[PackVersion]:
        """Get available versions for a pack."""
        pass


class LocalPackCatalog(PackCatalog):
    """Catalog for locally installed packs."""
    
    def __init__(self, packs_directory: str):
        self.packs_directory = Path(packs_directory)
        self._cache: Dict[str, PackInfo] = {}
        self._lock = threading.Lock()
    
    async def refresh(self) -> None:
        """Refresh the catalog from disk."""
        with self._lock:
            self._cache.clear()
            
            if not self.packs_directory.exists():
                return
            
            for item in self.packs_directory.iterdir():
                if item.is_dir():
                    manifest_path = item / "pack.json"
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, "r") as f:
                                data = json.load(f)
                            manifest = PackManifest.from_dict(data)
                            info = PackInfo.from_manifest(manifest, "local")
                            info.source_url = str(item)
                            self._cache[manifest.pack_id] = info
                        except Exception as e:
                            logger.warning(f"Failed to read pack {item}: {e}")
    
    async def search(
        self,
        query: Optional[str] = None,
        pack_type: Optional[PackType] = None,
        keywords: Optional[List[str]] = None,
    ) -> List[PackInfo]:
        """Search for packs."""
        with self._lock:
            results = list(self._cache.values())
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or
                   query_lower in p.description.lower() or
                   query_lower in p.pack_id.lower()
            ]
        
        if pack_type:
            results = [p for p in results if p.pack_type == pack_type]
        
        if keywords:
            keywords_lower = [k.lower() for k in keywords]
            results = [
                p for p in results
                if any(
                    k in [kw.lower() for kw in p.keywords]
                    for k in keywords_lower
                )
            ]
        
        return results
    
    async def get_info(self, pack_id: str) -> Optional[PackInfo]:
        """Get pack info by ID."""
        with self._lock:
            return self._cache.get(pack_id)
    
    async def get_versions(self, pack_id: str) -> List[PackVersion]:
        """Get available versions for a pack."""
        info = await self.get_info(pack_id)
        if info:
            return [info.version]
        return []


class MarketplaceCatalog(PackCatalog):
    """
    Catalog for marketplace packs.
    
    This is a placeholder for marketplace integration.
    In production, this would connect to a real marketplace API.
    """
    
    def __init__(self, marketplace_url: str = "https://marketplace.iop.local"):
        self.marketplace_url = marketplace_url
        self._cache: Dict[str, PackInfo] = {}
    
    async def search(
        self,
        query: Optional[str] = None,
        pack_type: Optional[PackType] = None,
        keywords: Optional[List[str]] = None,
    ) -> List[PackInfo]:
        """Search marketplace for packs."""
        # Placeholder - would make HTTP request to marketplace
        logger.info(f"Searching marketplace: query={query}, type={pack_type}")
        return []
    
    async def get_info(self, pack_id: str) -> Optional[PackInfo]:
        """Get pack info from marketplace."""
        # Placeholder
        return None
    
    async def get_versions(self, pack_id: str) -> List[PackVersion]:
        """Get available versions from marketplace."""
        # Placeholder
        return []
    
    async def download(
        self,
        pack_id: str,
        version: Optional[str] = None,
        target_directory: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download a pack from the marketplace.
        
        Args:
            pack_id: Pack identifier
            version: Version to download (latest if not specified)
            target_directory: Where to save the pack
            
        Returns:
            Path to downloaded pack or None
        """
        # Placeholder - would download pack from marketplace
        logger.info(f"Downloading pack {pack_id}@{version or 'latest'}")
        return None


class PackRegistry:
    """
    Central registry for managing packs.
    
    Features:
    - Track installed packs
    - Search local and marketplace packs
    - Install/uninstall packs
    - Version management
    
    Example:
        registry = PackRegistry("/path/to/packs")
        
        # Search packs
        packs = await registry.search("anomaly")
        
        # Install from marketplace
        await registry.install("anomaly-detection-pack", version="1.0.0")
        
        # Get installed pack
        pack = registry.get_installed("anomaly-detection-pack")
    """
    
    def __init__(
        self,
        packs_directory: Optional[str] = None,
        enable_marketplace: bool = True,
        marketplace_url: str = "https://marketplace.iop.local",
    ):
        """
        Initialize pack registry.
        
        Args:
            packs_directory: Directory for installed packs
            enable_marketplace: Enable marketplace integration
            marketplace_url: Marketplace API URL
        """
        self.packs_directory = Path(packs_directory) if packs_directory else None
        
        # Catalogs
        self.catalogs: List[PackCatalog] = []
        
        if self.packs_directory:
            self.local_catalog = LocalPackCatalog(str(self.packs_directory))
            self.catalogs.append(self.local_catalog)
        
        if enable_marketplace:
            self.marketplace_catalog = MarketplaceCatalog(marketplace_url)
            self.catalogs.append(self.marketplace_catalog)
        
        # Pack manager for loaded packs
        self.manager = PackManager(
            packs_directory=str(self.packs_directory) if self.packs_directory else None,
        )
        
        # Installation hooks
        self._pre_install_hooks: List[Callable[[PackInfo], bool]] = []
        self._post_install_hooks: List[Callable[[Pack], None]] = []
        
        self._lock = threading.Lock()
    
    async def refresh(self) -> None:
        """Refresh all catalogs."""
        if hasattr(self, "local_catalog"):
            await self.local_catalog.refresh()
    
    async def search(
        self,
        query: Optional[str] = None,
        pack_type: Optional[PackType] = None,
        keywords: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> List[PackInfo]:
        """
        Search for packs across all catalogs.
        
        Args:
            query: Search query
            pack_type: Filter by pack type
            keywords: Filter by keywords
            source: Filter by source (local, marketplace)
            
        Returns:
            List of matching PackInfo
        """
        all_results = []
        
        for catalog in self.catalogs:
            if source:
                if source == "local" and not isinstance(catalog, LocalPackCatalog):
                    continue
                if source == "marketplace" and not isinstance(catalog, MarketplaceCatalog):
                    continue
            
            results = await catalog.search(query, pack_type, keywords)
            all_results.extend(results)
        
        # Deduplicate by pack_id, preferring local
        seen = {}
        for info in all_results:
            if info.pack_id not in seen or info.source == "local":
                seen[info.pack_id] = info
        
        return list(seen.values())
    
    async def get_info(self, pack_id: str) -> Optional[PackInfo]:
        """Get pack info from any catalog."""
        for catalog in self.catalogs:
            info = await catalog.get_info(pack_id)
            if info:
                return info
        return None
    
    async def install(
        self,
        pack_id: str,
        version: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Optional[Pack]:
        """
        Install a pack.
        
        Args:
            pack_id: Pack identifier
            version: Version to install
            source_path: Local path to pack (for local install)
            
        Returns:
            Installed Pack or None
        """
        if not self.packs_directory:
            raise RuntimeError("No packs directory configured")
        
        # Get pack info
        info = await self.get_info(pack_id)
        
        # Run pre-install hooks
        for hook in self._pre_install_hooks:
            if not hook(info):
                logger.warning(f"Pre-install hook rejected pack {pack_id}")
                return None
        
        # Handle installation based on source
        if source_path:
            # Install from local path
            target_dir = self.packs_directory / pack_id
            
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_path, target_dir)
        elif hasattr(self, "marketplace_catalog"):
            # Download from marketplace
            await self.marketplace_catalog.download(
                pack_id, version, str(self.packs_directory)
            )
        else:
            logger.error(f"Cannot install pack {pack_id}: no source available")
            return None
        
        # Load the installed pack
        pack_path = self.packs_directory / pack_id
        pack = self.manager.load_pack(str(pack_path))
        
        # Run post-install hooks
        for hook in self._post_install_hooks:
            hook(pack)
        
        logger.info(f"Installed pack {pack_id} v{pack.version}")
        return pack
    
    async def uninstall(self, pack_id: str) -> bool:
        """
        Uninstall a pack.
        
        Args:
            pack_id: Pack identifier
            
        Returns:
            True if uninstalled
        """
        if not self.packs_directory:
            return False
        
        # Unload from manager
        self.manager.unload_pack(pack_id)
        
        # Remove from disk
        pack_path = self.packs_directory / pack_id
        if pack_path.exists():
            import shutil
            shutil.rmtree(pack_path)
            
            await self.refresh()
            logger.info(f"Uninstalled pack {pack_id}")
            return True
        
        return False
    
    def get_installed(self, pack_id: str) -> Optional[Pack]:
        """Get an installed and loaded pack."""
        return self.manager.get_pack(pack_id)
    
    def list_installed(self) -> List[Pack]:
        """List all installed packs."""
        return self.manager.list_packs()
    
    def add_pre_install_hook(self, hook: Callable[[PackInfo], bool]) -> None:
        """Add a pre-install hook."""
        self._pre_install_hooks.append(hook)
    
    def add_post_install_hook(self, hook: Callable[[Pack], None]) -> None:
        """Add a post-install hook."""
        self._post_install_hooks.append(hook)


# -------------------------------------------------------------------------
# Global Instance
# -------------------------------------------------------------------------

_pack_registry: Optional[PackRegistry] = None


def get_pack_registry() -> PackRegistry:
    """Get the global pack registry instance."""
    global _pack_registry
    if _pack_registry is None:
        _pack_registry = PackRegistry()
    return _pack_registry


def set_pack_registry(registry: PackRegistry) -> None:
    """Set the global pack registry instance."""
    global _pack_registry
    _pack_registry = registry


def configure_pack_registry(
    packs_directory: str,
    enable_marketplace: bool = True,
    marketplace_url: str = "https://marketplace.iop.local",
) -> PackRegistry:
    """
    Configure and set the global pack registry.
    
    Args:
        packs_directory: Directory for installed packs
        enable_marketplace: Enable marketplace integration
        marketplace_url: Marketplace API URL
        
    Returns:
        Configured PackRegistry
    """
    registry = PackRegistry(
        packs_directory=packs_directory,
        enable_marketplace=enable_marketplace,
        marketplace_url=marketplace_url,
    )
    set_pack_registry(registry)
    return registry

'''Acts as a catalog + installer for packs.
PackInfo: lightweight summary of a pack.
Catalogs:
LocalPackCatalog (reads installed packs on disk).
MarketplaceCatalog (placeholder for remote).
PackRegistry:
Search packs (local + marketplace).
Install/uninstall packs.
Get installed pack from PackManager.
Pre/post install hooks.'''