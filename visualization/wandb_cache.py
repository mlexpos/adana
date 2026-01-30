#!/usr/bin/env python
"""
WandB API Cache Wrapper

Provides a caching layer for WandB API calls to avoid slow repeated API fetches.
Cache data is stored locally in `wandb_cache/` folder.

Usage:
    from wandb_cache import CachedWandBApi
    api = CachedWandBApi()  # Drop-in replacement for wandb.Api()
    runs = api.runs("entity/project", filters={"group": "my_group"})

    # Force refresh to bypass cache
    api = CachedWandBApi(force_refresh=True)
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Iterator


class CachedRun:
    """Wrapper for cached run data with interface similar to wandb.Run."""

    def __init__(self, run_data: Dict[str, Any], cache_dir: Path, api_getter):
        """
        Args:
            run_data: Dict with 'id', 'name', 'config', 'summary' keys
            cache_dir: Path to cache directory
            api_getter: Callable that returns wandb.Api() (lazy loaded)
        """
        self._data = run_data
        self._cache_dir = cache_dir
        self._api_getter = api_getter
        self._history_cache = None

    @property
    def id(self) -> str:
        return self._data['id']

    @property
    def name(self) -> str:
        return self._data['name']

    @property
    def config(self) -> Dict[str, Any]:
        return self._data.get('config', {})

    @property
    def summary(self) -> Dict[str, Any]:
        return self._data.get('summary', {})

    def _get_history_cache_path(self) -> Path:
        """Get path to history cache file for this run."""
        history_dir = self._cache_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir / f"{self.id}.json"

    def _load_history_cache(self) -> Optional[List[Dict]]:
        """Load history from cache if available."""
        cache_path = self._get_history_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_history_cache(self, history: List[Dict]):
        """Save history to cache."""
        cache_path = self._get_history_cache_path()
        with open(cache_path, 'w') as f:
            json.dump(history, f)

    def _fetch_and_cache_history(self, keys: Optional[List[str]] = None) -> List[Dict]:
        """Fetch history from WandB API and cache it."""
        api = self._api_getter()
        # Get the actual run object from WandB
        run = api.run(f"{self._data['entity']}/{self._data['project']}/{self.id}")

        # Fetch all history (we cache everything, filter later)
        history = list(run.scan_history())
        self._save_history_cache(history)
        return history

    def _get_history(self, keys: Optional[List[str]] = None, force_refresh: bool = False) -> List[Dict]:
        """Get history, using cache if available."""
        if self._history_cache is None and not force_refresh:
            self._history_cache = self._load_history_cache()

        if self._history_cache is None or force_refresh:
            self._history_cache = self._fetch_and_cache_history(keys)

        history = self._history_cache

        # Filter to requested keys if specified
        if keys:
            history = [
                {k: h[k] for k in keys if k in h}
                for h in history
            ]

        return history

    def scan_history(self, keys: Optional[List[str]] = None) -> Iterator[Dict]:
        """Iterate through run history, using cache if available.

        Args:
            keys: List of keys to include in history entries

        Yields:
            Dict entries from run history
        """
        history = self._get_history(keys)
        for entry in history:
            yield entry

    def history(self, keys: Optional[List[str]] = None, samples: int = 500, pandas: bool = True):
        """Get run history as pandas DataFrame or list.

        Args:
            keys: List of keys to include
            samples: Number of samples (ignored for cached data)
            pandas: If True, return pandas DataFrame

        Returns:
            pandas.DataFrame or list of dicts
        """
        history = self._get_history(keys)

        if pandas:
            import pandas as pd
            return pd.DataFrame(history)
        return history


class CachedRunsIterator:
    """Iterator over cached runs."""

    def __init__(self, runs: List[CachedRun]):
        self._runs = runs
        self._index = 0

    def __iter__(self):
        return iter(self._runs)

    def __len__(self):
        return len(self._runs)

    def __getitem__(self, index):
        return self._runs[index]


class CachedWandBApi:
    """Cached wrapper for WandB API.

    Drop-in replacement for wandb.Api() that caches run metadata and history
    to avoid slow repeated API calls.

    Usage:
        api = CachedWandBApi()
        runs = api.runs("entity/project", filters={"group": "my_group"})

        # Force refresh cache
        api = CachedWandBApi(force_refresh=True)
    """

    def __init__(self, cache_dir: str = "wandb_cache", force_refresh: bool = False):
        """
        Args:
            cache_dir: Directory to store cache files
            force_refresh: If True, bypass cache and fetch fresh data
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._force_refresh = force_refresh
        self._api = None

    def _get_api(self):
        """Lazy-load wandb.Api()."""
        if self._api is None:
            import wandb
            self._api = wandb.Api()
        return self._api

    def _get_cache_key(self, path: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key from path and filters."""
        # Create deterministic hash of filters
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        key_str = f"{path}_{filter_str}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_runs_cache_path(self, path: str, filters: Optional[Dict] = None) -> Path:
        """Get path to runs cache file."""
        runs_dir = self._cache_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Create readable filename
        path_safe = path.replace("/", "_")
        group = filters.get("group", "all") if filters else "all"
        cache_key = self._get_cache_key(path, filters)

        return runs_dir / f"{path_safe}_{group}_{cache_key}.json"

    def _load_runs_cache(self, path: str, filters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Load runs from cache if available."""
        cache_path = self._get_runs_cache_path(path, filters)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_runs_cache(self, path: str, filters: Optional[Dict], runs_data: List[Dict]):
        """Save runs to cache."""
        cache_path = self._get_runs_cache_path(path, filters)
        with open(cache_path, 'w') as f:
            json.dump(runs_data, f, indent=2)
        print(f"  Cached {len(runs_data)} runs to {cache_path}")

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert unknown types to string
            return str(obj)

    def _fetch_and_cache_runs(self, path: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Fetch runs from WandB API and cache them."""
        print(f"  Fetching runs from WandB API: {path}, filters={filters}")
        api = self._get_api()
        runs = api.runs(path, filters=filters)

        # Extract entity and project from path
        parts = path.split("/")
        entity = parts[0] if len(parts) > 0 else ""
        project = parts[1] if len(parts) > 1 else ""

        runs_data = []
        for run in runs:
            run_data = {
                'id': run.id,
                'name': run.name,
                'config': self._make_json_serializable(dict(run.config)),
                'summary': self._make_json_serializable(dict(run.summary)),
                'entity': entity,
                'project': project,
            }
            runs_data.append(run_data)

        self._save_runs_cache(path, filters, runs_data)
        return runs_data

    def runs(self, path: str, filters: Optional[Dict] = None) -> CachedRunsIterator:
        """Get runs for a project, using cache if available.

        Args:
            path: WandB path in format "entity/project"
            filters: Optional filters dict (e.g., {"group": "my_group"})

        Returns:
            CachedRunsIterator of CachedRun objects
        """
        # Check cache first (unless force_refresh)
        runs_data = None
        if not self._force_refresh:
            runs_data = self._load_runs_cache(path, filters)
            if runs_data is not None:
                print(f"  Using cached runs ({len(runs_data)} runs)")

        # Fetch from API if not cached
        if runs_data is None:
            runs_data = self._fetch_and_cache_runs(path, filters)

        # Convert to CachedRun objects
        cached_runs = [
            CachedRun(data, self._cache_dir, self._get_api)
            for data in runs_data
        ]

        return CachedRunsIterator(cached_runs)

    def run(self, path: str):
        """Get a single run by path.

        Args:
            path: WandB path in format "entity/project/run_id"

        Returns:
            wandb.Run object (not cached - use runs() for caching)
        """
        return self._get_api().run(path)
