"""URI-agnostic path helpers built on ``pyarrow.fs``.

Abstracts over local filesystem and cloud object stores (S3, GCS, Azure)
so the rest of the codebase never has to branch on URI scheme.

Cloud path notes
----------------
For cloud URIs (``s3://``, ``gs://``, ``abfs://``, etc.) :func:`path_exists`
performs a real remote metadata call (e.g. ``HeadObject`` on S3).  To avoid
the per-call latency when registering many entities, existence checks are
**skipped at** :meth:`~core_lens.base.entity.BaseEntity._resolve` **time for
cloud roots** and are only performed once inside
:func:`~core_lens.aoi._validate_entity` via the schema-read path (Polars
raises a meaningful error if the file is absent).  Local paths continue to
receive eager existence checks as before.

Performance notes
-----------------
:func:`resolve_fs_and_path` is called on every Parquet scan, index build, and
schema read.  The hot path for cloud URIs previously paid
``pyarrow.fs.FileSystem.from_uri`` C++ object-construction cost on every
invocation.  This is now eliminated by a two-level cache:

1. **Per-URI cache** — :func:`_resolve_fs_and_path_cached` is wrapped with
   :func:`functools.lru_cache`.  Identical URIs (same string) return the
   cached ``(FileSystem, path)`` tuple in ~50 ns instead of ~50–200 µs.
2. **LocalFileSystem singleton** — ``pafs.LocalFileSystem()`` is allocated
   once at module import time; local-path calls reuse the same object.

Cache diagnostics::

    from core_lens.utils.paths import _resolve_fs_and_path_cached
    _resolve_fs_and_path_cached.cache_info()   # hits / misses / maxsize

Cache invalidation (tests / credential rotation)::

    _resolve_fs_and_path_cached.cache_clear()
"""

from __future__ import annotations

import functools
import pathlib

import pyarrow.fs as pafs
from loguru import logger

# URI scheme prefixes that indicate a cloud / non-local filesystem.
_CLOUD_SCHEMES = ("s3://", "gs://", "gcs://", "abfs://", "az://", "adl://", "https://")

# Module-level singletons / caches

# Singleton LocalFileSystem — avoids re-allocating a C++ object on every call
# for local paths (measured at ~1–3 µs per construction).
_LOCAL_FS: pafs.LocalFileSystem = pafs.LocalFileSystem()

# Cached cwd string — pathlib.Path.cwd() makes a getcwd() syscall each time.
# Only computed once; relative paths are rare in production (data_root is
# almost always absolute), but the cache removes the overhead entirely.
_CWD: str = str(pathlib.Path.cwd())


def is_cloud_uri(uri: str) -> bool:
    """Return ``True`` if *uri* refers to a cloud object store.

    Args:
        uri (str): Any path or URI string.

    Returns:
        bool: ``True`` for URIs starting with a recognised cloud scheme.

    """
    return any(uri.startswith(scheme) for scheme in _CLOUD_SCHEMES)


@functools.lru_cache(maxsize=256)
def _resolve_fs_and_path_cached(uri: str) -> tuple[pafs.FileSystem, str]:
    """Cached inner implementation of :func:`resolve_fs_and_path`.

    ``lru_cache`` keyed on the full URI string.  Cloud URIs for the same
    bucket/prefix always resolve to the same ``(FileSystem, normalised_path)``
    pair, so caching is safe and correct.

    The cache is intentionally **not** invalidated automatically.  If cloud
    credentials rotate during a long-running process, call
    ``_resolve_fs_and_path_cached.cache_clear()`` before the next access.

    Args:
        uri (str): A local path or cloud URI.

    Returns:
        tuple[pyarrow.fs.FileSystem, str]: Resolved filesystem and normalised path.

    """
    if is_cloud_uri(uri):
        # pyarrow.fs.FileSystem.from_uri: parses scheme + authority, builds the
        # appropriate C++ FileSystem object (S3FileSystem, GcsFileSystem, …).
        # Cost: ~50–200 µs first call; cached result returned in ~50 ns.
        fs, path = pafs.FileSystem.from_uri(uri)
        return fs, path

    # Local path — normalise to absolute using the cached singleton and cwd.
    p = pathlib.Path(uri)
    if not p.is_absolute():
        p = pathlib.Path(_CWD) / p
    return _LOCAL_FS, str(p)


def resolve_fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    """Resolve a URI to a ``(FileSystem, path)`` pair.

    Delegates to :func:`pyarrow.fs.FileSystem.from_uri` for cloud URIs and
    returns a :class:`pyarrow.fs.LocalFileSystem` for plain local paths.

    Results are cached by :func:`_resolve_fs_and_path_cached` so repeated
    calls with the same URI are effectively free (~50 ns per call after the
    first hit).

    Args:
        uri (str): A local path or cloud URI (e.g. ``s3://bucket/prefix``).

    Returns:
        tuple[pyarrow.fs.FileSystem, str]: The resolved filesystem and the
        normalised path within that filesystem.

    """
    return _resolve_fs_and_path_cached(uri)


def path_exists(uri: str) -> bool:
    """Return ``True`` if the path or object exists on the filesystem.

    Uses :meth:`pyarrow.fs.FileSystem.get_file_info` so it works for both
    local paths and cloud URIs.

    Args:
        uri (str): A local path or cloud URI.

    Returns:
        bool: ``True`` if the path exists (file or directory).

    """
    try:
        fs, path = resolve_fs_and_path(uri)
        info = fs.get_file_info(path)
        return bool(info.type != pafs.FileType.NotFound)
    except Exception as exc:  # pragma: no cover
        logger.debug("path_exists check failed for {!r}: {}", uri, exc)
        return False


def join_uri(root: str, rel: str) -> str:
    """Join *rel* onto *root*, handling both local paths and cloud URIs.

    For cloud roots the join is a simple string concatenation with ``/`` as
    separator (object-store "directories" are just key prefixes).  For local
    roots :func:`pathlib.Path.__truediv__` is used for correct OS handling.

    Args:
        root (str): The root directory path or cloud URI prefix.
        rel (str): A relative sub-path to append.

    Returns:
        str: The joined URI or absolute path string.

    """
    if is_cloud_uri(root):
        return root.rstrip("/") + "/" + rel.lstrip("/")
    return str(pathlib.Path(root) / rel)
