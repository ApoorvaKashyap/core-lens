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
"""

from __future__ import annotations

import pathlib

import pyarrow.fs as pafs
from loguru import logger

# URI scheme prefixes that indicate a cloud / non-local filesystem.
_CLOUD_SCHEMES = ("s3://", "gs://", "gcs://", "abfs://", "az://", "adl://", "https://")


def is_cloud_uri(uri: str) -> bool:
    """Return ``True`` if *uri* refers to a cloud object store.

    Args:
        uri (str): Any path or URI string.

    Returns:
        bool: ``True`` for URIs starting with a recognised cloud scheme.
    """
    return any(uri.startswith(scheme) for scheme in _CLOUD_SCHEMES)


def resolve_fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    """Resolve a URI to a ``(FileSystem, path)`` pair.

    Delegates to :func:`pyarrow.fs.FileSystem.from_uri` for cloud URIs and
    returns a :class:`pyarrow.fs.LocalFileSystem` for plain local paths.

    Args:
        uri (str): A local path or cloud URI (e.g. ``s3://bucket/prefix``).

    Returns:
        tuple[pyarrow.fs.FileSystem, str]: The resolved filesystem and the
        normalised path within that filesystem.
    """
    if is_cloud_uri(uri):
        fs, path = pafs.FileSystem.from_uri(uri)
        return fs, path
    # Local path — normalise to absolute.
    p = pathlib.Path(uri)
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    return pafs.LocalFileSystem(), str(p)


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
