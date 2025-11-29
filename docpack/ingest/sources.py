"""
Virtual filesystem source implementations.

Provides secure, read-only access to files from:
- Directories (local filesystem)
- Zip files (in-memory, no extraction to disk)
- URLs (download to temp, auto-cleanup)

All sources implement the VirtualFS protocol.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.request import urlopen
from urllib.parse import urlparse

from .vfs import VirtualFile


# -----------------------------------------------------------------------------
# Ignore patterns
# -----------------------------------------------------------------------------

# Directories to always skip
IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        "__pycache__",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        "dist",
        "build",
        "*.egg-info",
        ".venv",
        "venv",
        ".env",
        "env",
    }
)

# Files to always skip
IGNORE_FILES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        ".gitignore",
        ".gitattributes",
    }
)


def should_ignore(name: str, is_dir: bool = False) -> bool:
    """Check if a file or directory should be ignored."""
    if is_dir:
        return name in IGNORE_DIRS
    return name in IGNORE_FILES or name.endswith(".pyc")


def normalize_path(path: str) -> str:
    """
    Normalize path to forward slashes, remove leading ./

    Converts Windows backslashes and ensures consistent format.
    """
    # Convert to forward slashes
    normalized = path.replace("\\", "/")
    # Remove leading ./
    while normalized.startswith("./"):
        normalized = normalized[2:]
    # Remove leading /
    normalized = normalized.lstrip("/")
    return normalized


def is_path_safe(path: str) -> bool:
    """
    Check if path is safe (no directory traversal).

    Prevents zip slip attacks by rejecting paths that:
    - Contain .. components
    - Are absolute paths
    - Escape the root

    Args:
        path: Normalized path to check

    Returns:
        True if path is safe
    """
    # Normalize first
    normalized = normalize_path(path)

    # Reject empty paths
    if not normalized:
        return False

    # Use PurePosixPath for consistent handling
    ppath = PurePosixPath(normalized)

    # Reject absolute paths
    if ppath.is_absolute():
        return False

    # Reject paths with .. that could escape
    try:
        # Resolve will raise if path escapes
        parts = list(ppath.parts)
        if ".." in parts:
            return False
    except ValueError:
        return False

    return True


# -----------------------------------------------------------------------------
# Directory Source
# -----------------------------------------------------------------------------


class DirectorySource:
    """
    Virtual filesystem source for local directories.

    Walks the directory tree and provides VirtualFile access
    to each file. Skips common ignore patterns.

    Usage:
        with DirectorySource("./project") as source:
            for vfile in source.walk():
                print(vfile.path)
    """

    def __init__(self, root: str | Path) -> None:
        """
        Initialize directory source.

        Args:
            root: Path to root directory
        """
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")

    def __enter__(self) -> "DirectorySource":
        return self

    def __exit__(self, *args: object) -> None:
        pass  # No cleanup needed

    def walk(self) -> Iterator[VirtualFile]:
        """
        Walk directory and yield VirtualFiles.

        Yields:
            VirtualFile for each non-ignored file
        """
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Filter directories in-place to prevent descent
            dirnames[:] = [d for d in dirnames if not should_ignore(d, is_dir=True)]

            for filename in filenames:
                if should_ignore(filename):
                    continue

                full_path = Path(dirpath) / filename
                rel_path = full_path.relative_to(self.root)
                normalized = normalize_path(str(rel_path))

                try:
                    size = full_path.stat().st_size
                except OSError:
                    continue  # Skip unreadable files

                # Create loader that captures the path
                def make_loader(p: Path):  # noqa: ANN202
                    def loader() -> bytes:
                        return p.read_bytes()

                    return loader

                yield VirtualFile(
                    path=normalized,
                    size=size,
                    _content_loader=make_loader(full_path),
                )


# -----------------------------------------------------------------------------
# Zip Source
# -----------------------------------------------------------------------------


class ZipSource:
    """
    Virtual filesystem source for zip files.

    Reads zip contents entirely in memory - no extraction to disk.
    Prevents zip slip attacks by validating all paths.

    Security:
    - All paths are validated against directory traversal
    - Content is read via cursor, never written to filesystem
    - Zip is opened in read-only mode

    Usage:
        with ZipSource("archive.zip") as source:
            for vfile in source.walk():
                content = vfile.read_bytes()
    """

    def __init__(self, path: str | Path | bytes) -> None:
        """
        Initialize zip source.

        Args:
            path: Path to zip file, or raw bytes of zip content
        """
        if isinstance(path, bytes):
            # In-memory zip from bytes
            import io

            self._zip_file = zipfile.ZipFile(io.BytesIO(path), "r")
            self._owns_file = True
        else:
            self._path = Path(path)
            if not self._path.is_file():
                raise FileNotFoundError(f"Zip file not found: {self._path}")
            self._zip_file = zipfile.ZipFile(self._path, "r")
            self._owns_file = True

    def __enter__(self) -> "ZipSource":
        return self

    def __exit__(self, *args: object) -> None:
        if self._owns_file and self._zip_file:
            self._zip_file.close()

    def walk(self) -> Iterator[VirtualFile]:
        """
        Walk zip contents and yield VirtualFiles.

        Validates paths for safety and skips directories.

        Yields:
            VirtualFile for each valid file in the zip
        """
        for info in self._zip_file.infolist():
            # Skip directories
            if info.is_dir():
                continue

            # Normalize and validate path
            normalized = normalize_path(info.filename)

            # Security: reject unsafe paths
            if not is_path_safe(normalized):
                continue

            # Skip ignored files
            name = normalized.rsplit("/", 1)[-1]
            if should_ignore(name):
                continue

            # Skip files in ignored directories
            parts = normalized.split("/")
            if any(should_ignore(p, is_dir=True) for p in parts[:-1]):
                continue

            # Create loader that reads from zip
            def make_loader(zf: zipfile.ZipFile, fname: str):  # noqa: ANN202
                def loader() -> bytes:
                    return zf.read(fname)

                return loader

            yield VirtualFile(
                path=normalized,
                size=info.file_size,
                _content_loader=make_loader(self._zip_file, info.filename),
            )


# -----------------------------------------------------------------------------
# URL Source
# -----------------------------------------------------------------------------


class URLSource:
    """
    Virtual filesystem source for URLs.

    Downloads content to system temp directory and delegates to
    appropriate source (ZipSource for zips, otherwise treated as single file).

    The temp file is automatically cleaned up on context exit.

    Usage:
        with URLSource("https://example.com/repo.zip") as source:
            for vfile in source.walk():
                print(vfile.path)
    """

    def __init__(self, url: str, *, timeout: int = 30) -> None:
        """
        Initialize URL source.

        Args:
            url: URL to download
            timeout: Download timeout in seconds
        """
        self.url = url
        self.timeout = timeout
        self._temp_file: tempfile._TemporaryFileWrapper | None = None  # type: ignore[name-defined]
        self._delegate: DirectorySource | ZipSource | None = None

    def __enter__(self) -> "URLSource":
        # Parse URL for filename hint
        parsed = urlparse(self.url)
        path = parsed.path
        suffix = ""
        if "." in path.split("/")[-1]:
            suffix = "." + path.rsplit(".", 1)[-1]

        # Download to temp file
        self._temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            suffix=suffix, delete=False
        )

        try:
            with urlopen(self.url, timeout=self.timeout) as response:
                while chunk := response.read(8192):
                    self._temp_file.write(chunk)
            self._temp_file.flush()
            self._temp_file.close()

            # Determine delegate based on content
            temp_path = Path(self._temp_file.name)

            if suffix.lower() == ".zip" or self._is_zip(temp_path):
                self._delegate = ZipSource(temp_path)
                self._delegate.__enter__()
            else:
                # Single file - wrap in a simple iterator
                self._single_file_path = temp_path

        except Exception:
            self._cleanup()
            raise

        return self

    def __exit__(self, *args: object) -> None:
        if self._delegate:
            self._delegate.__exit__(*args)
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove temp file if it exists."""
        if self._temp_file:
            try:
                os.unlink(self._temp_file.name)
            except OSError:
                pass

    def _is_zip(self, path: Path) -> bool:
        """Check if file is a zip by magic bytes."""
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                return magic[:2] == b"PK"
        except OSError:
            return False

    def walk(self) -> Iterator[VirtualFile]:
        """
        Walk downloaded content.

        Delegates to ZipSource for zips, yields single file otherwise.

        Yields:
            VirtualFile for each file
        """
        if self._delegate:
            yield from self._delegate.walk()
        elif hasattr(self, "_single_file_path"):
            # Single file download
            path = self._single_file_path
            parsed = urlparse(self.url)
            name = parsed.path.rsplit("/", 1)[-1] or "downloaded"

            def loader() -> bytes:
                return path.read_bytes()

            yield VirtualFile(
                path=name,
                size=path.stat().st_size,
                _content_loader=loader,
            )
