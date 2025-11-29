"""
Virtual File System abstraction.

Provides a secure, read-only interface for accessing file content
from various sources (directories, zips, URLs) without filesystem writes.
All operations are in-memory with cursor-based access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


# -----------------------------------------------------------------------------
# Binary detection
# -----------------------------------------------------------------------------

# Common binary file signatures (magic bytes)
BINARY_SIGNATURES: list[bytes] = [
    b"\x89PNG",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",  # GIF
    b"GIF89a",  # GIF
    b"PK\x03\x04",  # ZIP
    b"PK\x05\x06",  # ZIP (empty)
    b"\x1f\x8b",  # GZIP
    b"BZ",  # BZIP2
    b"\xfd7zXZ",  # XZ
    b"Rar!\x1a\x07",  # RAR
    b"\x7fELF",  # ELF
    b"MZ",  # DOS/PE executable
    b"\xca\xfe\xba\xbe",  # Mach-O
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit
    b"%PDF",  # PDF
    b"SQLite format 3",  # SQLite
]

# Extensions that are always binary
# Note: PDF, DOCX, PPTX are listed here but handled specially by the
# extraction module in freeze.py - they get extracted to text + images
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".tiff",
        ".tif",
        ".svg",
        # Audio
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
        # Video
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".webm",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Executables
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        # Documents (PDF, DOCX, PPTX extracted by docpack.extract module)
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # Databases
        ".db",
        ".sqlite",
        ".sqlite3",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # Other
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".ear",
    }
)


def is_binary_content(data: bytes, extension: str | None = None) -> bool:
    """
    Detect if content is binary.

    Uses a combination of:
    1. Extension-based detection
    2. Magic byte signatures
    3. Null byte detection
    4. High-byte ratio heuristic

    Args:
        data: Raw bytes to check
        extension: Optional file extension (e.g., ".py")

    Returns:
        True if content appears to be binary
    """
    # Extension check
    if extension and extension.lower() in BINARY_EXTENSIONS:
        return True

    # Empty files are text
    if not data:
        return False

    # Check for magic signatures
    for sig in BINARY_SIGNATURES:
        if data.startswith(sig):
            return True

    # Sample first 8KB for heuristics
    sample = data[:8192]

    # Null bytes indicate binary
    if b"\x00" in sample:
        return True

    # High ratio of non-printable bytes indicates binary
    # Allow common whitespace and printable ASCII
    non_text = sum(
        1
        for b in sample
        if b < 0x09 or (0x0D < b < 0x20) or b > 0x7E
    )

    # More than 30% non-text bytes = binary
    return (non_text / len(sample)) > 0.30


# -----------------------------------------------------------------------------
# Virtual File
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VirtualFile:
    """
    Represents a file in the virtual filesystem.

    All content access is via bytes cursor - no filesystem writes occur.
    Immutable after creation for thread safety.

    Attributes:
        path: Normalized path relative to source root (always forward slashes)
        size: Size in bytes
        _content_loader: Lazy loader for content bytes
        _cached_content: Cached content after first access
    """

    path: str
    size: int
    _content_loader: Callable[[], bytes]

    # Not part of hash/eq - mutable cache
    _cached_content: bytes | None = None

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VirtualFile):
            return NotImplemented
        return self.path == other.path

    @property
    def extension(self) -> str | None:
        """File extension including dot, or None."""
        if "." not in self.path:
            return None
        ext = "." + self.path.rsplit(".", 1)[-1]
        return ext.lower()

    @property
    def name(self) -> str:
        """Filename without directory."""
        return self.path.rsplit("/", 1)[-1]

    def read_bytes(self) -> bytes:
        """
        Read file content as bytes.

        Content is loaded lazily and cached on first access.
        Uses cursor-based access - no filesystem writes.

        Returns:
            Raw bytes content
        """
        # Use object.__setattr__ since frozen
        if self._cached_content is None:
            content = self._content_loader()
            object.__setattr__(self, "_cached_content", content)
        return self._cached_content  # type: ignore[return-value]

    def read_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """
        Read file content as text.

        Args:
            encoding: Text encoding (default utf-8)
            errors: Error handling (default replace)

        Returns:
            Decoded text content
        """
        return self.read_bytes().decode(encoding, errors=errors)

    def cursor(self) -> BytesIO:
        """
        Get a seekable cursor over the file content.

        Returns:
            BytesIO cursor positioned at start
        """
        return BytesIO(self.read_bytes())

    def sha256(self) -> str:
        """
        Compute SHA-256 hash of content.

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(self.read_bytes()).hexdigest()

    def is_binary(self) -> bool:
        """
        Detect if file is binary.

        Uses content-based heuristics plus extension matching.

        Returns:
            True if file appears to be binary
        """
        return is_binary_content(self.read_bytes(), self.extension)


# -----------------------------------------------------------------------------
# Virtual FS Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class VirtualFS(Protocol):
    """
    Protocol for virtual filesystem sources.

    Implementations provide read-only access to files from various
    sources (directories, zips, URLs) without filesystem writes.
    """

    def walk(self) -> Iterator[VirtualFile]:
        """
        Iterate over all files in the source.

        Yields:
            VirtualFile for each file in the source
        """
        ...

    def __enter__(self) -> "VirtualFS":
        """Enter context manager."""
        ...

    def __exit__(self, *args: object) -> None:
        """Exit context manager, cleanup resources."""
        ...
