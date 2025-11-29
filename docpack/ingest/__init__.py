"""
Docpack ingest layer.

Provides a secure virtual filesystem abstraction for ingesting content
from directories, zip files, and URLs without filesystem write operations.

Usage:
    from docpack.ingest import freeze, DirectorySource, ZipSource

    # Freeze a directory
    freeze("./my-project", "output.docpack")

    # Or use sources directly
    with DirectorySource("./my-project") as source:
        for vfile in source.walk():
            print(vfile.path, vfile.size)
"""

from .sources import DirectorySource, URLSource, ZipSource
from .vfs import VirtualFile, VirtualFS
from .freeze import freeze, detect_source

__all__ = [
    # VFS
    "VirtualFile",
    "VirtualFS",
    # Sources
    "DirectorySource",
    "ZipSource",
    "URLSource",
    # Freeze
    "freeze",
    "detect_source",
]
