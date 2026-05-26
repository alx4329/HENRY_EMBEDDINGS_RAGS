"""Tests del MarkdownDirectoryLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from clase3.adapters.markdown_loader import MarkdownDirectoryLoader


def test_loads_all_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Alfa\n\nContenido alfa.")
    (tmp_path / "b.md").write_text("# Beta\n\nContenido beta.")
    docs = MarkdownDirectoryLoader(tmp_path).load()
    titles = {d.metadata["title"] for d in docs}
    assert titles == {"Alfa", "Beta"}
    assert all(d.content for d in docs)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MarkdownDirectoryLoader(tmp_path / "no_existe").load()


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MarkdownDirectoryLoader(tmp_path).load()


def test_extra_metadata_is_propagated(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("# Doc\n\nContenido.")
    docs = MarkdownDirectoryLoader(tmp_path, extra_metadata={"corpus": "demo"}).load()
    assert docs[0].metadata["corpus"] == "demo"
