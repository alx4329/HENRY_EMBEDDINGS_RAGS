"""Adapter: cargador de documentos Markdown desde un directorio.

Implementa el port ``DocumentLoader`` leyendo archivos .md. Cada archivo se
convierte en un ``Document`` con metadatos automáticos: nombre del archivo,
ruta relativa, primera línea como título.
"""

from __future__ import annotations

from pathlib import Path

from clase3.domain.document import Document


class MarkdownDirectoryLoader:
    """Carga todos los .md de un directorio (no recursivo por defecto)."""

    def __init__(
        self,
        directory: Path,
        *,
        recursive: bool = False,
        extra_metadata: dict | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.recursive = recursive
        self.extra_metadata = extra_metadata or {}

    def load(self) -> list[Document]:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"No existe el directorio {self.directory}")

        pattern = "**/*.md" if self.recursive else "*.md"
        files = sorted(self.directory.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No hay archivos .md en {self.directory}")

        documents: list[Document] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            title = self._extract_title(content) or file_path.stem
            metadata = {
                "source": str(file_path.relative_to(self.directory)),
                "filename": file_path.name,
                "title": title,
                **self.extra_metadata,
            }
            documents.append(Document(content=content, metadata=metadata, id=file_path.stem))
        return documents

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line.lstrip("# ").strip()
        return None
