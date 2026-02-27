from dataclasses import dataclass, field

from ..base import BaseParser


try:
    import docx  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    docx = None


@dataclass
class DocxParser(BaseParser):
    supported_extensions: set[str] = field(
        default_factory=lambda: {".docx"}
    )

    def parse(self, path: str) -> str:
        if docx is None:
            raise ImportError(
                "python-docx is required to parse .docx files. "
                "Install it with `pip install python-docx`."
            )
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs if p.text)

