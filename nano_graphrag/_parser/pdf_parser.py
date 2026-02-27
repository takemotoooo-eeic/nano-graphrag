from dataclasses import dataclass, field

from ..base import BaseParser


try:
    from pypdf import PdfReader  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


@dataclass
class PdfParser(BaseParser):
    supported_extensions: set[str] = field(
        default_factory=lambda: {".pdf"}
    )

    def parse(self, path: str) -> str:
        if PdfReader is None:
            raise ImportError(
                "pypdf is required to parse .pdf files. "
                "Install it with `pip install pypdf`."
            )
        reader = PdfReader(path)
        texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                texts.append(text)
        return "\n".join(texts)

