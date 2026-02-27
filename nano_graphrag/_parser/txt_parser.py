from dataclasses import dataclass, field

from ..base import BaseParser


@dataclass
class TxtParser(BaseParser):
    supported_extensions: set[str] = field(default_factory=lambda: {".txt"})

    def parse(self, path: str) -> str:
        # Assume UTF-8 text; ignore undecodable bytes to be robust against noisy files.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

