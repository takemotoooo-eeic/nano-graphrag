from dataclasses import dataclass, field

from ..base import BaseParser

try:
    import openpyxl  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    openpyxl = None


@dataclass
class XlsxParser(BaseParser):
    supported_extensions: set[str] = field(
        default_factory=lambda: {".xlsx", ".xlsm", ".xltx", ".xltm"}
    )

    def parse(self, path: str) -> str:
        if openpyxl is None:
            raise ImportError(
                "openpyxl is required to parse .xlsx files. "
                "Install it with `pip install openpyxl`."
            )

        workbook = openpyxl.load_workbook(path, data_only=True)
        lines: list[str] = []
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                cells = [str(cell) for cell in row if cell is not None]
                if cells:
                    lines.append("\t".join(cells))
        return "\n".join(lines)

