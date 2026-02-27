import os
from typing import List

from ..base import BaseParser
from .txt_parser import TxtParser
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .xlsx_parser import XlsxParser
from .azure_form_recognizer_pdf_parser import AzureFormRecognizerPdfParser


def get_default_parsers() -> List[BaseParser]:
    """Return the default set of file parsers.

    Currently includes parsers for:
    - .txt
    - .docx
    - .pdf (Azure Document Intelligence if available, otherwise local parser)
    - .xlsx family
    """

    parsers: list[BaseParser] = [TxtParser(), DocxParser()]

    # Prefer Azure Document Intelligence for PDF if the SDK and env vars are available.
    try:
        pdf_parser = AzureFormRecognizerPdfParser()
        parsers.append(pdf_parser)
    except Exception:
        parsers.append(PdfParser())

    parsers.append(XlsxParser())
    return parsers


def parse_file(path: str, parsers: List[BaseParser] | None = None) -> str:
    """Parse a file into plain text using the provided parsers.

    If no parser supports the file extension, this falls back to reading the
    file as UTF-8 text.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if parsers is None:
        parsers = get_default_parsers()

    for parser in parsers:
        if parser.can_handle(path):
            return parser.parse(path)

    # default: assume plain text UTF-8
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

