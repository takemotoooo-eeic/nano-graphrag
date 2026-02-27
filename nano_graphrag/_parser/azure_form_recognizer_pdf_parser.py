import os
from dataclasses import dataclass, field

from ..base import BaseParser


try:  # optional dependency
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore[import-untyped]
    from azure.ai.documentintelligence.models import (  # type: ignore[import-untyped]
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )
    from azure.core.credentials import AzureKeyCredential  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    DocumentIntelligenceClient = None  # type: ignore[assignment]
    AnalyzeDocumentRequest = None  # type: ignore[assignment]
    DocumentContentFormat = None  # type: ignore[assignment]
    AzureKeyCredential = None  # type: ignore[assignment]


@dataclass
class AzureFormRecognizerPdfParser(BaseParser):
    """Parse PDF using Azure Document Intelligence with Markdown output."""

    supported_extensions: set[str] = field(default_factory=lambda: {".pdf"})

    def _get_client(self) -> "DocumentIntelligenceClient":
        if (
            DocumentIntelligenceClient is None
            or AzureKeyCredential is None
            or AnalyzeDocumentRequest is None
            or DocumentContentFormat is None
        ):
            raise ImportError(
                "azure-ai-documentintelligence is required to use AzureFormRecognizerPdfParser. "
                "Install it with `pip install azure-ai-documentintelligence`."
            )

        # Use Azure Document Intelligence env vars
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise RuntimeError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "must be set in environment variables to use AzureFormRecognizerPdfParser."
            )

        return DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def parse(self, path: str) -> str:
        client = self._get_client()

        with open(path, "rb") as f:
            file_bytes = f.read()

        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_bytes),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()
        print(result.content)

        # result.content already contains the full document in Markdown format
        return result.content or ""

