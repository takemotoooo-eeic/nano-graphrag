import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..base import BaseParser

try:  # optional dependencies
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore[import-untyped]
    from azure.ai.documentintelligence.models import (  # type: ignore[import-untyped]
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )
    from azure.core.credentials import AzureKeyCredential  # type: ignore[import-untyped]
    from pdf2image import convert_from_path  # type: ignore[import-untyped]
    import google.generativeai as genai  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    DocumentIntelligenceClient = None  # type: ignore[assignment]
    AnalyzeDocumentRequest = None  # type: ignore[assignment]
    DocumentContentFormat = None  # type: ignore[assignment]
    AzureKeyCredential = None  # type: ignore[assignment]
    convert_from_path = None  # type: ignore[assignment]
    genai = None  # type: ignore[assignment]


@dataclass
class MultimodalPdfParser(BaseParser):
    """PDF parser that enriches Azure Document Intelligence output with Gemini captions for figures."""

    supported_extensions: set[str] = field(default_factory=lambda: {".pdf"})
    # v1beta で利用可能なデフォルトモデル（環境変数 GEMINI_MODEL_NAME で上書き可能）
    gemini_model_name: str = "gemini-1.5-flash-latest"
    debug: bool = True
    debug_output_root: str | None = None

    def _get_azure_client(self) -> "DocumentIntelligenceClient":
        if (
            DocumentIntelligenceClient is None
            or AzureKeyCredential is None
            or AnalyzeDocumentRequest is None
            or DocumentContentFormat is None
        ):
            raise ImportError(
                "azure-ai-documentintelligence is required to use MultimodalPdfParser. "
                "Install it with `pip install azure-ai-documentintelligence`."
            )

        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise RuntimeError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "must be set in environment variables to use MultimodalPdfParser."
            )

        return DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def _init_gemini(self):
        if genai is None:
            raise ImportError(
                "google-generativeai is required to use MultimodalPdfParser. "
                "Install it with `pip install google-generativeai`."
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set in environment variables.")
        genai.configure(api_key=api_key)

    def _caption_figure(self, image, index: int) -> str:
        """Call Gemini with the figure image and return a caption."""
        self._init_gemini()
        model_name = os.getenv("GEMINI_MODEL_NAME", self.gemini_model_name)
        model = genai.GenerativeModel(model_name)

        prompt = (
            "以下の図の内容を、日本語で分かりやすく説明してください。"
            "グラフや表の場合は、軸や凡例、傾向、重要な数値・ラベルをできるだけ具体的に述べてください。"
            "出力は Markdown の段落テキストのみとし、画像埋め込みやLaTeX、HTMLタグは使わないでください。"
        )

        try:
            response = model.generate_content([prompt, image])
        except Exception as e:  # pragma: no cover - defensive
            # モデル名や権限の問題などで失敗した場合は、例外で全体が落ちないようにメッセージだけ埋め込む
            return f"（図{index + 1} の説明生成に失敗しました: {e}）"

        return (getattr(response, "text", "") or "").strip()

    def _build_page_image_map(self, path: str, result) -> dict[int, "Image.Image"]:
        if convert_from_path is None:
            raise ImportError(
                "pdf2image is required to crop figure regions from PDF pages. "
                "Install it with `pip install pdf2image`."
            )
        # 1-based index: page_number -> PIL.Image
        images = convert_from_path(path)
        return {i + 1: img for i, img in enumerate(images)}

    def _get_debug_dir(self, pdf_path: str) -> Path:
        if not self.debug:
            return None  # type: ignore[return-value]

        if self.debug_output_root:
            base = Path(self.debug_output_root)
        else:
            pdf = Path(pdf_path)
            base = pdf.parent / f"{pdf.stem}_mm_debug"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _crop_figure_image(self, page_image, page, bounding_region):
        # bounding_region.polygon: [x1, y1, x2, y2, ...] in same units as page.width/height
        polygon: List[float] = getattr(bounding_region, "polygon", [])
        if not polygon:
            return None

        xs = polygon[0::2]
        ys = polygon[1::2]
        if not xs or not ys:
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        page_width = getattr(page, "width", None)
        page_height = getattr(page, "height", None)
        if not page_width or not page_height:
            return None

        img_w, img_h = page_image.size

        # Scale from page coordinate space to image pixel space
        left = max(0, int(img_w * (min_x / page_width)))
        right = min(img_w, int(img_w * (max_x / page_width)))
        top = max(0, int(img_h * (min_y / page_height)))
        bottom = min(img_h, int(img_h * (max_y / page_height)))

        if left >= right or top >= bottom:
            return None

        return page_image.crop((left, top, right, bottom))

    def parse(self, path: str) -> str:
        azure_client = self._get_azure_client()

        with open(path, "rb") as f:
            file_bytes = f.read()

        poller = azure_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_bytes),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()

        content: str = result.content or ""
        debug_dir: Path | None = self._get_debug_dir(path) if self.debug else None
        figures_debug: list[dict] = []
        figures = getattr(result, "figures", []) or []
        if not figures:
            if debug_dir is not None:
                (debug_dir / "parsed_markdown.md").write_text(content, encoding="utf-8")
            return content

        page_map = self._build_page_image_map(path, result)
        page_by_number = {p.page_number: p for p in getattr(result, "pages", [])}

        # Prepare replacements using spans; apply from back to front.
        replacements: list[tuple[int, int, str]] = []

        for idx, fig in enumerate(figures):
            bounding_regions = getattr(fig, "bounding_regions", []) or []
            spans = getattr(fig, "spans", []) or []
            if not bounding_regions or not spans:
                continue

            # Use first bounding region for cropping
            br = bounding_regions[0]
            page_number = getattr(br, "page_number", None)
            if (
                page_number is None
                or page_number not in page_map
                or page_number not in page_by_number
            ):
                continue

            page_image = page_map[page_number]
            page = page_by_number[page_number]
            cropped = self._crop_figure_image(page_image, page, br)
            if cropped is None:
                continue

            caption = self._caption_figure(cropped, idx)
            if not caption:
                continue

            if debug_dir is not None:
                img_name = f"page{page_number}_fig{idx + 1}.png"
                img_path = debug_dir / img_name
                cropped.save(img_path)
                figures_debug.append(
                    {
                        "index": idx,
                        "page_number": page_number,
                        "image_path": str(img_path),
                        "caption": caption,
                    }
                )

            replacement_text = f"\n\n> 図{idx + 1}: {caption}\n\n"

            for span in spans:
                start = getattr(span, "offset", None)
                length = getattr(span, "length", None)
                if start is None or length is None:
                    continue
                end = start + length
                replacements.append((start, end, replacement_text))

        if not replacements:
            if debug_dir is not None:
                (debug_dir / "parsed_markdown.md").write_text(content, encoding="utf-8")
                if figures_debug:
                    (debug_dir / "figures.json").write_text(
                        json.dumps(figures_debug, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            return content

        # Apply replacements from the back to avoid shifting offsets.
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, text in replacements:
            content = content[:start] + text + content[end:]

        if debug_dir is not None:
            (debug_dir / "parsed_markdown.md").write_text(content, encoding="utf-8")
            if figures_debug:
                (debug_dir / "figures.json").write_text(
                    json.dumps(figures_debug, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        return content


