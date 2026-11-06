"""PDF parsing pipeline for extracting instruction descriptions."""

from simdref.pdfparse import intel  # noqa: F401  # importing the module registers intel-sdm
from simdref.pdfparse.registry import get_pdf_source, iter_pdf_sources, register_pdf_source
from simdref.pdfparse.types import PdfDescriptionPayload, PdfEnrichmentResult, PdfSourceSpec

__all__ = [
    "PdfDescriptionPayload",
    "PdfEnrichmentResult",
    "PdfSourceSpec",
    "get_pdf_source",
    "iter_pdf_sources",
    "register_pdf_source",
]
