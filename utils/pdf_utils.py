"""
PDF processing utilities for AR demos.

This module provides utilities for extracting text from PDF documents,
primarily used for loading refund policy documents.
"""

import pymupdf


DEFAULT_REFUND_POLICY_PATH = "./docs/Customer Support Refund Policy.pdf"


def extract_pdf_text(pdf_path: str = DEFAULT_REFUND_POLICY_PATH) -> str:
    """
    Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file (default: refund policy path)

    Returns:
        Extracted text content from the PDF

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: For other PDF processing errors
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            text_content = ""

            for page in doc:
                text = page.get_text()
                text_content += text + "\n\n"  # Add spacing between pages

            return text_content.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")
