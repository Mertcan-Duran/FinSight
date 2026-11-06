from functools import lru_cache
from src.rag.filing_loader import (
    download_latest_10k_html,
    extract_text_from_10k_html,
)
from src.rag.chunking import chunk_text

@lru_cache(maxsize=8)
def build_10k_chunks(ticker: str):
    filing = download_latest_10k_html(ticker)

    if filing is None:
        return []

    metadata = filing["metadata"]

    text = extract_text_from_10k_html(
        filing["html"]
    )

    chunks = chunk_text(text)

    documents = []

    for chunk in chunks:
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "ticker": metadata["ticker"],
                "company": metadata["company"],
                "form": metadata["form"],
                "filing_date": metadata["filing_date"],
                "report_date": metadata["report_date"],
                "accession_number": metadata[
                    "accession_number"
                ],
                "source_url": filing["url"],
            }
        )

    return documents