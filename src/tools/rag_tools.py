from langchain_core.tools import tool

from src.rag.document_builder import (
    build_10k_chunks,
)
from src.rag.retriever import (
    extract_relevant_sentences,
    retrieve_chunks,
)


@tool
def search_latest_10k(
    ticker: str,
    query: str,
    top_k: int = 8,
):
    """
    Search the company's latest SEC 10-K filing for
    qualitative information.

    Use this tool for topics such as risks, strategy,
    competition, artificial intelligence, cybersecurity,
    supply chain, manufacturing, business segments,
    and management discussion.

    IMPORTANT:
    The SEC filing text is in English.

    The `query` argument MUST therefore be a concise
    English search query, even when the user's original
    question is written in another language.

    Translate or reformulate the user's intent into
    English before calling this tool.

    The final answer should still be written in the
    same language as the user's original question.
    """

    documents = build_10k_chunks(
        ticker
    )

    if not documents:
        return {
            "ticker": ticker,
            "results": [],
        }

    results = retrieve_chunks(
        query=query,
        documents=documents,
        top_k=top_k,
    )

    compact_results = []

    for item in results:
        evidence_text = (
            extract_relevant_sentences(
                text=item["text"],
                query=query,
                max_sentences=4,
            )
        )

        compact_results.append(
            {
                "chunk_id": item[
                    "chunk_id"
                ],
                "text": evidence_text,
                "retrieval_score": item[
                    "retrieval_score"
                ],
                "filing_date": item[
                    "filing_date"
                ],
                "report_date": item[
                    "report_date"
                ],
                "accession_number": item[
                    "accession_number"
                ],
                "source_url": item[
                    "source_url"
                ],
            }
        )

    return {
        "ticker": ticker,
        "company": documents[0][
            "company"
        ],
        "form": documents[0][
            "form"
        ],
        "results": compact_results,
    }