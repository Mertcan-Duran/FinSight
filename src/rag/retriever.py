import math
import re
from collections import Counter


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "regarding",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}
def normalize_query(
    query: str,
) -> str:
    """
    Normalizes common financial research query terms
    before lexical retrieval.

    This helps equivalent queries retrieve similar
    evidence even when abbreviations are used.
    """

    normalized = query.strip()

    normalized = re.sub(
        r"\bAI\b",
        "artificial intelligence",
        normalized,
        flags=re.IGNORECASE,
    )

    return normalized

def tokenize(text: str):
    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    ]


def retrieve_chunks(
    query,
    documents,
    top_k=5,
):
    query = normalize_query(query)

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    tokenized_documents = [
        tokenize(document["text"])
        for document in documents
    ]

    document_count = len(documents)

    if document_count == 0:
        return []

    average_document_length = (
        sum(
            len(tokens)
            for tokens in tokenized_documents
        )
        / document_count
    )

    document_frequency = Counter()

    for tokens in tokenized_documents:
        for token in set(tokens):
            document_frequency[token] += 1

    k1 = 1.5
    b = 0.75

    scored_documents = []

    for document, tokens in zip(
        documents,
        tokenized_documents,
    ):
        token_counts = Counter(tokens)
        document_length = len(tokens)

        score = 0.0

        for query_token in query_tokens:
            frequency = token_counts.get(
                query_token,
                0,
            )

            if frequency == 0:
                continue

            df = document_frequency[
                query_token
            ]

            idf = math.log(
                1
                + (
                    document_count
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            numerator = (
                frequency
                * (k1 + 1)
            )

            denominator = (
                frequency
                + k1
                * (
                    1
                    - b
                    + b
                    * document_length
                    / average_document_length
                )
            )

            score += (
                idf
                * numerator
                / denominator
            )

        if score > 0:
            result = document.copy()
            result["retrieval_score"] = score

            scored_documents.append(
                result
            )

    scored_documents.sort(
        key=lambda item: item[
            "retrieval_score"
        ],
        reverse=True,
    )

    return scored_documents[:top_k]
def extract_relevant_sentences(
    text: str,
    query: str,
    max_sentences: int = 4,
) -> str:
    """
    Extracts the sentences in a retrieved chunk that
    are most directly related to the search query.

    This reduces the chance that unrelated statements
    located in the same large chunk are treated as
    evidence for the user's question.
    """

    query_terms = set(
        tokenize(query)
    )

    # Handle the common AI abbreviation explicitly.
    if (
        "artificial" in query_terms
        and "intelligence" in query_terms
    ):
        query_terms.add("ai")

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    scored_sentences = []

    for index, sentence in enumerate(
        sentences
    ):
        sentence_terms = set(
            tokenize(sentence)
        )

        overlap = (
            query_terms
            & sentence_terms
        )

        score = len(overlap)

        if score > 0:
            scored_sentences.append(
                (
                    score,
                    index,
                    sentence.strip(),
                )
            )

    if not scored_sentences:
        return text

    scored_sentences.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    selected = scored_sentences[
        :max_sentences
    ]

    # Restore original document order.
    selected.sort(
        key=lambda item: item[1]
    )

    return " ".join(
        item[2]
        for item in selected
    )