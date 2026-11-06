from src.tools.rag_tools import search_latest_10k


result = search_latest_10k.invoke(
    {
        "ticker": "MSFT",
        "query": "risks artificial intelligence",
        "top_k": 5,
    }
)


print("Ticker:", result["ticker"])
print("Company:", result["company"])
print("Form:", result["form"])
print(
    "Number of results:",
    len(result["results"]),
)


for i, item in enumerate(
    result["results"],
    start=1,
):
    print("\n" + "=" * 80)
    print(f"RESULT {i}")

    print(
        "Chunk ID:",
        item["chunk_id"],
    )

    print(
        "Score:",
        item["retrieval_score"],
    )

    print("\nFiltered evidence:")
    print(item["text"])