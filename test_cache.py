from time import perf_counter

from src.rag.document_builder import (
    build_10k_chunks,
)


# Start from an empty cache so the test is reliable.
build_10k_chunks.cache_clear()


start = perf_counter()

first_result = build_10k_chunks(
    "MSFT"
)

first_time = perf_counter() - start


start = perf_counter()

second_result = build_10k_chunks(
    "MSFT"
)

second_time = perf_counter() - start


print(
    f"First call: "
    f"{first_time:.3f} seconds"
)

print(
    f"Second call: "
    f"{second_time:.6f} seconds"
)

print(
    "Chunks:",
    len(first_result),
)

print(
    "Cache info:",
    build_10k_chunks.cache_info(),
)