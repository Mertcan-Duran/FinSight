import json
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.agents.financial_agent import (
    run_financial_agent,
)


# ------------------------------------------------------
# Load RAG golden set
# ------------------------------------------------------

GOLDEN_SET_PATH = (
    Path(__file__).parent
    / "rag_golden_set.jsonl"
)


def load_golden_set():
    cases = []

    with open(
        GOLDEN_SET_PATH,
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            cases.append(
                json.loads(line)
            )

    return cases


TEST_CASES = load_golden_set()

print(
    f"Loaded {len(TEST_CASES)} "
    f"RAG golden-set examples."
)


# ------------------------------------------------------
# Evaluation
# ------------------------------------------------------

total_cases = 0
passed_cases = 0
total_content_checks = 0
passed_content_checks = 0


for case in TEST_CASES:
    print("\n" + "=" * 80)
    print(f"Running: {case['id']}")
    print("=" * 80)

    answer = run_financial_agent(
        case["question"]
    )

    print("\nANSWER:")
    print(answer)

    answer_lower = answer.lower()

    # --------------------------------------------------
    # Source checks
    # --------------------------------------------------

    has_accession = (
        case["expected_accession"]
        in answer
    )

    has_source = (
        case["expected_source"]
        in answer
    )

    has_sources_section = (
        "sources" in answer_lower
    )

    # --------------------------------------------------
    # Content checks
    # --------------------------------------------------

    content_results = []

    for required_terms in case[
        "content_checks"
    ]:
        check = all(
            term.lower() in answer_lower
            for term in required_terms
        )

        content_results.append(
            check
        )

    content_matches = sum(
        content_results
    )

    content_check_count = len(
        content_results
    )

    content_coverage = (
        content_matches
        / content_check_count
        if content_check_count > 0
        else 0
    )

    total_content_checks += (
        content_check_count
    )

    passed_content_checks += (
        content_matches
    )

    # --------------------------------------------------
    # Case pass rule
    # --------------------------------------------------

    case_passed = (
        has_accession
        and has_source
        and has_sources_section
        and (
            content_matches
            >= case[
                "min_content_matches"
            ]
        )
    )

    total_cases += 1

    if case_passed:
        passed_cases += 1

    # --------------------------------------------------
    # Case output
    # --------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"Has exact accession: "
        f"{has_accession}"
    )

    print(
        f"Has exact SEC source: "
        f"{has_source}"
    )

    print(
        f"Has Sources section: "
        f"{has_sources_section}"
    )

    print(
        f"Content checks passed: "
        f"{content_matches}/"
        f"{content_check_count}"
    )

    print(
        f"Content coverage: "
        f"{content_coverage:.2%}"
    )

    print(
        f"Minimum required matches: "
        f"{case['min_content_matches']}"
    )

    print(
        f"Case passed: "
        f"{case_passed}"
    )


# ------------------------------------------------------
# Overall summary
# ------------------------------------------------------

case_pass_rate = (
    passed_cases
    / total_cases
    if total_cases > 0
    else 0
)

overall_content_coverage = (
    passed_content_checks
    / total_content_checks
    if total_content_checks > 0
    else 0
)


print("\n" + "=" * 80)
print("RAG EVALUATION SUMMARY")
print("=" * 80)

print(
    f"Passed cases: "
    f"{passed_cases}/{total_cases}"
)

print(
    f"Case pass rate: "
    f"{case_pass_rate:.2%}"
)

print(
    f"Overall content coverage: "
    f"{overall_content_coverage:.2%}"
)