import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agents.financial_agent import model_with_tools, tool_map


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden_set():
    examples = []

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


if __name__ == "__main__":
    golden_set = load_golden_set()

    print(f"Loaded {len(golden_set)} golden-set examples.\n")

    tool_correct_count = 0

    value_correct_count = 0
    value_test_count = 0

    fiscal_year_correct_count = 0
    fiscal_year_test_count = 0

    currency_correct_count = 0
    currency_test_count = 0

    company_info_correct_count = 0
    company_info_test_count = 0

    for example in golden_set:
        print(f"\nRunning: {example['id']}")

        response = model_with_tools.invoke(
            example["question"]
        )

        predicted_tool = None
        tool_result = None

        if response.tool_calls:
            predicted_tool = response.tool_calls[0]["name"]

        expected_tool = example["expected_tool"]

        tool_correct = predicted_tool == expected_tool

        if tool_correct:
            tool_correct_count += 1

        value_correct = None
        actual_value = None

        fiscal_year_correct = None
        actual_fiscal_year = None

        currency_correct = None
        actual_currency = None

        company_info_correct = None
        actual_company = None
        actual_cik = None

        if (
            response.tool_calls
            and predicted_tool in tool_map
        ):
            tool_call = response.tool_calls[0]

            tool_result = tool_map[
                predicted_tool
            ].invoke(
                tool_call["args"]
            )

            if "expected_value" in example:
                value_test_count += 1

                if (
                    isinstance(tool_result, dict)
                    and "revenue" in tool_result
                ):
                    actual_value = tool_result["revenue"]

                elif (
                    isinstance(tool_result, dict)
                    and "net_income" in tool_result
                ):
                    actual_value = tool_result["net_income"]

                elif (
                    isinstance(tool_result, dict)
                    and "total_assets" in tool_result
                ):
                    actual_value = tool_result["total_assets"]

                value_correct = (
                    actual_value
                    == example["expected_value"]
                )

                if value_correct:
                    value_correct_count += 1

            if "expected_fiscal_year" in example:
                fiscal_year_test_count += 1

                if isinstance(tool_result, dict):
                    actual_fiscal_year = tool_result.get(
                        "fiscal_year"
                    )

                fiscal_year_correct = (
                    actual_fiscal_year
                    == example["expected_fiscal_year"]
                )

                if fiscal_year_correct:
                    fiscal_year_correct_count += 1

            if "expected_currency" in example:
                currency_test_count += 1

                if isinstance(tool_result, dict):
                    actual_currency = tool_result.get(
                        "currency"
                    )

                currency_correct = (
                    actual_currency
                    == example["expected_currency"]
                )

                if currency_correct:
                    currency_correct_count += 1

            if (
                "expected_company" in example
                and "expected_cik" in example
            ):
                company_info_test_count += 1

                if isinstance(tool_result, dict):
                    actual_company = tool_result.get("name")
                    actual_cik = tool_result.get("cik")

                company_info_correct = (
                    actual_company
                    == example["expected_company"]
                    and actual_cik
                    == example["expected_cik"]
                )

                if company_info_correct:
                    company_info_correct_count += 1

        print(f"ID: {example['id']}")
        print(f"Expected tool: {expected_tool}")
        print(f"Predicted tool: {predicted_tool}")
        print(f"Tool correct: {tool_correct}")

        if "expected_value" in example:
            print(
                f"Expected value: "
                f"{example['expected_value']}"
            )
            print(f"Actual value: {actual_value}")
            print(f"Value correct: {value_correct}")

        if "expected_fiscal_year" in example:
            print(
                f"Expected fiscal year: "
                f"{example['expected_fiscal_year']}"
            )
            print(
                f"Actual fiscal year: "
                f"{actual_fiscal_year}"
            )
            print(
                f"Fiscal year correct: "
                f"{fiscal_year_correct}"
            )

        if "expected_currency" in example:
            print(
                f"Expected currency: "
                f"{example['expected_currency']}"
            )
            print(
                f"Actual currency: "
                f"{actual_currency}"
            )
            print(
                f"Currency correct: "
                f"{currency_correct}"
            )

        if (
            "expected_company" in example
            and "expected_cik" in example
        ):
            print(
                f"Expected company: "
                f"{example['expected_company']}"
            )
            print(
                f"Actual company: "
                f"{actual_company}"
            )
            print(
                f"Expected CIK: "
                f"{example['expected_cik']}"
            )
            print(
                f"Actual CIK: "
                f"{actual_cik}"
            )
            print(
                f"Company info correct: "
                f"{company_info_correct}"
            )

        print("-" * 60)
        time.sleep(1)

    tool_accuracy = (
        tool_correct_count
        / len(golden_set)
    )

    print(
        f"\nTool selection accuracy: "
        f"{tool_accuracy:.2%}"
    )

    if value_test_count > 0:
        value_accuracy = (
            value_correct_count
            / value_test_count
        )

        print(
            f"Value prediction accuracy: "
            f"{value_accuracy:.2%}"
        )

    if fiscal_year_test_count > 0:
        fiscal_year_accuracy = (
            fiscal_year_correct_count
            / fiscal_year_test_count
        )

        print(
            f"Fiscal year accuracy: "
            f"{fiscal_year_accuracy:.2%}"
        )

    if currency_test_count > 0:
        currency_accuracy = (
            currency_correct_count
            / currency_test_count
        )

        print(
            f"Currency accuracy: "
            f"{currency_accuracy:.2%}"
        )

    if company_info_test_count > 0:
        company_info_accuracy = (
            company_info_correct_count
            / company_info_test_count
        )

        print(
            f"Company info accuracy: "
            f"{company_info_accuracy:.2%}"
        )