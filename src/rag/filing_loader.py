import requests
from bs4 import BeautifulSoup

from src.tools.sec_tools import (
    SEC_HEADERS,
    get_company_by_ticker,
)


def get_company_submissions(ticker: str):
    company = get_company_by_ticker(ticker)

    if company is None:
        return None

    cik = company["cik"]

    url = (
        f"https://data.sec.gov/submissions/"
        f"CIK{cik}.json"
    )

    response = requests.get(
        url,
        headers=SEC_HEADERS,
    )

    response.raise_for_status()

    return response.json()
def get_latest_10k_metadata(ticker: str):
    submissions = get_company_submissions(ticker)

    if submissions is None:
        return None

    company = get_company_by_ticker(ticker)
    recent = submissions["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "ticker": company["ticker"],
                "company": company["name"],
                "cik": company["cik"],
                "form": form,
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            }

    return None
def download_latest_10k_html(ticker: str):
    metadata = get_latest_10k_metadata(ticker)

    if metadata is None:
        return None

    cik_without_leading_zeros = str(
        int(metadata["cik"])
    )

    accession_without_dashes = metadata[
        "accession_number"
    ].replace("-", "")

    primary_document = metadata[
        "primary_document"
    ]

    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_without_leading_zeros}/"
        f"{accession_without_dashes}/"
        f"{primary_document}"
    )

    response = requests.get(
        url,
        headers=SEC_HEADERS,
    )

    response.raise_for_status()

    return {
        "metadata": metadata,
        "url": url,
        "html": response.text,
    }
def extract_text_from_10k_html(html: str):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove normal HTML noise
    for tag in soup.find_all(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    # Remove Inline XBRL metadata
    for tag in soup.find_all(
        lambda tag: (
            tag.name
            and (
                tag.name.lower() in {
                    "ix:header",
                    "ix:hidden",
                }
                or tag.name.lower().startswith("xbrli:")
                or tag.name.lower().startswith("xbrldi:")
                or tag.name.lower().startswith("xbrldt:")
                or tag.name.lower().startswith("link:")
            )
        )
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():
        cleaned = " ".join(
            line.split()
        )

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)