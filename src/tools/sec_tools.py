import requests
from datetime import date
from langchain_core.tools import tool



SEC_HEADERS = {
    "User-Agent": "FinSight mertcanduran05@gmail.com"
}
def get_company_tickers():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=SEC_HEADERS)
    response.raise_for_status()
    return response.json()



def get_company_by_ticker(ticker: str):
    companies = get_company_tickers()

    ticker = ticker.upper()

    for company in companies.values():
        if company["ticker"].upper() == ticker:
            return {
                "ticker": company["ticker"],
                "name": company["title"],
                "cik": str(company["cik_str"]).zfill(10),
            }

    return None
def get_company_facts(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    response = requests.get(url, headers=SEC_HEADERS)
    response.raise_for_status()

    return response.json()
@tool
def get_latest_annual_revenue(ticker: str):
    """
    Returns the latest annual revenue reported in SEC 10-K filings
    for a US public company identified by its stock ticker.
    """
    company = get_company_by_ticker(ticker)

    if company is None:
        return None

    facts = get_company_facts(company["cik"])
    us_gaap = facts["facts"]["us-gaap"]

    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]

    candidates = []

    for tag in revenue_tags:
        if tag not in us_gaap:
            continue

        records = us_gaap[tag]["units"].get("USD", [])

        for item in records:
            if item.get("form") != "10-K":
                continue

            if item.get("fp") != "FY":
                continue

            if "start" not in item or "end" not in item:
                continue

            start = date.fromisoformat(item["start"])
            end = date.fromisoformat(item["end"])

            duration = (end - start).days

            if duration < 300:
                continue

            candidates.append({
                "item": item,
                "tag": tag,
            })

    if not candidates:
        return None

    latest = max(
        candidates,
        key=lambda candidate: (
            candidate["item"]["end"],
            candidate["item"]["filed"]
        )
    )

    item = latest["item"]

    return {
        "ticker": company["ticker"],
        "company": company["name"],
        "revenue": item["val"],
        "currency": "USD",
        "fiscal_year": item["fy"],
        "period_end": item["end"],
        "filed": item["filed"],
        "accession_number": item["accn"],
        "xbrl_tag": latest["tag"],
    }
@tool
def get_company_info(ticker: str):
    """
    Returns basic SEC company information for a US public company,
    including ticker, company name, and CIK number.
    """
    return get_company_by_ticker(ticker)
@tool
def get_latest_annual_net_income(ticker: str):
    """
    Returns the latest annual net income reported in SEC 10-K filings
    for a US public company identified by its stock ticker.
    """
    company = get_company_by_ticker(ticker)

    if company is None:
        return None

    facts = get_company_facts(company["cik"])
    us_gaap = facts["facts"]["us-gaap"]

    net_income_tags = [
        "NetIncomeLoss",
        "ProfitLoss",
    ]

    net_income_data = None
    used_tag = None

    for tag in net_income_tags:
        if tag in us_gaap:
            net_income_data = us_gaap[tag]
            used_tag = tag
            break

    if net_income_data is None:
        return None

    annual_records = []

    for item in net_income_data["units"].get("USD", []):
        if item.get("form") != "10-K":
            continue

        if item.get("fp") != "FY":
            continue

        if "start" not in item or "end" not in item:
            continue

        start = date.fromisoformat(item["start"])
        end = date.fromisoformat(item["end"])

        duration = (end - start).days

        if duration >= 300:
            annual_records.append(item)

    if not annual_records:
        return None

    latest = max(
        annual_records,
        key=lambda item: (item["end"], item["filed"])
    )

    return {
        "ticker": company["ticker"],
        "company": company["name"],
        "net_income": latest["val"],
        "currency": "USD",
        "fiscal_year": latest["fy"],
        "period_end": latest["end"],
        "filed": latest["filed"],
        "accession_number": latest["accn"],
        "xbrl_tag": used_tag,
    }

@tool
def get_latest_total_assets(ticker: str):
    """
    Returns the latest total assets reported in SEC 10-K filings
    for a US public company identified by its stock ticker.
    """
    company = get_company_by_ticker(ticker)

    if company is None:
        return None

    facts = get_company_facts(company["cik"])
    us_gaap = facts["facts"]["us-gaap"]

    asset_tags = [
        "Assets",
    ]

    asset_data = None
    used_tag = None

    for tag in asset_tags:
        if tag in us_gaap:
            asset_data = us_gaap[tag]
            used_tag = tag
            break

    if asset_data is None:
        return None

    annual_records = []

    for item in asset_data["units"].get("USD", []):
        if item.get("form") != "10-K":
            continue

        if item.get("fp") != "FY":
            continue

        if "end" not in item:
            continue

        annual_records.append(item)

    if not annual_records:
        return None

    latest = max(
        annual_records,
        key=lambda item: (item["end"], item["filed"])
    )

    return {
        "ticker": company["ticker"],
        "company": company["name"],
        "total_assets": latest["val"],
        "currency": "USD",
        "fiscal_year": latest["fy"],
        "period_end": latest["end"],
        "filed": latest["filed"],
        "accession_number": latest["accn"],
        "xbrl_tag": used_tag,
    }