from src.tools.sec_tools import (
    get_company_info,
    get_latest_annual_revenue,
    get_latest_annual_net_income,
    get_latest_total_assets,
)


ticker = "MSFT"


print("\nCOMPANY INFO")
print(
    get_company_info.invoke(
        {"ticker": ticker}
    )
)


print("\nREVENUE")
print(
    get_latest_annual_revenue.invoke(
        {"ticker": ticker}
    )
)


print("\nNET INCOME")
print(
    get_latest_annual_net_income.invoke(
        {"ticker": ticker}
    )
)


print("\nTOTAL ASSETS")
print(
    get_latest_total_assets.invoke(
        {"ticker": ticker}
    )
)