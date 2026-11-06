from src.agents.financial_agent import (
    run_financial_agent,
)


question = (
    "Microsoft yapay zekâ konusunda "
    "hangi risklerden bahsediyor?"
)


print("Running FinSight agent...\n")


answer = run_financial_agent(
    question
)


print("\nFINAL ANSWER:")
print(answer)