from baseline import run_baseline
from src.agents.financial_agent import run_financial_agent

question = "What was Apple's latest annual revenue?"

print("BASELINE:")
print(run_baseline(question))

print("\n" + "=" * 80 + "\n")

print("FINSIGHT AGENT:")
print(run_financial_agent(question))