from src.agents.critic_agent import review_answer


question = (
    "What risks does Microsoft identify "
    "regarding artificial intelligence?"
)

evidence = """
The investments we are making in cloud and AI
infrastructure and devices will continue to increase
our operating costs and may decrease our operating
margins.

Expansion of our vertically-integrated capabilities,
including developing proprietary hardware,
infrastructure, and artificial intelligence ("AI")
models, could increase our cost structure, reduce
margins, and expose us to operational risks.
"""


# Intentionally correct and grounded answer.
draft_answer = """
Microsoft identifies two AI-related risks.

First, investments in cloud and AI infrastructure
and devices may increase operating costs and
may decrease operating margins.

Second, expanding vertically-integrated capabilities,
including proprietary hardware, infrastructure,
and AI models, could increase the company's cost
structure, reduce margins, and expose it to
operational risks.
"""


result = review_answer(
    user_question=question,
    draft_answer=draft_answer,
    evidence=evidence,
)


print("Approved:", result.approved)
print("Grounded:", result.grounded)
print("Relevant:", result.relevant)

print(
    "Source consistent:",
    result.source_consistent,
)

print("\nIssues:")

for issue in result.issues:
    print("-", issue)


print("\nRevision instructions:")

for instruction in result.revision_instructions:
    print("-", instruction)