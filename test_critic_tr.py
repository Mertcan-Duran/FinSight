from src.agents.critic_agent import review_answer


question = (
    "Microsoft yapay zekâ konusunda "
    "hangi risklerden bahsediyor?"
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


bad_draft = """
Microsoft yapay zekâ konusunda bahsedilen riskler:

- Yapay zekâ modelleri, donanım ve altyapı geliştirme
gibi dikey entegre yetkinliklerin genişletilmesi,
maliyet yapısını artırma, marjları azaltma ve
operasyonel risklere maruz kalma riskini oluşturabilir.

- Cloud ve yapay zekâ altyapısı ve cihazlarda yapılan
yatırımlar, operasyonel maliyetleri artırma ve
operasyonel marjları azaltma riskini oluşturabilir.
"""


result = review_answer(
    user_question=question,
    draft_answer=bad_draft,
    evidence=evidence,
)


print("Approved:", result.approved)
print("Grounded:", result.grounded)
print("Relevant:", result.relevant)
print(
    "Source consistent:",
    result.source_consistent,
)
print("Issues:", result.issues)
print(
    "Revision instructions:",
    result.revision_instructions,
)