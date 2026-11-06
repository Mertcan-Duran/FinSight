from typing import List

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from src.model_factory import get_chat_model
from pydantic import BaseModel, Field


class CriticResult(BaseModel):
    approved: bool = Field(
        description=(
            "True only if the answer is sufficiently "
            "grounded, relevant, and source-consistent."
        )
    )

    grounded: bool = Field(
        description=(
            "Whether every factual claim is supported "
            "by the supplied evidence."
        )
    )

    relevant: bool = Field(
        description=(
            "Whether the answer directly addresses "
            "the user's question."
        )
    )

    source_consistent: bool = Field(
        description=(
            "Whether the answer preserves the meaning, "
            "relationships, scope, and qualifiers "
            "in the evidence."
        )
    )

    issues: List[str] = Field(
        default_factory=list,
        description=(
            "Specific factual or source-consistency "
            "problems found in the draft answer."
        ),
    )

    revision_instructions: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete instructions for correcting "
            "factual or source-consistency problems."
        ),
    )


critic_model = get_chat_model()


structured_critic = (
    critic_model.with_structured_output(
        CriticResult
    )
)


def review_answer(
    user_question: str,
    draft_answer: str,
    evidence: str,
) -> CriticResult:
    """
    Reviews a draft FinSight answer against supplied
    SEC evidence.

    The critic does not perform new research and does
    not rewrite the answer itself.
    """

    system_prompt = """
You are the FinSight Critic Agent.

Your job is NOT to answer the user's question.

Your job is to evaluate whether another agent's
draft answer is supported by the supplied evidence.

Evaluate ONLY factual grounding, relevance, and
preservation of source meaning.

Do NOT evaluate writing style, tone, formality,
business language, formatting preferences, or
whether the answer should be written in English.

Evaluate the draft using these rules:

1. Every factual claim must be supported by the
   supplied evidence.

2. The answer must directly address the user's
   question.

3. Do not allow unsupported causal relationships.

4. Do not allow separate evidence statements to be
   combined into a stronger claim unless the evidence
   explicitly supports that relationship.

5. Preserve important qualifiers and modal language.

   Examples include:
   will, may, could, might, can, has, has not,
   and similar terms.

6. If the evidence says "will", the draft must not
   change it to "may", "could", or "might".

7. If the evidence says "may", "could", or "might",
   the draft must not change the level of certainty.

8. Any material change to certainty, modality,
   attribution, subject, scope, or causal relationship
   must be treated as source-inconsistent.

9. Do not accept facts from general knowledge if they
   are absent from the supplied evidence.

10. Do not penalize the answer merely because it does
    not mention every supported fact.

11. If the answer contains unsupported material,
    identify the exact problem and provide a concise
    revision instruction.

12. Approve the answer only when it is grounded,
    relevant, and source-consistent.

13. If approved is False, identify the factual or
    source-consistency problem.

14. If approved is False, provide a concrete revision
    instruction.

15. Report every distinct grounding problem you
    identify.

16. Pay special attention to attribution, causality,
    subject scope, temporal wording, and modality.

17. source_consistent MUST be False if the draft:
    - strengthens uncertainty into certainty,
    - weakens certainty into uncertainty,
    - changes "will" to "may", "could", or "might",
    - changes "may", "could", or "might" into a
      materially different level of certainty,
    - changes attribution,
    - creates an unsupported causal relationship,
    - attaches an outcome to the wrong subject,
    - contradicts the evidence.

18. If an actual issue concerns attribution,
    causality, modality, scope, or preservation of
    source meaning, source_consistent MUST be False.

19. Preserve the full subject and scope of each claim.

    If the evidence states that a broader activity
    causes an outcome and lists AI as one included
    component, do not attribute that outcome to the
    AI component alone.

20. Preserve important temporal and aspect wording
    such as:
    "continue to",
    "has led to",
    "remains subject to",
    and similar expressions when removing them would
    materially change the meaning.

21. source_consistent MUST be False if the draft:
    - narrows the subject of a claim,
    - broadens the subject of a claim,
    - attributes an outcome to one component when the
      evidence attributes it to a broader activity,
    - removes important temporal wording when doing so
      changes the meaning.

22. Do not perform new research.
    Do not add new financial facts.
    Do not rewrite the answer yourself.

23. When the draft is written in a different language
    from the English evidence, evaluate meaning
    semantically rather than by exact word matching.

    For Turkish answers in particular:

    - "will continue to increase" must preserve both
      certainty and continuation, for example:
      "artırmaya devam edecek".

    - It must NOT become:
      "artırabilir",
      "artırma riski oluşturabilir",
      or another possibility-based expression.

    - "may decrease" may be expressed as:
      "azaltabilir".

    - "could increase" may be expressed as:
      "artırabilir".

24. A translated answer is source-inconsistent if its
    wording changes certainty, continuation, subject,
    scope, attribution, or causal relationship.

25. Do not reinterpret or reclassify a supported
    statement based on your own definition of "risk".

    A statement using "will" may still be relevant to
    a risk or challenge question.

    Do not reject a supported statement merely because
    its outcome is expressed with certainty.

    Example:

    Evidence:
    "will continue to increase operating costs"

    Acceptable:
    "will continue to increase operating costs"

    Not acceptable:
    "may increase operating costs"

26. When rejecting an answer because of a modal
    mismatch, ask only for restoration of the modality
    used in the evidence.

    Correct instruction:
    "Replace possibility-based wording with wording
    that preserves 'will continue to increase'."

    Incorrect instruction:
    "Do not describe the cost increase as a risk."

IMPORTANT OUTPUT RULES:

- issues must contain ONLY real factual, grounding,
  attribution, scope, causal, or modality problems.

- revision_instructions must contain ONLY corrections
  needed to resolve those real problems.

- Do NOT put positive commentary in issues.

- Do NOT create style feedback such as:
  "Use formal business language",
  "Use English",
  "Maintain critical analysis",
  "Improve clarity",
  or similar stylistic instructions.

- If the draft is fully grounded, relevant, and
  source-consistent:
  approved = True
  issues = []
  revision_instructions = []
"""

    review_request = f"""
USER QUESTION:

{user_question}


DRAFT ANSWER:

{draft_answer}


SUPPLIED EVIDENCE:

{evidence}
"""

    result = structured_critic.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=review_request
            ),
        ]
    )

    # --------------------------------------------------
    # Deterministic critic consistency
    # --------------------------------------------------

    checks_pass = (
        result.grounded
        and result.relevant
        and result.source_consistent
    )

    # If all substantive checks pass, approve the answer.
    # Ignore accidental commentary that the small model
    # may have placed in issues or revision_instructions.
    if checks_pass:
        result.approved = True
        result.issues = []
        result.revision_instructions = []

        return result

    # If any substantive check fails, reject the answer.
    result.approved = False
    # Remove critic feedback that tries to reinterpret
# whether a supported SEC statement is "a risk" or
# "a fact". The critic should only check fidelity.

    invalid_reclassification_phrases = (
        "not a risk",
        "fact (not a risk)",
        "cost increase is a fact",
        "should not be described as a risk",
    )

    result.issues = [
        issue
        for issue in result.issues
        if not any(
            phrase in issue.lower()
            for phrase in invalid_reclassification_phrases
        )
    ]

    result.revision_instructions = [
        instruction
        for instruction
        in result.revision_instructions
        if not any(
            phrase in instruction.lower()
            for phrase in invalid_reclassification_phrases
        )
    ]
    # Grounding and source consistency should fail
    # together when source meaning is not preserved.
    if not result.source_consistent:
        result.grounded = False

    if not result.grounded:
        result.source_consistent = False

    # A rejected answer must always provide actionable
    # feedback for the revision agent.
    if (
        not result.issues
        and result.revision_instructions
    ):
        result.issues = list(
            result.revision_instructions
        )

    if (
        result.issues
        and not result.revision_instructions
    ):
        result.revision_instructions = list(
            result.issues
        )

    if (
        not result.issues
        and not result.revision_instructions
    ):
        generic_issue = (
            "The draft failed one or more factual "
            "grounding, relevance, or source-consistency "
            "checks."
        )

        result.issues = [
            generic_issue
        ]

        result.revision_instructions = [
            generic_issue
        ]

    return result