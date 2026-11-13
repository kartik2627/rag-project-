"""Pydantic schemas used for structured LLM output.

Class docstrings, field names and Field descriptions are sent to the LLM as part of
the JSON schema, so they act as prompt text. Fields are also generated in the order
they are declared, so "reasoning" fields come BEFORE the decision they justify.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Route = Literal["retrieve", "verify_claim", "direct_answer"]
Confidence = Literal["high", "medium", "low"]

MAX_SUPERSEDING_PAPERS = 3


# ── /btw side channel ─────────────────────────────────────────────────────────

class BtwRouteDecision(BaseModel):
    """Decide whether a side question needs live web data, and how to search for it."""

    reason: str = Field(
        description="One short sentence on why a web search is or is not needed. "
        "Write this first, before deciding."
    )
    needs_web_search: bool = Field(
        description="True only if the answer depends on recent or changing information "
        "(news, current prices, weather, events after your training data) or on facts you "
        "are unsure of. False if stable general knowledge is enough."
    )
    search_query: str = Field(
        description="If needs_web_search is true: a concise keyword query (3-8 words), "
        "including the year if recency matters. Otherwise an empty string."
    )


# ── Main router ───────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Routing decision for the user's latest message."""

    route: Route = Field(
        description="retrieve: needs the uploaded documents or live/current information. "
        "verify_claim: the user asks whether a claim is still true or has been superseded. "
        "direct_answer: stable general knowledge, or small talk."
    )
    standalone_query: str = Field(
        description="The latest message rewritten so it is understandable without the chat "
        "history (pronouns and references resolved). Unchanged if already self-contained. "
        "Do not answer it."
    )


# ── Relevancy grading ─────────────────────────────────────────────────────────

class RelevancyDecision(BaseModel):
    """Grade whether retrieved chunks help answer the user's question."""

    reason: str = Field(
        description="One short sentence: what matches the question, or what is missing. "
        "Write this first, before deciding."
    )
    is_relevant: bool = Field(
        description="True if the chunks meaningfully address the question, even partially. "
        "False only if they are clearly off-topic."
    )


# ── Claim verification ────────────────────────────────────────────────────────

class ClaimQueries(BaseModel):
    """Search queries prepared for fact-checking a claim."""

    claim: str = Field(description="The single factual claim to verify, stated self-contained.")
    web_query: str = Field(
        description="Web search query (max ~12 words) to find newer work that updates, "
        "challenges or supersedes the claim."
    )
    arxiv_query: str = Field(
        description="Core technical keywords likely to appear in titles/abstracts of "
        "newer arXiv papers on the same topic."
    )


class SupersedingPaper(BaseModel):
    """A newer paper that updates, challenges or replaces the claim."""

    title: str = Field(description="Title copied verbatim from the search results.")
    url: str = Field(description="URL copied verbatim from the search results (prefer arxiv.org).")
    summary: str = Field(
        description="ONE sentence on how this paper updates the claim, based only on its snippet."
    )


class ClaimVerificationResult(BaseModel):
    """Verdict on whether a claim has been superseded by newer work."""

    superseding_papers: list[SupersedingPaper] = Field(
        description=f"Up to {MAX_SUPERSEDING_PAPERS} papers from the search results that clearly "
        "supersede or update the claim. Empty list if the claim still holds or evidence is weak."
    )
    is_superseded: bool = Field(
        description="True only if the listed papers clearly improve on, contradict or replace "
        "the claim. False if the claim holds or the evidence is inconclusive."
    )
    confidence: Confidence = Field(
        description="How strongly the search results support this verdict."
    )
    verdict_summary: str = Field(
        description="1-2 plain sentences for the user, consistent with the fields above, "
        "mentioning if the evidence is inconclusive."
    )

    @model_validator(mode="after")
    def _make_consistent(self) -> "ClaimVerificationResult":
        self.superseding_papers = self.superseding_papers[:MAX_SUPERSEDING_PAPERS]
        if not self.is_superseded:
            self.superseding_papers = []
        elif not self.superseding_papers:
            self.is_superseded = False  # cannot claim "superseded" with no evidence
        return self
