#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nature Sites Intelligence Agent
================================

A local CLI agent that uses OpenAI function calling to query nip_agent.db
through the deterministic functions in agent_tools.py.

Expected folder layout:

    Agent DB/
    ├── site_intelligence_agent.py
    ├── agent_tools.py
    └── output/
        └── nip_agent.db

The OpenAI API key is loaded from the first existing .env file among:
    Agent DB/.env
    My Project/.env
    current working directory/.env

The model is fixed in this file. The .env is used only for OPENAI_API_KEY.

Usage:
    python site_intelligence_agent.py --check
    python site_intelligence_agent.py --question "מה ידוע על עין גדי?"
    python site_intelligence_agent.py --question "השווה בין אכזיב לקיסריה בנושא ניקיון ועומס" --trace
    python site_intelligence_agent.py --chat --trace
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

try:
    # Package import used by Streamlit: agent.site_intelligence_agent
    from .agent_tools import NIPAgentTools
except ImportError:
    # Standalone CLI fallback: python site_intelligence_agent.py
    from agent_tools import NIPAgentTools


# =============================================================================
# Fixed configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "output" / "nip_agent.db"
DEFAULT_LOG_PATH = SCRIPT_DIR / "logs" / "agent_runs.jsonl"

MODEL_NAME = "gpt-4.1"
MAX_TOOL_ROUNDS = 8
MAX_TOOL_OUTPUT_CHARS = 24_000
API_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_SAFETY_SECONDS = 1.0

AGENT_INSTRUCTIONS = """
You are the Nature Sites Intelligence Agent for Israeli nature reserves and
national parks.

Your only factual data source is the supplied local tools. Do not answer
site-specific factual questions from memory. Call the tools first.

Core rules:
1. Separate official site information from visitor-review analysis and apply
   an official-features-first hierarchy.
   - Official rules and physical/site features include dogs, water presence and
     type, permitted water entry, swimming or wading, accessibility, stroller
     suitability, family routes, parking, restrooms, picnic areas, camping,
     visitor centers, BBQ/fire rules, admission and opening information.
   - These must come from get_site_facts/filter_sites and must be the primary
     basis for feature comparisons and practical visit recommendations.
   - Visitor-review analysis is secondary. Use it when the user explicitly asks
     about visitor experience, reviews, cleanliness, crowding, staff, activities,
     value, maintenance or another review-derived topic.
   - Do not substitute review sentiment about the `water` topic for official
     information about whether water exists or whether entry is permitted.
2. Official constraints are hard constraints. Never recommend a site that
   violates an explicit user requirement such as dogs, stroller access,
   water access, BBQ, parking or region.
3. `dogs_mentioned=true` does not necessarily mean dogs are allowed. Use the
   explicit status returned by the tool.
4. Distinguish BBQ/grill from bonfires. A site may allow one and prohibit the
   other.
5. Distinguish water presence from permitted visitor water access.
   `water_present_access_unspecified` does NOT satisfy a request for water
   entry, swimming or wading.
6. Review counts are evidence volume, not visitor counts or site popularity.
   Never describe a site as more popular merely because it has more reviews.
7. When comparing review patterns:
   - `n_topic_mentions` means topic mentions;
   - `n_topic_reviews` means distinct reviews mentioning the topic;
   - report the correct measure and do not call mentions "reviews";
   - prefer smoothed_sentiment_score for ranking;
   - mention weak support when reliability_weight or mention counts are low;
   - do not overstate small differences.
8. Use review excerpts only as examples of patterns, not as proof by
   themselves. Keep excerpts brief.
9. If the user names a site ambiguously, use search_sites before proceeding.
10. If no site satisfies all constraints, say so directly. Do not silently
    relax constraints.
11. Use the user's language. Hebrew questions should receive natural Hebrew
    answers; English questions should receive English answers. In English
    answers, translate Hebrew source excerpts and do not display raw Hebrew
    sentences unless the user requests the original wording. Hebrew proper
    names may appear in parentheses for disambiguation.
12. Write like a concise assistant in a live product chat, not like a formal
    report.
    - Answer the question immediately and do not restate it.
    - For ordinary questions, aim for 50-110 words.
    - Detailed comparisons may use up to about 190 words.
    - Prefer compact tables over repeated prose when the same criteria apply to
      multiple sites.
    - Use at most one short heading.
    - Do not automatically use repetitive headings such as "Recommendation",
      "Official Information", "What Reviews Indicate", or "Limitations".
    - Keep caveats to one brief closing sentence unless uncertainty is central.

    Relevance discipline:
    - Answer only the dimensions explicitly requested by the user.
    - Do not add unrelated criteria merely because the tools returned them.
    - For a BBQ-and-picnic question, do not add dogs, water, admission,
      accessibility, parking, restrooms or camping unless requested.
    - Add one necessary distinction when it prevents misunderstanding, such as
      bonfire policy when the question is about BBQs.
    - An official-page link may be included without counting as an extra topic.

    Adaptive table rules:
    - When the answer includes two or more sites and the same one or more
      criteria apply to all of them, use a compact Markdown table by default,
      even if the user asked for a recommendation rather than an explicit
      comparison.
    - Choose the orientation dynamically:
        * if the number of sites is greater than the number of criteria, put
          sites in rows and criteria in columns;
        * if the number of criteria is greater than the number of sites, put
          criteria in rows and sites in columns;
        * if the dimensions are equal, put criteria in rows and sites in
          columns.
    - Example: five sites and two criteria -> one row per site, with Site,
      Criterion 1 and Criterion 2 as columns.
    - Example: two sites and five criteria -> one row per criterion, with
      Criterion, Site A and Site B as columns.
    - A two-site, two-criterion answer should therefore be a small 2-by-2
      comparison table, with criteria in rows.
    - Use only the requested criteria. Do not enlarge the table automatically.
    - Separate "Water setting" from "Permitted water access" only when both
      distinctions are needed to answer the user's wording.
    - Keep cells short and factual. Use "Not confirmed in the stored official
      data" when a fact is unknown.
    - If one site has an important unique fact directly relevant to the request
      that does not fit the shared table, add one short note below.
    - After the table, give only one or two sentences with the recommendation
      or conclusion.

    Broader-comparison offer:
    - If a broader comparison could help, offer it only after the concise
      answer.
    - Every offer must list the exact additional criteria before asking.
    - Example:
      "Would you like a broader comparison including accessibility, parking,
      restrooms, admission, family facilities, and visitor feedback?"
    - Never say only "Would you like a fuller comparison?" without naming the
      proposed criteria.
    - The user must be able to accept, reject, or remove specific criteria
      before the larger table is generated.

    Analytics visibility:
    - Do not show smoothed sentiment scores, reliability weights, internal
      ranking scores or evidence counts unless the user explicitly asks for
      review analytics.
    - Visitor feedback may determine ordering as a secondary tie-breaker, but
      do not discuss the scoring unless asked.
13. Include the official URL when the tools return it.
14. Do not expose internal tool names, JSON, SQL, call IDs or implementation
    details in the final answer.
15. Never describe the stored official-page data as "current", "currently",
    "up-to-date", "latest" or equivalent. For no-match results, say:
    "No site in the stored dataset satisfies all constraints." For
    time-sensitive operational details, advise checking the linked official
    page before departure.
16. Terminology must remain exact:
    - `n_topic_mentions`, `positive_mentions`, `neutral_mentions` and
      `negative_mentions` are topic mentions/segments, not reviews;
    - `n_topic_reviews` and `recent_n_reviews` are distinct reviews;
    - never call a mention a review;
    - never compare `n_topic_mentions` directly with total `n_reviews`;
    - when reporting coverage, use this structure:
      "`n_topic_reviews` of `n_reviews` reviews (`topic_review_coverage`)
      discuss the topic, producing `n_topic_mentions` topic mentions";
    - never infer an exact count by multiplying a rounded rate by a total.
17. The review taxonomy contains exactly these topic IDs:
    other, activities_and_attractions, hygiene_and_cleanliness,
    booking_and_entry, staff_service, information, crowding,
    supervision_and_enforcement, signage_and_navigation, visitor_amenities,
    infrastructure_and_maintenance, operational_efficiency, shop_and_food,
    accessibility, shade, parking, price_value, opening_hours, water,
    weather_and_temperature, insects, noise, showers_and_changing_rooms,
    animals, modesty_and_religion.
    There is NO review topic for dogs, dog policy, fire policy, BBQ policy or
    water access policy. Do not request those names from review-statistics or
    review-evidence tools. Use official facts for those policies. Use `water`
    only for visitors' water-related experience.
18. Never infer that no reviews discuss something merely because the taxonomy
    has no matching topic, and never invent a reason for absent review data.
    Say that the policy is evaluated from official information only.
19. After rank_sites returns candidates, call get_site_facts for the final
    recommended sites (normally up to three) before describing their official
    dog, water, fire or accessibility rules.
20. Natural-language constraint mapping:
    - Distinguish mandatory requirements from flexible preferences.
    - Words such as "must", "need", "with a dog", "requires", "only if" and
      "do not relax" normally indicate a hard constraint.
    - Words such as "prefer", "either", "ideally", "would like" and "view or
      access" normally indicate a flexible preference unless the user says
      otherwise.
    - In "for a family with a dog ... either water view or access", dog access
      is mandatory; water view or access is the flexible matching preference.
    - "allows dogs" normally accepts both `allowed` and
      `allowed_with_restrictions`, unless the user explicitly requires
      unrestricted access.
    - "permits water entry" normally accepts `allowed`,
      `allowed_with_restrictions`, and
      `partially_or_conditionally_allowed`.
    - `water_present_access_unspecified` never counts as permitted entry.
    - A request for "water view or water access" may include sites with an
      official water setting even when swimming or wading is prohibited, but
      the answer must state the difference clearly.
    - Never expand a flexible preference into unrelated comparison dimensions.

21. Tool-selection and ordering rules:
    - For a comparison of named sites based on features, call get_site_facts
      for every site before answering.
    - For a recommendation based mainly on official features, apply all hard
      official filters first. Reviews must never rescue a site that fails a
      requested official constraint.
    - If the user asks only "which sites qualify", return the official matches
      without pretending there is a quality ranking.
    - If the user asks for the "best", "recommended", "top" or "preferable"
      option, use review-derived experience only as a secondary ordering signal
      among the official matches.
    - Prefer review topics directly related to the request:
        family/general visit -> activities_and_attractions, visitor_amenities,
        accessibility, hygiene_and_cleanliness, crowding;
        picnic/BBQ -> visitor_amenities, hygiene_and_cleanliness, parking,
        crowding;
        water visit -> water, activities_and_attractions,
        hygiene_and_cleanliness, crowding;
        accessibility -> accessibility, signage_and_navigation,
        visitor_amenities.
    - When no directly relevant review topic exists, overall_sentiment_mean
      returned by filter_sites may be used as the secondary tie-breaker.
    - Never use review count or mention volume as a proxy for quality or
      popularity.
    - For more than four eligible sites, normally show the best three or four
      when the user asked for recommendations. If the user asked for a complete
      list, provide the complete concise list instead.
    - Use compare_sites/get_site_topic_stats/rank_sites only for this secondary
      ordering or when the user explicitly requests review analytics.
    - Do not expose the internal scoring unless asked.
    - Call get_site_facts for every final site shown in the answer.
    - Retrieved extra fields are for validation only. Do not display them unless
      they answer the user's requested dimensions.
    - When two or more final sites share the same requested criteria, use the
      adaptive table format in rule 12.
    - Use prose instead of a table only when there is a single site or when the
      candidates do not share comparable criteria.
""".strip()


# =============================================================================
# Function schemas exposed to the model
# =============================================================================

def nullable_string(description: str) -> dict[str, Any]:
    return {
        "type": ["string", "null"],
        "description": description,
    }


def nullable_boolean(description: str) -> dict[str, Any]:
    return {
        "type": ["boolean", "null"],
        "description": description,
    }


def nullable_string_array(
    description: str,
    enum: list[str] | None = None,
) -> dict[str, Any]:
    item_schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        item_schema["enum"] = enum
    return {
        "type": ["array", "null"],
        "items": item_schema,
        "description": description,
    }


DOG_STATUSES = [
    "allowed",
    "allowed_with_restrictions",
    "prohibited",
    "prohibited_by_catalog_policy",
    "unknown",
]

WATER_STATUSES = [
    "allowed",
    "allowed_with_restrictions",
    "partially_or_conditionally_allowed",
    "prohibited",
    "water_present_access_unspecified",
    "not_mentioned",
]

FIRE_STATUSES = [
    "allowed",
    "allowed_with_restrictions",
    "mixed_or_conditional",
    "prohibited",
    "prohibited_by_catalog_policy",
    "unspecified",
]

STROLLER_STATUSES = [
    "yes",
    "partial",
    "no",
    "unknown",
]


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_sites",
        "description": (
            "Resolve an incomplete, English, misspelled or ambiguous site name. "
            "Use before other tools when the intended site is uncertain."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The site name or partial name to search.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of matches.",
                },
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_site_facts",
        "description": (
            "Return official policies, facilities, operational text and overall "
            "review summary for one named site."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "site_name": {
                    "type": "string",
                    "description": "Hebrew or English site name.",
                },
            },
            "required": ["site_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "filter_sites",
        "description": (
            "Apply hard official constraints to find candidate sites. Use this "
            "before ranking when the user specifies required facilities or rules."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "region": nullable_string(
                    "Exact stored region: north, center, south, jerusalem, "
                    "or judea_samaria. Null when not constrained."
                ),
                "sub_region": nullable_string(
                    "Exact stored sub-region, or null."
                ),
                "site_type": nullable_string(
                    "Exact stored site type, or null."
                ),
                "dogs_status": nullable_string_array(
                    "Accepted dog-policy statuses, or null.",
                    DOG_STATUSES,
                ),
                "water_access_status": nullable_string_array(
                    "Accepted water-access statuses, or null.",
                    WATER_STATUSES,
                ),
                "bbq_status": nullable_string_array(
                    "Accepted BBQ/grill statuses, or null.",
                    FIRE_STATUSES,
                ),
                "bonfire_status": nullable_string_array(
                    "Accepted bonfire statuses, or null.",
                    FIRE_STATUSES,
                ),
                "stroller_access": nullable_string_array(
                    "Accepted stroller-access statuses, or null.",
                    STROLLER_STATUSES,
                ),
                "parking_available": nullable_boolean(
                    "Require parking when true; require absence when false; null ignores."
                ),
                "toilets_available": nullable_boolean(
                    "Require toilets when true; require absence when false; null ignores."
                ),
                "picnic_available": nullable_boolean(
                    "Require picnic facilities when true; null ignores."
                ),
                "camping_available": nullable_boolean(
                    "Require camping when true; null ignores."
                ),
                "visitor_center": nullable_boolean(
                    "Require a visitor center when true; null ignores."
                ),
                "family_easy_route": nullable_boolean(
                    "Require an easy family route when true; null ignores."
                ),
                "min_reviews": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Minimum number of source reviews.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 71,
                    "description": "Maximum number of candidates.",
                },
            },
            "required": [
                "region",
                "sub_region",
                "site_type",
                "dogs_status",
                "water_access_status",
                "bbq_status",
                "bonfire_status",
                "stroller_access",
                "parking_available",
                "toilets_available",
                "picnic_available",
                "camping_available",
                "visitor_center",
                "family_easy_route",
                "min_reviews",
                "limit",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_site_topic_stats",
        "description": (
            "Return review-derived statistics for one site, optionally limited "
            "to named topics such as water, cleanliness, crowding or staff."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "site_name": {
                    "type": "string",
                    "description": "Hebrew or English site name.",
                },
                "topics": nullable_string_array(
                    "Valid review topic IDs/aliases only. There is no dogs, dog_policy, fire_policy or water_access review topic. Use water for water-related visitor experience. Null returns all."
                ),
                "observed_only": {
                    "type": "boolean",
                    "description": "Exclude site-topic combinations with zero mentions.",
                },
                "min_mentions": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Minimum mentions required for each returned topic.",
                },
            },
            "required": [
                "site_name",
                "topics",
                "observed_only",
                "min_mentions",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "compare_sites",
        "description": (
            "Compare two or more sites using review statistics for selected "
            "topics. Use for explicit comparisons."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "site_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 10,
                    "description": "Sites to compare.",
                },
                "topics": nullable_string_array(
                    "Valid review topics only; use water, not water_experience or water_access. There is no dogs review topic. Null returns all."
                ),
                "min_mentions": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Minimum mentions per site-topic row.",
                },
            },
            "required": [
                "site_names",
                "topics",
                "min_mentions",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_review_evidence",
        "description": (
            "Return a few representative review segments for one site and topic. "
            "Use after statistics when examples would improve the explanation."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "site_name": {
                    "type": "string",
                    "description": "Hebrew or English site name.",
                },
                "topic": {
                    "type": "string",
                    "description": "Valid review topic ID, alias or Hebrew topic name. Operational policies such as dogs, fire and water access are not review topics.",
                },
                "sentiment": {
                    "type": ["string", "null"],
                    "enum": ["positive", "neutral", "negative", None],
                    "description": "Desired sentiment, or null for any sentiment.",
                },
                "recent_only": {
                    "type": "boolean",
                    "description": "Restrict to the recent review period.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Number of excerpts.",
                },
            },
            "required": [
                "site_name",
                "topic",
                "sentiment",
                "recent_only",
                "limit",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "rank_sites",
        "description": (
            "Rank candidate sites by weighted review topics after applying hard "
            "official constraints. Positive weights prefer higher sentiment for "
            "a topic. Use positive weights for all user-valued qualities, "
            "including cleanliness and crowding: higher sentiment means visitors "
            "experienced that aspect more positively."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "topic_weights": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Valid review topic ID, alias or Hebrew topic name. Operational policies such as dogs, fire and water access are not review topics.",
                            },
                            "weight": {
                                "type": "number",
                                "minimum": 0.1,
                                "maximum": 10,
                                "description": "Relative importance.",
                            },
                        },
                        "required": ["topic", "weight"],
                        "additionalProperties": False,
                    },
                },
                "region": nullable_string(
                    "Exact region or null."
                ),
                "sub_region": nullable_string(
                    "Exact sub-region or null."
                ),
                "site_type": nullable_string(
                    "Exact site type or null."
                ),
                "dogs_status": nullable_string_array(
                    "Accepted dog-policy statuses, or null.",
                    DOG_STATUSES,
                ),
                "water_access_status": nullable_string_array(
                    "Accepted water-access statuses, or null.",
                    WATER_STATUSES,
                ),
                "bbq_status": nullable_string_array(
                    "Accepted BBQ statuses, or null.",
                    FIRE_STATUSES,
                ),
                "bonfire_status": nullable_string_array(
                    "Accepted bonfire statuses, or null.",
                    FIRE_STATUSES,
                ),
                "stroller_access": nullable_string_array(
                    "Accepted stroller statuses, or null.",
                    STROLLER_STATUSES,
                ),
                "parking_available": nullable_boolean(
                    "Hard parking constraint or null."
                ),
                "toilets_available": nullable_boolean(
                    "Hard toilets constraint or null."
                ),
                "picnic_available": nullable_boolean(
                    "Hard picnic constraint or null."
                ),
                "camping_available": nullable_boolean(
                    "Hard camping constraint or null."
                ),
                "visitor_center": nullable_boolean(
                    "Hard visitor-center constraint or null."
                ),
                "family_easy_route": nullable_boolean(
                    "Hard easy-family-route constraint or null."
                ),
                "min_reviews": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Minimum source reviews for a candidate.",
                },
                "min_mentions_per_topic": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Minimum mentions required for every requested topic. "
                        "Use 5-10 for ordinary recommendations and 0 when exploring."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "description": "Number of ranked sites.",
                },
            },
            "required": [
                "topic_weights",
                "region",
                "sub_region",
                "site_type",
                "dogs_status",
                "water_access_status",
                "bbq_status",
                "bonfire_status",
                "stroller_access",
                "parking_available",
                "toilets_available",
                "picnic_available",
                "camping_available",
                "visitor_center",
                "family_easy_route",
                "min_reviews",
                "min_mentions_per_topic",
                "limit",
            ],
            "additionalProperties": False,
        },
    },
]


# =============================================================================
# Serialization and logging
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_for_model(
    value: Any,
    *,
    max_string_chars: int = 1_400,
    max_list_items: int = 15,
) -> Any:
    """Limit tool-output size without changing its core meaning."""
    if isinstance(value, dict):
        return {
            str(key): compact_for_model(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        trimmed = value[:max_list_items]
        result = [
            compact_for_model(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
            )
            for item in trimmed
        ]
        if len(value) > max_list_items:
            result.append(
                {
                    "_truncated_items": len(value) - max_list_items,
                }
            )
        return result

    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars].rstrip() + " …"

    return value


def serialize_tool_result(result: Any) -> str:
    compact = compact_for_model(result)
    serialized = json.dumps(
        compact,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    if len(serialized) <= MAX_TOOL_OUTPUT_CHARS:
        return serialized

    fallback = {
        "warning": "Tool result was truncated for model context.",
        "result_preview": serialized[:MAX_TOOL_OUTPUT_CHARS],
    }
    return json.dumps(
        fallback,
        ensure_ascii=False,
        allow_nan=False,
    )


def append_run_log(
    log_path: Path,
    payload: dict[str, Any],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )
            + "\n"
        )


# =============================================================================
# Agent implementation
# =============================================================================

class SiteIntelligenceAgent:
    @staticmethod
    def _emit_event(
        on_event: Callable[[str], None] | None,
        message: str,
    ) -> None:
        """Emit a safe execution update without exposing model reasoning."""
        if on_event is None:
            return

        try:
            on_event(message)
        except Exception:
            # UI callbacks must never interrupt the agent itself.
            pass

    @staticmethod
    def _friendly_join(items: list[str]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    @classmethod
    def _describe_tool_call(
        cls,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Translate a tool call into a short, non-technical activity label."""
        if name == "search_sites":
            query = str(arguments.get("query", "")).strip()
            return (
                f'Resolving the site name "{query}"...'
                if query
                else "Resolving the requested site name..."
            )

        if name == "get_site_facts":
            site_name = str(arguments.get("site_name", "")).strip()
            return (
                f"Checking official rules and facilities for {site_name}..."
                if site_name
                else "Checking official site rules and facilities..."
            )

        if name == "filter_sites":
            checks = []

            region = arguments.get("region")
            if region:
                region_labels = {
                    "north": "northern Israel",
                    "center": "central Israel",
                    "south": "southern Israel",
                    "jerusalem": "the Jerusalem area",
                    "judea_samaria": "Judea and Samaria",
                }
                checks.append(region_labels.get(str(region), str(region)))

            if arguments.get("dogs_status"):
                checks.append("dog policy")
            if arguments.get("water_access_status"):
                checks.append("water access")
            if arguments.get("bbq_status"):
                checks.append("BBQ rules")
            if arguments.get("bonfire_status"):
                checks.append("bonfire rules")
            if arguments.get("stroller_access"):
                checks.append("stroller access")
            if arguments.get("parking_available") is not None:
                checks.append("parking")
            if arguments.get("toilets_available") is not None:
                checks.append("restrooms")
            if arguments.get("picnic_available") is not None:
                checks.append("picnic facilities")
            if arguments.get("camping_available") is not None:
                checks.append("camping")
            if arguments.get("visitor_center") is not None:
                checks.append("visitor center")
            if arguments.get("family_easy_route") is not None:
                checks.append("family-friendly routes")

            detail = cls._friendly_join(checks)
            return (
                f"Filtering sites by {detail}..."
                if detail
                else "Filtering sites by the requested official requirements..."
            )

        if name == "get_site_topic_stats":
            site_name = str(arguments.get("site_name", "")).strip()
            topics = arguments.get("topics") or []
            topic_text = cls._friendly_join(
                [str(topic).replace("_", " ") for topic in topics]
            )

            if site_name and topic_text:
                return (
                    f"Reviewing visitor feedback about {topic_text} "
                    f"at {site_name}..."
                )
            if site_name:
                return f"Reviewing visitor feedback for {site_name}..."
            return "Reviewing relevant visitor-feedback patterns..."

        if name == "compare_sites":
            site_names = arguments.get("site_names") or []
            site_text = cls._friendly_join([str(site) for site in site_names])
            return (
                f"Comparing visitor-feedback patterns for {site_text}..."
                if site_text
                else "Comparing visitor-feedback patterns..."
            )

        if name == "get_review_evidence":
            site_name = str(arguments.get("site_name", "")).strip()
            topic = str(arguments.get("topic", "")).replace("_", " ").strip()
            if site_name and topic:
                return (
                    f"Finding representative visitor comments about "
                    f"{topic} at {site_name}..."
                )
            return "Finding representative visitor comments..."

        if name == "rank_sites":
            topic_items = arguments.get("topic_weights") or []
            topics = [
                str(item.get("topic", "")).replace("_", " ")
                for item in topic_items
                if isinstance(item, dict)
            ]
            topic_text = cls._friendly_join(topics)
            return (
                f"Ordering eligible sites using visitor feedback about "
                f"{topic_text}..."
                if topic_text
                else "Ordering the eligible sites using relevant visitor feedback..."
            )

        return "Checking the relevant stored data..."

    @classmethod
    def _describe_tool_result(
        cls,
        name: str,
        arguments: dict[str, Any],
        succeeded: bool,
    ) -> str:
        if not succeeded:
            return "One data check could not be completed; trying another route..."

        if name == "search_sites":
            return "Site-name check completed."
        if name == "get_site_facts":
            site_name = str(arguments.get("site_name", "")).strip()
            return (
                f"Official details retrieved for {site_name}."
                if site_name
                else "Official site details retrieved."
            )
        if name == "filter_sites":
            return "Matching sites identified."
        if name == "get_site_topic_stats":
            return "Relevant visitor-feedback statistics retrieved."
        if name == "compare_sites":
            return "Visitor-feedback comparison completed."
        if name == "get_review_evidence":
            return "Representative visitor comments retrieved."
        if name == "rank_sites":
            return "Eligible sites ordered using the selected feedback topics."

        return "Data check completed."

    def __init__(
        self,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        model: str = MODEL_NAME,
        trace: bool = False,
        log_path: str | Path | None = DEFAULT_LOG_PATH,
    ) -> None:
        self.tools = NIPAgentTools(db_path)
        self.model = model
        self.trace = trace
        self.log_path = Path(log_path) if log_path is not None else None

        load_api_key()
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
        )

        self._history: list[Any] = []
        self._tool_dispatch: dict[str, Callable[..., Any]] = {
            "search_sites": self.tools.search_sites,
            "get_site_facts": self.tools.get_site_facts,
            "filter_sites": self._call_filter_sites,
            "get_site_topic_stats": self.tools.get_site_topic_stats,
            "compare_sites": self.tools.compare_sites,
            "get_review_evidence": self.tools.get_review_evidence,
            "rank_sites": self._call_rank_sites,
        }

    def reset(self) -> None:
        self._history = []

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        """Parse OpenAI's 'Please try again in ...' guidance."""
        match = re.search(
            r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)(ms|s)",
            str(error),
            flags=re.IGNORECASE,
        )
        if not match:
            return None

        value = float(match.group(1))
        return value / 1000.0 if match.group(2).lower() == "ms" else value

    def _create_response_with_retry(self, **kwargs: Any) -> Any:
        """Retry only rate-limit failures and respect the server wait time."""
        for attempt in range(1, API_RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.responses.create(**kwargs)
            except RateLimitError as exc:
                if attempt >= API_RATE_LIMIT_RETRIES:
                    raise

                server_wait = self._retry_after_seconds(exc)
                fallback_wait = min(2 ** (attempt - 1), 30)
                wait_seconds = (
                    server_wait if server_wait is not None else fallback_wait
                )
                wait_seconds += RATE_LIMIT_SAFETY_SECONDS + random.uniform(0.0, 0.75)

                if self.trace:
                    print(
                        f"\n[rate limit] attempt {attempt}/"
                        f"{API_RATE_LIMIT_RETRIES}; sleeping "
                        f"{wait_seconds:.2f}s",
                        file=sys.stderr,
                    )

                time.sleep(wait_seconds)

    @staticmethod
    def _remove_nulls(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in arguments.items()
            if value is not None
        }

    def _call_filter_sites(self, **arguments: Any) -> Any:
        return self.tools.filter_sites(
            **self._remove_nulls(arguments)
        )

    def _call_rank_sites(
        self,
        topic_weights: list[dict[str, Any]],
        min_mentions_per_topic: int,
        limit: int,
        **constraint_arguments: Any,
    ) -> Any:
        weights: dict[str, float] = {}
        for item in topic_weights:
            topic = str(item["topic"])
            weights[topic] = (
                weights.get(topic, 0.0)
                + float(item["weight"])
            )

        constraints = self._remove_nulls(
            constraint_arguments
        )
        return self.tools.rank_sites(
            weights,
            constraints=constraints,
            min_mentions_per_topic=int(
                min_mentions_per_topic
            ),
            limit=int(limit),
        )

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        function = self._tool_dispatch.get(name)
        if function is None:
            return {
                "ok": False,
                "error": f"Unknown tool: {name}",
            }

        try:
            result = function(**arguments)
            return {
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def ask(
        self,
        question: str,
        on_event: Callable[[str], None] | None = None,
    ) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        self._history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        tool_trace: list[dict[str, Any]] = []

        self._emit_event(
            on_event,
            "Understanding the request and deciding which data to check...",
        )

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            if round_number > 1:
                self._emit_event(
                    on_event,
                    "Combining the retrieved facts and checking whether "
                    "another data lookup is needed...",
                )

            response = self._create_response_with_retry(
                model=self.model,
                instructions=AGENT_INSTRUCTIONS,
                input=self._history,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=True,
            )

            # Preserve all model outputs, including reasoning/function calls,
            # exactly as recommended for subsequent Responses API turns.
            self._history.extend(response.output)

            function_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            if not function_calls:
                self._emit_event(
                    on_event,
                    "Formatting the checked information into a concise answer...",
                )
                answer = (response.output_text or "").strip()
                if not answer:
                    raise RuntimeError(
                        "The model returned neither a final answer nor a tool call."
                    )

                if self.log_path is not None:
                    append_run_log(
                        self.log_path,
                        {
                            "timestamp_utc": utc_now_iso(),
                            "model": self.model,
                            "question": question,
                            "answer": answer,
                            "tool_trace": tool_trace,
                        },
                    )
                return answer

            if self.trace:
                print(
                    f"\n[tool round {round_number}]",
                    file=sys.stderr,
                )

            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError as exc:
                    arguments = {}
                    result = {
                        "ok": False,
                        "error_type": "JSONDecodeError",
                        "error": str(exc),
                    }
                else:
                    self._emit_event(
                        on_event,
                        self._describe_tool_call(
                            call.name,
                            arguments,
                        ),
                    )
                    result = self._execute_tool(
                        call.name,
                        arguments,
                    )
                    self._emit_event(
                        on_event,
                        self._describe_tool_result(
                            call.name,
                            arguments,
                            bool(result.get("ok", False)),
                        ),
                    )

                trace_item = {
                    "round": round_number,
                    "name": call.name,
                    "arguments": arguments,
                    "ok": result.get("ok", False),
                }
                if not result.get("ok"):
                    trace_item["error"] = result.get("error")
                tool_trace.append(trace_item)

                if self.trace:
                    print(
                        f"- {call.name}("
                        + json.dumps(
                            arguments,
                            ensure_ascii=False,
                        )
                        + ")",
                        file=sys.stderr,
                    )
                    if not result.get("ok"):
                        print(
                            f"  ERROR: {result.get('error')}",
                            file=sys.stderr,
                        )

                self._history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": serialize_tool_result(result),
                    }
                )

        raise RuntimeError(
            f"The agent exceeded {MAX_TOOL_ROUNDS} tool rounds."
        )


# =============================================================================
# Environment, checks and CLI
# =============================================================================

def find_env_file() -> Path | None:
    candidates = [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    return None


def load_api_key() -> None:
    env_file = find_env_file()
    if env_file is not None:
        load_dotenv(env_file)

    if not os.getenv("OPENAI_API_KEY"):
        searched = [
            str(SCRIPT_DIR / ".env"),
            str(SCRIPT_DIR.parent / ".env"),
            str(Path.cwd() / ".env"),
        ]
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Checked: "
            + ", ".join(searched)
        )


def installation_check(db_path: Path) -> dict[str, Any]:
    tools = NIPAgentTools(db_path)
    summary = tools.database_summary()

    expected = {
        "n_sites": 71,
        "distinct_sites": 71,
        "n_reviews": 30_442,
        "n_topic_mentions": 62_634,
        "n_topics": 25,
        "site_topic_rows": 1_775,
    }

    errors: list[str] = []
    for key, expected_value in expected.items():
        actual = int(summary.get(key, -1))
        if actual != expected_value:
            errors.append(
                f"{key}: expected {expected_value}, got {actual}"
            )

    env_file = find_env_file()
    if env_file is not None:
        load_dotenv(env_file)

    api_key_found = bool(os.getenv("OPENAI_API_KEY"))

    return {
        "database_path": str(db_path.resolve()),
        "agent_tools_import": "ok",
        "database_summary": summary,
        "api_key_found": api_key_found,
        "env_file": str(env_file) if env_file else None,
        "fixed_model": MODEL_NAME,
        "tool_count": len(TOOLS),
        "errors": errors,
        "status": (
            "READY"
            if not errors and api_key_found
            else "NOT_READY"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI agent for the NIP SQLite database."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to nip_agent.db.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print tool calls to stderr.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append runs to logs/agent_runs.jsonl.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate installation without calling OpenAI.",
    )
    mode.add_argument(
        "--question",
        type=str,
        help="Ask one question and exit.",
    )
    mode.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat session.",
    )
    return parser.parse_args()


def run_chat(agent: SiteIntelligenceAgent) -> None:
    print(
        "Nature Sites Intelligence Agent\n"
        "Commands: /reset clears conversation, /exit quits.\n"
    )

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue
        if question.casefold() in {"/exit", "exit", "quit"}:
            return
        if question.casefold() == "/reset":
            agent.reset()
            print("Conversation reset.\n")
            continue

        try:
            answer = agent.ask(question)
        except Exception as exc:
            print(
                f"\nERROR: {type(exc).__name__}: {exc}\n",
                file=sys.stderr,
            )
            continue

        print(f"\nAgent>\n{answer}\n")


def main() -> int:
    args = parse_args()

    if args.check:
        print(
            json.dumps(
                installation_check(args.db),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    agent = SiteIntelligenceAgent(
        db_path=args.db,
        trace=args.trace,
        log_path=None if args.no_log else DEFAULT_LOG_PATH,
    )

    if args.question is not None:
        print(agent.ask(args.question))
        return 0

    if args.chat:
        run_chat(agent)
        return 0

    raise AssertionError("Unreachable CLI state.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
