#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deterministic data-access tools for the Nature Sites Intelligence Agent.

Default database location:
    <this script folder>/output/nip_agent.db

The module does not call an LLM. It exposes safe, structured functions that a
future agent or Streamlit application can call.

Examples:
    python agent_tools.py --self-test
    python agent_tools.py --summary
    python agent_tools.py --search "עין גדי"
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "output" / "nip_agent.db"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_lookup_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("־", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r'["״“”\'׳‘’`]', "", text)
    text = re.sub(r"[^0-9a-z\u0590-\u05ff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _site_lookup_variants(value: Any) -> set[str]:
    """
    Build safe lookup variants for site names.

    This removes generic English site-type suffixes and optional parenthetical
    aliases, so inputs such as "Gan HaShlosha National Park" resolve to the
    stored alias "Gan HaShlosha (Sakhne)" without weakening ambiguity checks.
    """
    raw = "" if value is None else str(value)
    raw_variants = {
        raw,
        re.sub(r"\([^)]*\)", " ", raw),
    }

    variants: set[str] = set()
    suffix_pattern = re.compile(
        r"\b(?:national\s+park|nature\s+reserve|national\s+reserve|"
        r"park|reserve)\b$"
    )

    for raw_variant in raw_variants:
        normalized = _normalize_lookup_text(raw_variant)
        if not normalized:
            continue

        variants.add(normalized)
        stripped = suffix_pattern.sub("", normalized).strip()
        if stripped:
            variants.add(stripped)

    return variants


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def _split_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    return [
        clause.strip()
        for clause in re.split(
            r"[.;:\n]+|(?:\s+(?:אך|אבל|אולם)\s+)",
            normalized,
        )
        if clause.strip()
    ]


def _clause_policy(
    text: str,
    keywords: Sequence[str],
) -> dict[str, Any]:
    """
    Detect allow/prohibit language only in clauses containing the target term.
    This avoids classifying "מותר מנגל, אסור מדורה" as a BBQ prohibition.
    """
    allow_tokens = ("מותר", "מותרת", "מותרים", "מותרות", "ניתן", "אפשר")
    prohibit_tokens = (
        "אסור",
        "אסורה",
        "אסורים",
        "אסורות",
        "אין להיכנס",
        "אין כניסה",
        "לא ניתן",
    )

    matched_clauses: list[str] = []
    has_allow = False
    has_prohibit = False
    has_restriction = False

    for clause in _split_clauses(text):
        if not any(keyword in clause for keyword in keywords):
            continue

        matched_clauses.append(clause)
        if any(token in clause for token in allow_tokens):
            has_allow = True
        if any(token in clause for token in prohibit_tokens):
            has_prohibit = True
        if any(
            token in clause
            for token in (
                "רק",
                "בלבד",
                "למעט",
                "בתנאי",
                "בעונה",
                "בנוכחות",
                "בשטח",
                "באזור",
                "במסלול",
            )
        ):
            has_restriction = True

    if has_allow and has_prohibit:
        status = "mixed_or_conditional"
    elif has_allow and has_restriction:
        status = "allowed_with_restrictions"
    elif has_allow:
        status = "allowed"
    elif has_prohibit:
        status = "prohibited"
    else:
        status = "unspecified"

    return {
        "status": status,
        "matched_clauses": matched_clauses,
    }


def _dogs_policy(mentioned: Any, summary: str) -> str:
    if not bool(mentioned):
        return "prohibited_by_catalog_policy"

    text = summary or ""
    has_allow = bool(re.search(r"מותר(?:ת|ים|ות)?|ניתן|אפשר", text))
    has_prohibit = bool(
        re.search(
            r"אסור(?:ה|ים|ות)?|אין\s+(?:כניסה|להכניס|להיכנס)",
            text,
        )
    )
    has_restriction = any(
        token in text
        for token in (
            "רק",
            "בלבד",
            "אך",
            "למעט",
            "מסלול",
            "מבנה",
            "מערה",
            "מתחם",
            "מוזיאון",
        )
    )

    if has_allow and has_prohibit:
        return "allowed_with_restrictions"
    if has_allow and has_restriction:
        return "allowed_with_restrictions"
    if has_allow:
        return "allowed"
    if has_prohibit:
        return "prohibited"
    return "unknown"


def _fire_policies(mentioned: Any, summary: str) -> dict[str, str]:
    if not bool(mentioned):
        return {
            "bbq_status": "prohibited_by_catalog_policy",
            "bonfire_status": "prohibited_by_catalog_policy",
        }

    text = summary or ""
    bbq = _clause_policy(
        text,
        ("מנגל", "מנגלים", "ברביקיו", "מצלה", "צליית בשר"),
    )
    bonfire = _clause_policy(text, ("מדורה", "מדורות"))

    generic_fire_prohibited = bool(
        re.search(
            r"(?:אסור|אסורה)\s+(?:להדליק|להבעיר)\s+אש|"
            r"הדלקת\s+אש\s+אסורה|הבערת\s+אש\s+אסורה",
            text,
        )
    )

    bbq_status = bbq["status"]
    bonfire_status = bonfire["status"]

    if bbq_status == "unspecified" and generic_fire_prohibited:
        bbq_status = "prohibited"
    if bonfire_status == "unspecified" and generic_fire_prohibited:
        bonfire_status = "prohibited"

    return {
        "bbq_status": bbq_status,
        "bonfire_status": bonfire_status,
    }


def _water_access_policy(mentioned: Any, summary: str) -> str:
    """
    Classify visitor water access from the official Hebrew summary.

    The implementation deliberately distinguishes:
    - water being present,
    - historical water installations,
    - permission to enter/swim/wade,
    - explicit prohibitions,
    - mixed or conditional access.

    It also ignores negated/uncertain phrases such as:
    "לא צוין שמותר להיכנס למים" and
    "אין מידע על אפשרות רחצה".
    """
    if not bool(mentioned):
        return "not_mentioned"

    text = re.sub(r"\s+", " ", summary or "").strip()
    if not text:
        return "water_present_access_unspecified"

    uncertainty_patterns = (
        r"אין\s+(?:מידע|אזכור|פירוט|ציון)\b",
        r"לא\s+(?:צוין|צוינה|צוינו|נכתב|נכתבה|מצוין|מצוינת|נמצא)\b",
        r"לא\s+ברור\b",
        r"לא\s+ידוע\b",
    )

    allow_patterns = (
        r"(?:ש)?מותר(?:ת|ים|ות)?.{0,55}"
        r"(?:להיכנס|כניסה|רחצה|לרחוץ|שחייה|לשחות|שכשוך|לשכשך|לשהות)",
        r"(?:להיכנס|כניסה|רחצה|לרחוץ|שחייה|לשחות|שכשוך|לשכשך|לשהות)"
        r".{0,55}(?:ש)?מותר(?:ת|ים|ות)?",
        r"\b(?:ניתן|אפשר)\b.{0,45}"
        r"(?:להיכנס|לרחוץ|לשחות|לשכשך|לשהות)\b",
        r"(?:בריכ(?:ה|ות)|חוף).{0,40}"
        r"(?:מותאמת|מותאמות|מיועדת|מיועדות|מוסדר|מוסדרים)"
        r".{0,25}(?:לרחצה|לשחייה|לשכשוך)",
        r"(?:כולל|מחייב|דורש).{0,45}(?:הליכה במים|שחייה)",
        r"מסלול\s+רטוב.{0,90}(?:הליכה|בתוך מי|שחייה)",
        r"הזדמנויות.{0,35}(?:לשחייה|להשתכשכות|לשכשוך)",
        r"אפשרות.{0,25}(?:רחצה|שחייה|שכשוך|שנירקול)",
        r"חוף\s+(?:ים\s+)?מוסדר\s+לרחצה",
        r"בריכ(?:ת|ות)\s+שכשוך.{0,45}"
        r"(?:לילדים|מיועדת|מיועדות|מותר|ניתן)",
    )

    prohibit_patterns = (
        r"\bאסור(?:ה|ים|ות)?\b.{0,55}"
        r"(?:להיכנס|כניסה|רחצה|לרחוץ|שחייה|לשחות|שכשוך|לשכשך)",
        r"(?:להיכנס|כניסה|רחצה|לרחוץ|שחייה|לשחות|שכשוך|לשכשך)"
        r".{0,55}\bאסור(?:ה|ים|ות)?\b",
        r"אין\s+(?:להיכנס|כניסה)\b",
    )

    restriction_tokens = (
        "רק",
        "בלבד",
        "בחלק",
        "במקומות המסומנים",
        "בעונה",
        "בחודשי",
        "בנוכחות מציל",
        "בנוכחות שירותי הצלה",
        "ללא פיקוח",
        "לפי שעות",
        "עד שעתיים",
        "שאר השמורה",
        "בשאר",
        "למעט",
        "בהתאם למסלול",
    )

    has_allow = False
    has_prohibit = False
    has_restriction = any(token in text for token in restriction_tokens)

    for clause in _split_clauses(text):
        clause_has_uncertainty = any(
            re.search(pattern, clause)
            for pattern in uncertainty_patterns
        )

        if any(re.search(pattern, clause) for pattern in prohibit_patterns):
            has_prohibit = True

        if not clause_has_uncertainty and any(
            re.search(pattern, clause)
            for pattern in allow_patterns
        ):
            has_allow = True

    if has_allow and has_prohibit:
        return "partially_or_conditionally_allowed"
    if has_allow and has_restriction:
        return "allowed_with_restrictions"
    if has_allow:
        return "allowed"
    if has_prohibit:
        return "prohibited"
    return "water_present_access_unspecified"


TOPIC_ALIASES: dict[str, str] = {
    "cleanliness": "hygiene_and_cleanliness",
    "hygiene": "hygiene_and_cleanliness",
    "ניקיון": "hygiene_and_cleanliness",
    "מים": "water",
    "water": "water",
    "water experience": "water",
    "water_experience": "water",
    "crowding": "crowding",
    "עומס": "crowding",
    "parking": "parking",
    "חניה": "parking",
    "accessibility": "accessibility",
    "נגישות": "accessibility",
    "price": "price_value",
    "מחיר": "price_value",
    "staff": "staff_service",
    "service": "staff_service",
    "צוות": "staff_service",
    "activities": "activities_and_attractions",
    "אטרקציות": "activities_and_attractions",
    "shade": "shade",
    "צל": "shade",
    "noise": "noise",
    "רעש": "noise",
    "animals": "animals",
    "חיות": "animals",
    "insects": "insects",
    "חרקים": "insects",
    "opening hours": "opening_hours",
    "שעות": "opening_hours",
    "booking": "booking_and_entry",
    "entry": "booking_and_entry",
    "כניסה": "booking_and_entry",
    "signage": "signage_and_navigation",
    "שילוט": "signage_and_navigation",
    "maintenance": "infrastructure_and_maintenance",
    "תחזוקה": "infrastructure_and_maintenance",
    "amenities": "visitor_amenities",
    "facilities": "visitor_amenities",
    "שירותים נוספים": "visitor_amenities",
}


# -----------------------------------------------------------------------------
# Main tool class
# -----------------------------------------------------------------------------

class NIPAgentTools:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"NIP agent database was not found: {self.db_path}"
            )
        self._validate_database()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _validate_database(self) -> None:
        required = {"sites", "site_topic_stats", "review_evidence"}
        with self._connect() as connection:
            available = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing = sorted(required - available)
            if missing:
                raise RuntimeError(
                    "Database is missing required tables: "
                    + ", ".join(missing)
                )
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"SQLite integrity check failed: {integrity}"
                )

    # ------------------------------------------------------------------
    # Discovery and resolution
    # ------------------------------------------------------------------

    def database_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            sites = connection.execute(
                """
                SELECT
                    COUNT(*) AS n_sites,
                    COUNT(DISTINCT site_name) AS distinct_sites,
                    SUM(n_reviews) AS n_reviews,
                    SUM(n_topic_mentions) AS n_topic_mentions,
                    MIN(first_review_year) AS first_review_year,
                    MAX(last_review_year) AS last_review_year
                FROM sites
                """
            ).fetchone()

            topics = connection.execute(
                """
                SELECT
                    COUNT(DISTINCT topic_id) AS n_topics,
                    COUNT(*) AS site_topic_rows,
                    SUM(CASE WHEN n_topic_mentions > 0 THEN 1 ELSE 0 END)
                        AS observed_site_topic_rows
                FROM site_topic_stats
                """
            ).fetchone()

            evidence = connection.execute(
                """
                SELECT
                    COUNT(*) AS n_evidence_rows,
                    COUNT(DISTINCT site_name || '|' || topic_id || '|' ||
                        sentiment_label_norm) AS n_evidence_groups
                FROM review_evidence
                """
            ).fetchone()

        return _json_safe(
            {
                **dict(sites),
                **dict(topics),
                **dict(evidence),
            }
        )

    def list_topics(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    topic_id,
                    MIN(topic_label_he) AS topic_label_he,
                    SUM(n_topic_mentions) AS n_mentions,
                    AVG(topic_global_sentiment_mean)
                        AS global_sentiment_mean
                FROM site_topic_stats
                GROUP BY topic_id
                ORDER BY n_mentions DESC, topic_id
                """
            ).fetchall()
        return [_json_safe(dict(row)) for row in rows]

    def search_sites(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query_variants = _site_lookup_variants(query)
        if not query_variants:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    site_name,
                    site_name_en,
                    region,
                    sub_region,
                    site_type,
                    official_url
                FROM sites
                """
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            candidate_variants = (
                _site_lookup_variants(item["site_name"])
                | _site_lookup_variants(item["site_name_en"])
            )

            if query_variants & candidate_variants:
                score = 1.0
            elif any(
                query_variant in candidate_variant
                for query_variant in query_variants
                for candidate_variant in candidate_variants
            ):
                score = 0.92
            elif any(
                candidate_variant in query_variant
                for query_variant in query_variants
                for candidate_variant in candidate_variants
            ):
                score = 0.88
            else:
                score = max(
                    difflib.SequenceMatcher(
                        None,
                        query_variant,
                        candidate_variant,
                    ).ratio()
                    for query_variant in query_variants
                    for candidate_variant in candidate_variants
                )

            if score >= 0.45:
                item["match_score"] = round(score, 4)
                scored.append((score, item))

        scored.sort(
            key=lambda pair: (
                pair[0],
                pair[1]["site_name"],
            ),
            reverse=True,
        )
        return [
            _json_safe(item)
            for _, item in scored[: max(1, int(limit))]
        ]

    def resolve_site_name(self, query: str) -> str:
        matches = self.search_sites(query, limit=5)
        if not matches:
            raise ValueError(f"No site matched {query!r}.")

        top = matches[0]
        second_score = (
            float(matches[1]["match_score"])
            if len(matches) > 1
            else -1.0
        )

        # Exact normalized/alias matches resolve immediately. Fuzzy matches
        # resolve only when they clearly outrank the second candidate.
        if float(top["match_score"]) >= 1.0:
            return str(top["site_name"])
        if (
            float(top["match_score"]) >= 0.88
            and float(top["match_score"]) - second_score >= 0.08
        ):
            return str(top["site_name"])

        suggestions = ", ".join(
            str(item["site_name"])
            for item in matches
        )
        raise ValueError(
            f"Site name {query!r} is ambiguous. Suggestions: {suggestions}"
        )

    def resolve_topic_id(self, query: str) -> str:
        query_norm = _normalize_lookup_text(query)
        alias = TOPIC_ALIASES.get(query_norm)
        if alias:
            return alias

        topics = self.list_topics()
        exact: list[str] = []
        scored: list[tuple[float, str]] = []

        for item in topics:
            topic_id = str(item["topic_id"])
            label = str(item["topic_label_he"])
            candidates = (
                _normalize_lookup_text(topic_id),
                _normalize_lookup_text(label),
            )

            if query_norm in candidates:
                exact.append(topic_id)
                continue

            score = max(
                difflib.SequenceMatcher(
                    None,
                    query_norm,
                    candidate,
                ).ratio()
                for candidate in candidates
            )
            scored.append((score, topic_id))

        if exact:
            return exact[0]

        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.72:
            return scored[0][1]

        raise ValueError(f"Unknown topic: {query!r}")

    # ------------------------------------------------------------------
    # Official facts
    # ------------------------------------------------------------------

    def get_site_facts(self, site_name: str) -> dict[str, Any]:
        resolved = self.resolve_site_name(site_name)

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sites WHERE site_name = ?",
                (resolved,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Site not found: {resolved}")

        raw = dict(row)
        fire = _fire_policies(
            raw.get("fire_bbq_mentioned"),
            raw.get("fire_bbq_llm_summary") or "",
        )

        result = {
            "site_name": raw["site_name"],
            "site_name_en": raw.get("site_name_en"),
            "official_url": raw.get("official_url"),
            "location": {
                "region": raw.get("region"),
                "sub_region": raw.get("sub_region"),
                "site_type": raw.get("site_type"),
            },
            "official_policies": {
                "dogs": {
                    "status": _dogs_policy(
                        raw.get("dogs_mentioned"),
                        raw.get("dogs_llm_summary") or "",
                    ),
                    "mentioned": _to_bool(raw.get("dogs_mentioned")),
                    "summary": raw.get("dogs_llm_summary"),
                    "source_text": raw.get("dogs_source_text"),
                },
                "water": {
                    "access_status": _water_access_policy(
                        raw.get("water_mentioned"),
                        raw.get("water_llm_summary") or "",
                    ),
                    "mentioned": _to_bool(raw.get("water_mentioned")),
                    "summary": raw.get("water_llm_summary"),
                    "source_text": raw.get("water_source_text"),
                },
                "fire_and_bbq": {
                    **fire,
                    "mentioned": _to_bool(
                        raw.get("fire_bbq_mentioned")
                    ),
                    "summary": raw.get("fire_bbq_llm_summary"),
                    "source_text": raw.get("fire_bbq_source_text"),
                },
                "accessibility": {
                    "mentioned": _to_bool(
                        raw.get("accessibility_mentioned")
                    ),
                    "summary": raw.get(
                        "accessibility_llm_summary"
                    ),
                    "source_text": raw.get(
                        "accessibility_source_text"
                    ),
                },
            },
            "facilities": {
                "family_easy_route": _to_bool(
                    raw.get("family_easy_route")
                ),
                "stroller_access": raw.get("stroller_access"),
                "camping_available": _to_bool(
                    raw.get("camping_available")
                ),
                "picnic_available": _to_bool(
                    raw.get("picnic_available")
                ),
                "visitor_center": _to_bool(
                    raw.get("visitor_center")
                ),
                "visitor_service_center": _to_bool(
                    raw.get("visitor_service_center")
                ),
                "parking_available": _to_bool(
                    raw.get("parking_available")
                ),
                "toilets_available": _to_bool(
                    raw.get("toilets_available")
                ),
                "shade_mentioned": _to_bool(
                    raw.get("shade_mentioned")
                ),
            },
            "visitor_information": {
                "opening_hours_text": raw.get(
                    "opening_hours_text"
                ),
                "entrance_fee_text": raw.get(
                    "entrance_fee_text"
                ),
                "parking_text": raw.get("parking_text"),
            },
            "review_summary": {
                "n_reviews": raw.get("n_reviews"),
                "first_review_year": raw.get(
                    "first_review_year"
                ),
                "last_review_year": raw.get(
                    "last_review_year"
                ),
                "n_topic_mentions": raw.get(
                    "n_topic_mentions"
                ),
                "overall_sentiment_mean": raw.get(
                    "overall_sentiment_mean"
                ),
                "overall_positive_rate": raw.get(
                    "overall_positive_rate"
                ),
                "overall_neutral_rate": raw.get(
                    "overall_neutral_rate"
                ),
                "overall_negative_rate": raw.get(
                    "overall_negative_rate"
                ),
            },
            "official_agent_text": raw.get(
                "official_agent_text"
            ),
        }
        return _json_safe(result)

    def filter_sites(
        self,
        *,
        region: str | None = None,
        sub_region: str | None = None,
        site_type: str | None = None,
        dogs_status: str | Sequence[str] | None = None,
        water_access_status: str | Sequence[str] | None = None,
        bbq_status: str | Sequence[str] | None = None,
        bonfire_status: str | Sequence[str] | None = None,
        stroller_access: str | Sequence[str] | None = None,
        parking_available: bool | None = None,
        toilets_available: bool | None = None,
        picnic_available: bool | None = None,
        camping_available: bool | None = None,
        visitor_center: bool | None = None,
        family_easy_route: bool | None = None,
        min_reviews: int = 0,
        limit: int = 71,
    ) -> list[dict[str, Any]]:
        def as_set(value: str | Sequence[str] | None) -> set[str] | None:
            if value is None:
                return None
            if isinstance(value, str):
                return {value}
            return {str(item) for item in value}

        dog_set = as_set(dogs_status)
        water_set = as_set(water_access_status)
        bbq_set = as_set(bbq_status)
        bonfire_set = as_set(bonfire_status)
        stroller_set = as_set(stroller_access)

        clauses = ["n_reviews >= ?"]
        parameters: list[Any] = [int(min_reviews)]

        for column, value in (
            ("region", region),
            ("sub_region", sub_region),
            ("site_type", site_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)

        boolean_filters = {
            "parking_available": parking_available,
            "toilets_available": toilets_available,
            "picnic_available": picnic_available,
            "camping_available": camping_available,
            "visitor_center": visitor_center,
            "family_easy_route": family_easy_route,
        }
        for column, value in boolean_filters.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(1 if value else 0)

        sql = f"""
            SELECT *
            FROM sites
            WHERE {" AND ".join(clauses)}
            ORDER BY n_reviews DESC, site_name
        """

        with self._connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    sql,
                    parameters,
                ).fetchall()
            ]

        results: list[dict[str, Any]] = []
        for raw in rows:
            dogs = _dogs_policy(
                raw.get("dogs_mentioned"),
                raw.get("dogs_llm_summary") or "",
            )
            water = _water_access_policy(
                raw.get("water_mentioned"),
                raw.get("water_llm_summary") or "",
            )
            fire = _fire_policies(
                raw.get("fire_bbq_mentioned"),
                raw.get("fire_bbq_llm_summary") or "",
            )
            stroller = raw.get("stroller_access")

            if dog_set is not None and dogs not in dog_set:
                continue
            if water_set is not None and water not in water_set:
                continue
            if bbq_set is not None and fire["bbq_status"] not in bbq_set:
                continue
            if (
                bonfire_set is not None
                and fire["bonfire_status"] not in bonfire_set
            ):
                continue
            if stroller_set is not None and stroller not in stroller_set:
                continue

            results.append(
                {
                    "site_name": raw["site_name"],
                    "site_name_en": raw.get("site_name_en"),
                    "region": raw.get("region"),
                    "sub_region": raw.get("sub_region"),
                    "site_type": raw.get("site_type"),
                    "dogs_status": dogs,
                    "water_access_status": water,
                    **fire,
                    "stroller_access": stroller,
                    "parking_available": _to_bool(
                        raw.get("parking_available")
                    ),
                    "toilets_available": _to_bool(
                        raw.get("toilets_available")
                    ),
                    "picnic_available": _to_bool(
                        raw.get("picnic_available")
                    ),
                    "camping_available": _to_bool(
                        raw.get("camping_available")
                    ),
                    "n_reviews": raw.get("n_reviews"),
                    "overall_sentiment_mean": raw.get(
                        "overall_sentiment_mean"
                    ),
                    "official_url": raw.get("official_url"),
                }
            )

            if len(results) >= max(1, int(limit)):
                break

        return _json_safe(results)

    # ------------------------------------------------------------------
    # Review statistics and evidence
    # ------------------------------------------------------------------

    def get_site_topic_stats(
        self,
        site_name: str,
        topics: Sequence[str] | None = None,
        *,
        observed_only: bool = True,
        min_mentions: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_site = self.resolve_site_name(site_name)
        topic_ids = (
            [self.resolve_topic_id(topic) for topic in topics]
            if topics
            else None
        )

        clauses = ["site_name = ?", "n_topic_mentions >= ?"]
        parameters: list[Any] = [
            resolved_site,
            int(min_mentions),
        ]

        if observed_only:
            clauses.append("n_topic_mentions > 0")

        if topic_ids:
            placeholders = ",".join("?" for _ in topic_ids)
            clauses.append(f"topic_id IN ({placeholders})")
            parameters.extend(topic_ids)

        sql = f"""
            SELECT *
            FROM site_topic_stats
            WHERE {" AND ".join(clauses)}
            ORDER BY n_topic_mentions DESC, topic_id
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [_json_safe(dict(row)) for row in rows]

    def compare_sites(
        self,
        site_names: Sequence[str],
        topics: Sequence[str] | None = None,
        *,
        min_mentions: int = 0,
    ) -> list[dict[str, Any]]:
        if not site_names:
            return []

        resolved_sites = [
            self.resolve_site_name(name)
            for name in site_names
        ]
        topic_ids = (
            [self.resolve_topic_id(topic) for topic in topics]
            if topics
            else None
        )

        site_placeholders = ",".join("?" for _ in resolved_sites)
        clauses = [
            f"site_name IN ({site_placeholders})",
            "n_topic_mentions >= ?",
        ]
        parameters: list[Any] = [
            *resolved_sites,
            int(min_mentions),
        ]

        if topic_ids:
            topic_placeholders = ",".join("?" for _ in topic_ids)
            clauses.append(f"topic_id IN ({topic_placeholders})")
            parameters.extend(topic_ids)

        sql = f"""
            SELECT
                site_name,
                site_name_en,
                topic_id,
                topic_label_he,
                n_topic_mentions,
                n_topic_reviews,
                topic_review_coverage,
                positive_rate,
                neutral_rate,
                negative_rate,
                sentiment_mean,
                smoothed_sentiment_score,
                reliability_weight,
                recent_n_mentions,
                recent_sentiment_mean,
                sentiment_trend_delta,
                trend_is_reliable
            FROM site_topic_stats
            WHERE {" AND ".join(clauses)}
            ORDER BY topic_id, smoothed_sentiment_score DESC, site_name
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [_json_safe(dict(row)) for row in rows]

    def get_review_evidence(
        self,
        site_name: str,
        topic: str,
        *,
        sentiment: str | None = None,
        recent_only: bool = False,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        resolved_site = self.resolve_site_name(site_name)
        topic_id = self.resolve_topic_id(topic)

        clauses = ["site_name = ?", "topic_id = ?"]
        parameters: list[Any] = [resolved_site, topic_id]

        if sentiment is not None:
            sentiment_norm = sentiment.strip().casefold()
            allowed = {"positive", "neutral", "negative"}
            if sentiment_norm not in allowed:
                raise ValueError(
                    "sentiment must be positive, neutral or negative"
                )
            clauses.append("sentiment_label_norm = ?")
            parameters.append(sentiment_norm)

        if recent_only:
            clauses.append("is_recent_period = 1")

        sql = f"""
            SELECT
                site_name,
                site_name_en,
                topic_id,
                topic_label_he,
                sentiment_label_norm,
                sentiment_numeric_score,
                review_year,
                source,
                segment_text,
                mean_sentiment_confidence,
                n_merged_segments,
                evidence_quality_score,
                evidence_rank
            FROM review_evidence
            WHERE {" AND ".join(clauses)}
            ORDER BY
                evidence_quality_score DESC,
                evidence_rank ASC
            LIMIT ?
        """
        parameters.append(max(1, int(limit)))

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [_json_safe(dict(row)) for row in rows]

    def rank_sites(
        self,
        topic_weights: Mapping[str, float],
        *,
        constraints: Mapping[str, Any] | None = None,
        min_mentions_per_topic: int = 0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not topic_weights:
            raise ValueError("topic_weights cannot be empty")

        resolved_weights: dict[str, float] = {}
        for topic, weight in topic_weights.items():
            numeric_weight = float(weight)
            if numeric_weight == 0:
                continue
            topic_id = self.resolve_topic_id(topic)
            resolved_weights[topic_id] = (
                resolved_weights.get(topic_id, 0.0)
                + numeric_weight
            )

        if not resolved_weights:
            raise ValueError(
                "topic_weights contain only zero-valued weights"
            )

        candidates = self.filter_sites(
            **dict(constraints or {}),
            limit=71,
        )
        if not candidates:
            return []

        candidate_names = [
            item["site_name"]
            for item in candidates
        ]
        site_placeholders = ",".join("?" for _ in candidate_names)
        topic_ids = list(resolved_weights)
        topic_placeholders = ",".join("?" for _ in topic_ids)

        sql = f"""
            SELECT
                site_name,
                topic_id,
                topic_label_he,
                n_topic_mentions,
                smoothed_sentiment_score,
                reliability_weight,
                sentiment_trend_delta,
                trend_is_reliable
            FROM site_topic_stats
            WHERE site_name IN ({site_placeholders})
              AND topic_id IN ({topic_placeholders})
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                [*candidate_names, *topic_ids],
            ).fetchall()

        stats_by_site: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            stats_by_site.setdefault(
                item["site_name"],
                {},
            )[item["topic_id"]] = item

        candidate_by_name = {
            item["site_name"]: item
            for item in candidates
        }
        weight_denominator = sum(
            abs(weight)
            for weight in resolved_weights.values()
        )

        ranked: list[dict[str, Any]] = []
        for site_name in candidate_names:
            site_stats = stats_by_site.get(site_name, {})

            if any(
                int(site_stats.get(topic_id, {}).get(
                    "n_topic_mentions",
                    0,
                ))
                < int(min_mentions_per_topic)
                for topic_id in topic_ids
            ):
                continue

            weighted_score_sum = 0.0
            support_sum = 0.0
            requested_mentions = 0
            topic_breakdown: list[dict[str, Any]] = []

            for topic_id, weight in resolved_weights.items():
                stat = site_stats.get(topic_id)
                if stat is None:
                    continue

                score = float(stat["smoothed_sentiment_score"])
                reliability = float(stat["reliability_weight"])
                mentions = int(stat["n_topic_mentions"])

                weighted_score_sum += weight * score
                support_sum += abs(weight) * reliability
                requested_mentions += mentions

                topic_breakdown.append(
                    {
                        "topic_id": topic_id,
                        "topic_label_he": stat[
                            "topic_label_he"
                        ],
                        "weight": weight,
                        "n_topic_mentions": mentions,
                        "smoothed_sentiment_score": score,
                        "reliability_weight": reliability,
                        "sentiment_trend_delta": stat[
                            "sentiment_trend_delta"
                        ],
                        "trend_is_reliable": bool(
                            stat["trend_is_reliable"]
                        ),
                    }
                )

            if not topic_breakdown:
                continue

            weighted_score = (
                weighted_score_sum / weight_denominator
            )
            support_score = support_sum / weight_denominator

            ranked.append(
                {
                    **candidate_by_name[site_name],
                    "weighted_topic_score": round(
                        weighted_score,
                        6,
                    ),
                    "data_support_score": round(
                        support_score,
                        6,
                    ),
                    "requested_topic_mentions": requested_mentions,
                    "topic_breakdown": topic_breakdown,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["weighted_topic_score"],
                item["data_support_score"],
                item["requested_topic_mentions"],
                item["n_reviews"],
            ),
            reverse=True,
        )

        for index, item in enumerate(
            ranked[: max(1, int(limit))],
            start=1,
        ):
            item["rank"] = index

        return _json_safe(ranked[: max(1, int(limit))])


# -----------------------------------------------------------------------------
# Smoke test and small CLI
# -----------------------------------------------------------------------------

def run_self_test(db_path: Path) -> dict[str, Any]:
    tools = NIPAgentTools(db_path)

    summary = tools.database_summary()
    expected = {
        "n_sites": 71,
        "distinct_sites": 71,
        "n_reviews": 30442,
        "n_topic_mentions": 62634,
        "n_topics": 25,
        "site_topic_rows": 1775,
    }
    for key, value in expected.items():
        if int(summary[key]) != value:
            raise AssertionError(
                f"{key}: expected {value}, got {summary[key]}"
            )

    facts = tools.get_site_facts("עין גדי")
    if facts["site_name"] != "עין גדי":
        raise AssertionError("Site resolution failed.")

    gan_hashlosha = tools.resolve_site_name(
        "Gan HaShlosha National Park"
    )
    if gan_hashlosha != "גן השלושה (סחנה)":
        raise AssertionError(
            "English site-type suffix resolution failed."
        )

    yarkon_matches = tools.search_sites("Yarkon", limit=5)
    if len(yarkon_matches) < 2:
        raise AssertionError(
            "Ambiguous Yarkon search should return both sections."
        )

    water_regression_cases = {
        "תל מגידו": "water_present_access_unspecified",
        "כורזים": "water_present_access_unspecified",
        "ציפורי": "water_present_access_unspecified",
        "גמלא": "prohibited",
        "תל דן": "partially_or_conditionally_allowed",
        "עין חניה": "allowed",
    }
    for site_name, expected_status in water_regression_cases.items():
        actual_status = tools.get_site_facts(site_name)[
            "official_policies"
        ]["water"]["access_status"]
        if actual_status != expected_status:
            raise AssertionError(
                f"Water policy regression for {site_name}: "
                f"expected {expected_status}, got {actual_status}"
            )

    comparison = tools.compare_sites(
        ["אכזיב", "עין גדי"],
        ["water", "cleanliness"],
    )
    if not comparison:
        raise AssertionError("Site comparison returned no rows.")

    evidence = tools.get_review_evidence(
        "אכזיב",
        "water",
        limit=2,
    )
    if not evidence:
        raise AssertionError("Evidence retrieval returned no rows.")

    ranking = tools.rank_sites(
        {
            "water": 2.0,
            "hygiene_and_cleanliness": 1.0,
            "crowding": 1.0,
        },
        constraints={"region": "north"},
        limit=5,
    )
    if not ranking:
        raise AssertionError("Ranking returned no rows.")

    return {
        "database_summary": summary,
        "resolved_site": facts["site_name"],
        "comparison_rows": len(comparison),
        "evidence_rows": len(evidence),
        "ranking_rows": len(ranking),
        "top_ranked_site": ranking[0]["site_name"],
        "status": "QUALITY GATE PASSED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIP deterministic agent tools"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to nip_agent.db",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--summary", action="store_true")
    group.add_argument("--search", type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools = NIPAgentTools(args.db)

    if args.self_test:
        payload = run_self_test(args.db)
    elif args.summary:
        payload = tools.database_summary()
    elif args.search is not None:
        payload = tools.search_sites(args.search)
    else:
        raise AssertionError("Unreachable CLI state")

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
