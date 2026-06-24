#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build the public-safe SQLite database used by the Streamlit agent.

The output excludes full review text, reviewer identifiers, batch paths and
internal extraction diagnostics. It retains only aggregate statistics,
public official-site information, and short representative review excerpts.

Usage:
    python scripts/build_public_agent_db.py \
        --source path/to/nip_agent.db \
        --output data/public/nip_agent_public.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SITE_COLUMNS = [
    "site_name", "site_name_en", "official_url", "region", "sub_region",
    "site_type", "water_mentioned", "water_source_text",
    "water_llm_summary", "accessibility_mentioned",
    "accessibility_source_text", "accessibility_llm_summary",
    "dogs_mentioned", "dogs_source_text", "dogs_llm_summary",
    "fire_bbq_mentioned", "fire_bbq_source_text", "fire_bbq_llm_summary",
    "family_easy_route", "stroller_access", "camping_available",
    "picnic_available", "visitor_center", "visitor_service_center",
    "opening_hours_text", "entrance_fee_text", "parking_available",
    "parking_text", "toilets_available", "shade_mentioned", "n_reviews",
    "first_review_year", "last_review_year", "n_topic_mentions",
    "overall_sentiment_mean", "overall_positive_rate",
    "overall_neutral_rate", "overall_negative_rate", "official_agent_text",
]

EVIDENCE_COLUMNS = [
    "site_name", "site_name_en", "topic_id", "topic_label_he",
    "sentiment_label_norm", "sentiment_numeric_score", "review_year",
    "source", "segment_text", "mean_sentiment_confidence",
    "n_merged_segments", "is_recent_period", "evidence_quality_score",
    "evidence_rank",
]


def sanitize_excerpt(value: Any, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"https?://\S+|www\.\S+", "[link removed]", text)
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[email removed]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\d)(?:\+?972[-\s]?)?0?5\d[-\s]?\d{3}[-\s]?\d{4}(?!\d)",
        "[phone removed]",
        text,
    )
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened + "…"


def copy_table_subset(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> None:
    column_sql = ", ".join(f'"{column}"' for column in columns)
    source_rows = source.execute(
        f'SELECT {column_sql} FROM "{table}"'
    ).fetchall()

    type_map = {
        row[1]: row[2] or "TEXT"
        for row in source.execute(f'PRAGMA table_info("{table}")')
    }
    definitions = ", ".join(
        f'"{column}" {type_map.get(column, "TEXT")}' for column in columns
    )
    target.execute(f'CREATE TABLE "{table}" ({definitions})')

    placeholders = ", ".join("?" for _ in columns)
    target.executemany(
        f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})',
        [tuple(row[column] for column in columns) for row in source_rows],
    )


def build_public_db(source_path: Path, output_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(temp_path)

    try:
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("PRAGMA synchronous=FULL")

        copy_table_subset(source, target, "sites", SITE_COLUMNS)

        topic_columns = [
            row[1]
            for row in source.execute("PRAGMA table_info(site_topic_stats)")
        ]
        copy_table_subset(source, target, "site_topic_stats", topic_columns)

        definitions = {
            row[1]: row[2] or "TEXT"
            for row in source.execute("PRAGMA table_info(review_evidence)")
            if row[1] in EVIDENCE_COLUMNS
        }
        target.execute(
            "CREATE TABLE review_evidence ("
            + ", ".join(
                f'"{column}" {definitions.get(column, "TEXT")}'
                for column in EVIDENCE_COLUMNS
            )
            + ")"
        )

        select_sql = ", ".join(f'"{column}"' for column in EVIDENCE_COLUMNS)
        evidence_rows = source.execute(
            f"SELECT {select_sql} FROM review_evidence"
        ).fetchall()
        placeholders = ", ".join("?" for _ in EVIDENCE_COLUMNS)

        public_rows = []
        for row in evidence_rows:
            values = dict(row)
            values["segment_text"] = sanitize_excerpt(values["segment_text"])
            public_rows.append(tuple(values[column] for column in EVIDENCE_COLUMNS))

        target.executemany(
            "INSERT INTO review_evidence ("
            + select_sql
            + f") VALUES ({placeholders})",
            public_rows,
        )

        target.execute("CREATE TABLE _build_metadata (key TEXT, value TEXT)")
        counts = {
            "public_built_at_utc": datetime.now(timezone.utc).isoformat(),
            "site_count": source.execute("SELECT COUNT(*) FROM sites").fetchone()[0],
            "review_count": source.execute("SELECT SUM(n_reviews) FROM sites").fetchone()[0],
            "mention_count": source.execute("SELECT SUM(n_topic_mentions) FROM sites").fetchone()[0],
            "topic_count": source.execute("SELECT COUNT(DISTINCT topic_id) FROM site_topic_stats").fetchone()[0],
            "site_topic_row_count": source.execute("SELECT COUNT(*) FROM site_topic_stats").fetchone()[0],
            "evidence_row_count": len(public_rows),
            "privacy_note": (
                "Full review text and internal review identifiers are excluded; "
                "only short representative excerpts without reviewer metadata are retained."
            ),
        }
        target.executemany(
            "INSERT INTO _build_metadata (key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in counts.items()],
        )

        target.execute(
            "CREATE VIEW site_topic_observed AS "
            "SELECT * FROM site_topic_stats WHERE n_topic_mentions > 0"
        )
        target.execute("CREATE UNIQUE INDEX idx_sites_name ON sites(site_name)")
        target.execute(
            "CREATE INDEX idx_sites_filters ON sites(region, site_type, "
            "parking_available, toilets_available)"
        )
        target.execute(
            "CREATE UNIQUE INDEX idx_site_topic_key "
            "ON site_topic_stats(site_name, topic_id)"
        )
        target.execute(
            "CREATE INDEX idx_site_topic_lookup "
            "ON site_topic_stats(site_name, topic_id, n_topic_mentions)"
        )
        target.execute(
            "CREATE INDEX idx_evidence_lookup ON review_evidence("
            "site_name, topic_id, sentiment_label_norm, is_recent_period, "
            "evidence_quality_score DESC)"
        )

        target.commit()
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Public DB integrity check failed: {integrity}")
        target.execute("VACUUM")
        target.commit()
    finally:
        source.close()
        target.close()

    temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_public_db(args.source.resolve(), args.output.resolve())
    print(f"Created public agent database: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
