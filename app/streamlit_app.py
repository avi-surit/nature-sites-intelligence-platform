from pathlib import Path
from html import escape
import json
import os
import sys
import time


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Nature Sites Intelligence Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA_DIR = PROJECT_ROOT / "data" / "public"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENT_IMPORT_ERROR = None
try:
    from agent.site_intelligence_agent import SiteIntelligenceAgent
except Exception as exc:
    SiteIntelligenceAgent = None
    AGENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

SITE_TOPIC_PATH = PUBLIC_DATA_DIR / "site_topic_sentiment_stats_public.parquet"
SITE_STATS_PATH = PUBLIC_DATA_DIR / "site_sentiment_stats_public.parquet"
TOPIC_STATS_PATH = PUBLIC_DATA_DIR / "topic_sentiment_stats_public.parquet"
METADATA_PATH = PUBLIC_DATA_DIR / "app_metadata.json"
AGENT_DB_PATH = PUBLIC_DATA_DIR / "nip_agent_public.db"
AGENT_SESSION_KEY = "site_intelligence_agent_instance"
AGENT_MESSAGES_KEY = "site_intelligence_agent_messages"
AGENT_QUESTION_COUNT_KEY = "site_intelligence_agent_question_count"
AGENT_MAX_QUESTIONS_PER_SESSION = 12
AI_MODEL = "gpt-5-nano"
AI_MAX_TOPICS = 8
AI_MAX_OUTPUT_TOKENS = 450
CHART_TITLE_COLOR = "#F97316"  # warm terracotta, readable in light/dark themes
CHART_TITLE_SIZE = 21
LIVE_ANALYZER_MODEL = "gpt-5-nano"

SAMPLE_REVIEWS = [
    "The site was beautiful and the staff were helpful, but the entrance process was slow and the bathrooms were not clean.",
    "The trail was impressive and well worth the visit. However, the signs were confusing and we had trouble finding the main viewpoint.",
    "The kids enjoyed the activities, but the place was extremely crowded and there was not enough shade near the waiting area.",
    "The historical exhibits were interesting, but some of the audio explanations were unclear and a few screens did not seem to work.",
    "Parking was easy and the staff were friendly. The only issue was that the food stand was closed even though it appeared online as open.",
    "The nature reserve is stunning, but the booking process was frustrating and the entry line took much longer than expected.",
    "The site was clean, calm, and well maintained. We especially appreciated the clear explanations and helpful staff.",
    "The water area was great, but there were too many visitors, not enough supervision, and the changing rooms were dirty.",
    "The visit felt overpriced for what was available. Several facilities were closed and there was little information at the entrance.",
    "The trail was beautiful, but accessibility was limited and it was difficult for older visitors to reach some of the main areas.",
]

LIVE_TOPIC_SCHEMA = {
    "other": "Other / General Feedback",
    "activities_and_attractions": "Activities & Attractions",
    "booking_and_entry": "Booking & Entry",
    "hygiene_and_cleanliness": "Hygiene & Cleanliness",
    "staff_service": "Staff & Service",
    "information": "Information & Explanations",
    "crowding": "Crowding",
    "supervision_and_enforcement": "Supervision & Enforcement",
    "signage_and_navigation": "Signage & Navigation",
    "visitor_amenities": "Visitor Amenities",
    "infrastructure_and_maintenance": "Infrastructure & Maintenance",
    "operational_efficiency": "Operational Efficiency",
    "shop_and_food": "Shop & Food",
    "accessibility": "Accessibility",
    "shade": "Shade",
    "parking": "Parking",
    "price_value": "Price & Value",
    "opening_hours": "Opening Hours",
    "water": "Water",
    "weather_and_temperature": "Weather & Temperature",
    "insects": "Insects & Pests",
    "animals": "Animals",
    "showers_and_changing_rooms": "Showers & Changing Rooms",
    "modesty_and_religion": "Modesty & Religion",
    "noise": "Noise",
}

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, #0f3d2e 0%, #1f7a5a 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 760;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin-top: 0.6rem;
        font-size: 1.02rem;
        opacity: 0.93;
        max-width: 980px;
    }

    .metric-card {
        background: white;
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.06);
        min-height: 112px;
    }

    .metric-card:hover {
        box-shadow: 0 8px 26px rgba(0,0,0,0.10);
        transform: translateY(-1px);
        transition: 0.15s ease-in-out;
    }

    .metric-title {
        color: #667085;
        font-size: 0.84rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.55rem;
        font-weight: 750;
    }

    .metric-sub {
        color: #667085;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 780;
        color: #F97316;
        margin-top: 1.25rem;
        margin-bottom: 0.55rem;
        letter-spacing: -0.01em;
    }

    .note-box {
        padding: 0.9rem 1rem;
        border-radius: 0.9rem;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        font-size: 0.9rem;
    }

    .small-muted {
        color: #667085;
        font-size: 0.86rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 1rem;
        overflow: hidden;
    }

    .agent-activity-line {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-height: 1.65rem;
        color: #FB2943;
        font-size: 0.92rem;
        font-weight: 680;
        line-height: 1.35;
        letter-spacing: 0.005em;
    }

    .agent-activity-dot {
        width: 0.52rem;
        height: 0.52rem;
        flex: 0 0 0.52rem;
        border-radius: 999px;
        background: #FB2943;
        box-shadow: 0 0 0 0 rgba(251, 41, 67, 0.35);
        animation: agentActivityPulse 1.15s ease-in-out infinite;
    }

    @keyframes agentActivityPulse {
        0%, 100% {
            opacity: 0.42;
            transform: scale(0.88);
            box-shadow: 0 0 0 0 rgba(251, 41, 67, 0.28);
        }
        50% {
            opacity: 1;
            transform: scale(1.08);
            box-shadow: 0 0 0 0.34rem rgba(251, 41, 67, 0);
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

def add_empirical_priority_context(df: pd.DataFrame, score_col: str = "priority_score", prefix: str = "priority"):
    """
    Adds empirical rank/percentile/range columns for priority_score.

    Rank:
        1 = highest priority / most urgent

    Severity percentile:
        1.0 = highest observed score in the comparison group
        0.0/low = low observed score
    """
    df = df.copy()

    if score_col not in df.columns or df.empty:
        return df

    valid_scores = df[score_col].dropna()

    if valid_scores.empty:
        df[f"{prefix}_rank"] = pd.NA
        df[f"{prefix}_scope_n"] = 0
        df[f"{prefix}_severity_percentile"] = pd.NA
        df[f"{prefix}_min"] = pd.NA
        df[f"{prefix}_max"] = pd.NA
        df[f"{prefix}_rank_label"] = "No priority data"
        return df

    scope_n = len(valid_scores)
    min_score = float(valid_scores.min())
    max_score = float(valid_scores.max())

    df[f"{prefix}_rank"] = df[score_col].rank(
        method="min",
        ascending=False,
    ).astype("Int64")

    df[f"{prefix}_scope_n"] = scope_n

    df[f"{prefix}_severity_percentile"] = df[score_col].rank(
        method="max",
        ascending=True,
        pct=True,
    )

    df[f"{prefix}_min"] = min_score
    df[f"{prefix}_max"] = max_score

    def make_label(row):
        rank = row.get(f"{prefix}_rank")
        pct_value = row.get(f"{prefix}_severity_percentile")

        if pd.isna(rank) or pd.isna(pct_value):
            return "No priority data"

        return f"Rank {int(rank)} / {scope_n} · {100 * float(pct_value):.0f}% severity percentile"

    df[f"{prefix}_rank_label"] = df.apply(make_label, axis=1)

    return df


@st.cache_data(show_spinner=False)
def load_data():
    site_topic = pd.read_parquet(SITE_TOPIC_PATH)
    site_stats = pd.read_parquet(SITE_STATS_PATH)
    topic_stats = pd.read_parquet(TOPIC_STATS_PATH)

    for df in [site_topic, site_stats, topic_stats]:
        for col in ["negative", "neutral", "positive"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

    site_stats = add_empirical_priority_context(
        site_stats,
        score_col="priority_score",
        prefix="site_priority",
    )

    topic_stats = add_empirical_priority_context(
        topic_stats,
        score_col="priority_score",
        prefix="topic_priority",
    )

    site_topic = add_empirical_priority_context(
        site_topic,
        score_col="priority_score",
        prefix="global_site_topic_priority",
    )

    return site_topic, site_stats, topic_stats


site_topic_df, site_stats_df, topic_stats_df = load_data()


# ============================================================
# Helpers
# ============================================================

def priority_context_text(row, prefix: str, fallback: str = "Operational triage score"):
    rank = row.get(f"{prefix}_rank")
    scope_n = row.get(f"{prefix}_scope_n")
    pct_value = row.get(f"{prefix}_severity_percentile")
    min_score = row.get(f"{prefix}_min")
    max_score = row.get(f"{prefix}_max")

    if pd.isna(rank) or pd.isna(scope_n) or pd.isna(pct_value):
        return fallback

    lines = [
        f"Rank {int(rank)}/{int(scope_n)}",
        f"{100 * float(pct_value):.0f}% severity percentile",
    ]

    if not pd.isna(min_score) and not pd.isna(max_score):
        lines.append(f"Observed range {float(min_score):.2f}–{float(max_score):.2f}")

    return "\n".join(lines)

METRIC_TOOLTIPS = {
    "Sites in view": (
        "Number of sites currently included after the sidebar filters."
    ),
    "Topics in view": (
        "Number of topic categories currently included after the sidebar filters."
    ),
    "Topic mentions": (
        "Total review-topic sentiment mentions after segmentation. "
        "A single original review can contribute to multiple topics and sentiment categories."
    ),
    "Mentions": (
        "Aggregated sentiment mentions. This is not necessarily the raw number of original reviews, "
        "because one review can contain multiple topic-specific segments."
    ),
    "Negative share": (
        "negative / (negative + neutral + positive). "
        "Shows the share of negative mentions within the selected scope."
    ),
    "Net sentiment": (
        "(positive - negative) / (negative + neutral + positive). "
        "Range: -1 to 1. Higher is better; lower means more negative concentration."
    ),
    "Priority score": (
        "log(1 + negative) × negative_rate. "
        "Combines complaint volume with negative concentration. "
        "Used for operational triage, not as statistical ground truth."
    ),
}


SENTIMENT_DISPLAY = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
}

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

SENTIMENT_COLOR_MAP = {
    "Positive": "#2563EB",  # blue
    "Neutral": "#9CA3AF",   # gray
    "Negative": "#D64545",  # red
}

METRIC_DISPLAY = {
    "total_topic_reviews": "Total Mentions",
    "priority_score": "Priority Score",
    "negative": "Negative Mentions",
    "negative_rate": "Negative Share",
    "net_sentiment_score": "Net Sentiment Score",
    "positive_rate": "Positive Share",
    "neutral_rate": "Neutral Share",
    "positive": "Positive Mentions",
    "neutral": "Neutral Mentions",
}

COLUMN_DISPLAY = {
    "site_name_en": "Site",
    "site_name_he": "Site Name (Hebrew)",
    "topic_label_en": "Topic",
    "topic_group_en": "Topic Group",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "total_topic_reviews": "Total Mentions",
    "total_reviews_with_mentions": "Total Mentions",
    "negative_rate": "Negative Share",
    "neutral_rate": "Neutral Share",
    "positive_rate": "Positive Share",
    "net_sentiment_score": "Net Sentiment Score",
    "priority_score": "Priority Score",
}

PLOTLY_LABELS = {
    "site_name_en": "Site",
    "site_name_he": "Site Name (Hebrew)",
    "site_display_name": "Site",
    "topic_label_en": "Topic",
    "topic_group_en": "Topic Group",
    "topic_id": "Topic ID",

    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "count": "Mentions",
    "total_topic_reviews": "Total Mentions",
    "total_reviews_with_mentions": "Total Mentions",

    "negative_rate": "Negative Share",
    "neutral_rate": "Neutral Share",
    "positive_rate": "Positive Share",
    "net_sentiment_score": "Net Sentiment Score",
    "priority_score": "Priority Score",

    "sentiment": "Sentiment",
    "sentiment_display": "Sentiment",
    "site_topic": "Site · Topic",

    "view_priority_rank_label": "Rank in Current View",
    "view_priority_severity_percentile": "Severity Percentile in Current View",
    "view_priority_min": "Priority Score Min in Current View",
    "view_priority_max": "Priority Score Max in Current View",

    "site_priority_rank_label": "Site Priority Rank",
    "site_priority_severity_percentile": "Site Severity Percentile",
    "site_priority_min": "Site Priority Min",
    "site_priority_max": "Site Priority Max",

    "topic_priority_rank_label": "Topic Priority Rank",
    "topic_priority_severity_percentile": "Topic Severity Percentile",
    "topic_priority_min": "Topic Priority Min",
    "topic_priority_max": "Topic Priority Max",

    "topic_view_priority_rank_label": "Rank Within Selected Topic",
    "topic_view_priority_severity_percentile": "Severity Percentile Within Selected Topic",
    "topic_view_priority_min": "Priority Min Within Selected Topic",
    "topic_view_priority_max": "Priority Max Within Selected Topic",
}


LIVE_PROCESS_STEPS = [
    ("Segmenting review...", "blue"),
    ("Mapping topics to schema...", "violet"),
    ("Estimating sentiment...", "orange"),
    ("Aggregating recommendations...", "green"),
]


def render_terminal_logs(log_placeholder, active_step: int, done: bool = False):
    """
    Temporary Streamlit-native colored process logs.
    Each step keeps its original color even after completion.
    """
    lines = []

    for idx, (label, color) in enumerate(LIVE_PROCESS_STEPS):
        if done or idx < active_step:
            prefix = "✓"
            lines.append(f":{color}[{prefix} {label}]")
        elif idx == active_step:
            prefix = "›"
            lines.append(f":{color}[{prefix} {label}]")
        else:
            prefix = "·"
            lines.append(f":gray[{prefix} {label}]")

    if done:
        lines.append(":green[✓ Done.]")

    log_placeholder.markdown("\n\n".join(lines))


def render_done_log(log_placeholder):
    log_placeholder.markdown(":green[✓ Done.]")




def initialize_sample_review_state():
    if "sample_review_index" not in st.session_state:
        import random
        st.session_state.sample_review_index = random.randint(0, len(SAMPLE_REVIEWS) - 1)

    if "live_review_text" not in st.session_state:
        st.session_state.live_review_text = SAMPLE_REVIEWS[st.session_state.sample_review_index]


def suggest_next_review():
    current_idx = st.session_state.get("sample_review_index", 0)
    next_idx = (current_idx + 1) % len(SAMPLE_REVIEWS)

    st.session_state.sample_review_index = next_idx
    st.session_state.live_review_text = SAMPLE_REVIEWS[next_idx]


def soften_sparse_bars(fig, n_categories: int, *, grouped: bool = False):
    """
    Makes bars visually narrower when the chart has only a few categories.
    Useful for charts with Positive / Neutral / Negative only.
    """
    if n_categories <= 3:
        fig.update_traces(width=0.32 if grouped else 0.42)
        fig.update_layout(bargap=0.55, bargroupgap=0.22)
    elif n_categories <= 6:
        fig.update_traces(width=0.48 if grouped else 0.58)
        fig.update_layout(bargap=0.38, bargroupgap=0.16)

    return fig

def pretty_metric_name(metric_name: str) -> str:
    return METRIC_DISPLAY.get(
        metric_name,
        str(metric_name).replace("_", " ").title(),
    )


def add_sentiment_display_column(df: pd.DataFrame, source_col: str = "sentiment") -> pd.DataFrame:
    df = df.copy()
    df["sentiment_display"] = (
        df[source_col]
        .map(SENTIMENT_DISPLAY)
        .fillna(df[source_col].astype(str).str.title())
    )

    df["sentiment_display"] = pd.Categorical(
        df["sentiment_display"],
        categories=SENTIMENT_ORDER,
        ordered=True,
    )

    return df


def display_dataframe(df: pd.DataFrame, cols: list[str], *, use_container_width=True, hide_index=True):
    display_df = ensure_columns(df, cols).rename(columns=COLUMN_DISPLAY)

    st.dataframe(
        display_df,
        use_container_width=use_container_width,
        hide_index=hide_index,
    )

def pct(x):
    if pd.isna(x):
        return "0.0%"
    return f"{100 * float(x):.1f}%"


def number(x):
    if pd.isna(x):
        return "0"
    return f"{int(x):,}"



def apply_plot_readability(fig):
    fig.update_layout(
        font=dict(size=14),
        xaxis=dict(
            tickfont=dict(size=14),
            title_font=dict(size=15),
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            title_font=dict(size=15),
        ),
        legend=dict(
            font=dict(size=13),
            title_font=dict(size=13),
        ),
    )
    return fig

def metric_card(title, value, sub="", tooltip=None):
    tooltip_text = tooltip or METRIC_TOOLTIPS.get(title, "")
    info_icon = " ⓘ" if tooltip_text else ""

    safe_title = escape(str(title))
    safe_value = escape(str(value))
    safe_sub = escape(str(sub)).replace("\n", "<br>")
    safe_tooltip = escape(str(tooltip_text))

    st.markdown(
        f"""
        <div class="metric-card" title="{safe_tooltip}">
            <div class="metric-title">{safe_title}{info_icon}</div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-sub">{safe_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_long_format(df, id_cols, total_col):
    value_vars = [col for col in ["negative", "neutral", "positive"] if col in df.columns]

    out = df.melt(
        id_vars=id_cols + [total_col],
        value_vars=value_vars,
        var_name="sentiment",
        value_name="count",
    )

    return out


def ensure_columns(df, cols):
    existing = [col for col in cols if col in df.columns]
    return df[existing].copy()


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        if not api_key and hasattr(st, "secrets"):
            api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None

    return api_key


@st.cache_resource(show_spinner=False)
def get_cached_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


def get_openai_client():
    if OpenAI is None:
        return None

    api_key = get_openai_api_key()

    if not api_key:
        return None

    return get_cached_openai_client(api_key)


def initialize_agent_state():
    if AGENT_MESSAGES_KEY not in st.session_state:
        st.session_state[AGENT_MESSAGES_KEY] = []
    if AGENT_QUESTION_COUNT_KEY not in st.session_state:
        st.session_state[AGENT_QUESTION_COUNT_KEY] = 0


def get_agent_readiness_issues():
    issues = []

    if SiteIntelligenceAgent is None:
        issues.append(
            "The agent package could not be loaded. Install the updated "
            "requirements and restart Streamlit."
        )

    if not AGENT_DB_PATH.is_file():
        issues.append(
            "The public agent database is missing: "
            "data/public/nip_agent_public.db"
        )

    if not get_openai_api_key():
        issues.append(
            "OPENAI_API_KEY was not found in Streamlit secrets."
        )

    return issues


def get_site_intelligence_agent():
    issues = get_agent_readiness_issues()
    if issues:
        raise RuntimeError(" ".join(issues))

    api_key = get_openai_api_key()
    os.environ["OPENAI_API_KEY"] = str(api_key)

    if AGENT_SESSION_KEY not in st.session_state:
        st.session_state[AGENT_SESSION_KEY] = SiteIntelligenceAgent(
            db_path=AGENT_DB_PATH,
            trace=False,
            log_path=None,
        )

    return st.session_state[AGENT_SESSION_KEY]


def clear_site_agent_conversation():
    agent = st.session_state.get(AGENT_SESSION_KEY)
    if agent is not None:
        agent.reset()

    st.session_state.pop(AGENT_SESSION_KEY, None)
    st.session_state[AGENT_MESSAGES_KEY] = []
    st.session_state[AGENT_QUESTION_COUNT_KEY] = 0


def build_site_brief_context(site_name, site_topic_detail, max_topics=AI_MAX_TOPICS):
    cols = [
        "topic_label_en",
        "negative",
        "neutral",
        "positive",
        "total_topic_reviews",
        "negative_rate",
        "net_sentiment_score",
        "priority_score",
    ]

    existing_cols = [col for col in cols if col in site_topic_detail.columns]

    rows = (
        site_topic_detail
        .sort_values(
            ["priority_score", "negative", "total_topic_reviews"],
            ascending=[False, False, False],
        )
        .head(max_topics)[existing_cols]
        .copy()
    )

    for col in ["negative_rate", "net_sentiment_score", "priority_score"]:
        if col in rows.columns:
            rows[col] = rows[col].round(3)

    return {
        "site": site_name,
        "top_topics": rows.to_dict(orient="records"),
        "metric_notes": {
            "negative_rate": "negative / total mentions",
            "net_sentiment_score": "(positive - negative) / total mentions; range -1 to 1",
            "priority_score": "log(1 + negative) * negative_rate",
        },
    }

def stream_site_brief(site_name, site_topic_detail):
    client = get_openai_client()

    if client is None:
        yield (
            "Live AI features are not configured. "
            "Add OPENAI_API_KEY to Streamlit secrets to enable this."
        )
        return

    context = build_site_brief_context(site_name, site_topic_detail)

    prompt = f"""
You are an operations analyst reviewing aggregated visitor-feedback analytics.

Use only the aggregated data below. Do not invent raw review examples.

Write a concise operational brief for the selected site.

The first part should be readable as an executive recommendation, not as a metrics report.
Prioritize plain-language interpretation before numbers.
Use numbers only when they clarify the recommendation.

Required format:

### Executive summary
- 2 short bullets.
- Start with the core operational issue and the overall recommended direction.
- Avoid listing many metrics here.

### Recommended actions
- 3 practical actions max, sorted by urgency.
- Each action should be specific enough for an operations manager to understand what to do next.
- Focus on what should be fixed, improved, monitored, or investigated.

### Main pain points
- 3 bullets max.
- Mention the topic names and only the most relevant numbers.
- Keep this section more analytical than the first two sections.

### Caveat
Only include this section if there is a site-specific caveat worth mentioning.
Do not include a generic disclaimer such as "this is based on aggregated model-derived analytics" unless it adds useful context.

Keep the entire answer under 180 words.

Aggregated data:
{json.dumps(context, ensure_ascii=False)}
""".strip()

    try:
        stream = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce short, practical operational insights from aggregated "
                        "visitor-feedback analytics. Be concise and do not invent data."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
            max_completion_tokens=900,
            reasoning_effort="minimal",
        )

        yielded_anything = False

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and getattr(delta, "content", None):
                yielded_anything = True
                yield delta.content

        if not yielded_anything:
            yield (
                "No visible text was returned by the model. "
                "Try switching AI_MODEL to 'gpt-5-mini' or increasing the output token budget."
            )

    except TypeError:
        # Fallback for SDK/API versions that do not accept reasoning_effort.
        stream = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce short, practical operational insights from aggregated "
                        "visitor-feedback analytics. Be concise and do not invent data."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
            max_completion_tokens=900,
        )

        yielded_anything = False

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and getattr(delta, "content", None):
                yielded_anything = True
                yield delta.content

        if not yielded_anything:
            yield (
                "No visible text was returned by the model. "
                "Try switching AI_MODEL to 'gpt-5-mini' or increasing the output token budget."
            )

    except Exception as e:
        yield f"AI brief generation failed: {type(e).__name__}: {e}"


def analyze_live_review(review_text: str):
    client = get_openai_client()

    if client is None:
        return None, (
            "Live AI features are not configured. "
            "Add OPENAI_API_KEY to Streamlit secrets to enable this."
        )

    review_text = str(review_text).strip()

    if not review_text:
        return None, "Please enter a review before running the analyzer."

    topic_schema_text = "\n".join(
        [f"- {topic_id}: {label}" for topic_id, label in LIVE_TOPIC_SCHEMA.items()]
    )

    json_schema = {
        "name": "live_review_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_summary": {
                    "type": "string",
                    "description": "One concise sentence summarizing the review."
                },
                "overall_sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative", "mixed"]
                },
                "segments": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "segment_text": {
                                "type": "string"
                            },
                            "topic_id": {
                                "type": "string",
                                "enum": list(LIVE_TOPIC_SCHEMA.keys())
                            },
                            "topic_label_en": {
                                "type": "string"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"]
                            },
                            "short_reason": {
                                "type": "string",
                                "description": "Brief explanation of why this topic/sentiment was assigned."
                            }
                        },
                        "required": [
                            "segment_text",
                            "topic_id",
                            "topic_label_en",
                            "sentiment",
                            "short_reason"
                        ]
                    }
                },
                "recommended_actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "string"
                    }
                },
                "caveat": {
                    "type": "string",
                    "description": "Short note about the live English-compatible path if relevant."
                }
            },
            "required": [
                "overall_summary",
                "overall_sentiment",
                "segments",
                "recommended_actions",
                "caveat"
            ]
        }
    }

    prompt = f"""
Analyze the visitor review below.

This is a live English-compatible demo path. It should use the same closed topic schema as the dashboard, but it is not the exact Hebrew batch pipeline.

Tasks:
1. Split the review into topic-specific segments.
2. Assign one topic_id from the closed schema to each segment.
3. Assign sentiment: positive, neutral, or negative.
4. Recommend practical operational actions.

Closed topic schema:
{topic_schema_text}

Rules:
- Use only the topic IDs listed above.
- If a sentence is generic praise or too vague, use "other".
- If a segment contains both praise and criticism about different issues, split it.
- Keep segment_text short and close to the user's wording.
- Do not invent facts.
- Write output in English.

Review:
{review_text}
""".strip()

    try:
        response = client.chat.completions.create(
            model=LIVE_ANALYZER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visitor-feedback analyst. "
                        "Return only structured JSON that matches the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            max_completion_tokens=1200,
            reasoning_effort="minimal",
        )

    except TypeError:
        response = client.chat.completions.create(
            model=LIVE_ANALYZER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visitor-feedback analyst. "
                        "Return only structured JSON that matches the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            max_completion_tokens=1200,
        )

    except Exception as e:
        return None, f"Live analysis failed: {type(e).__name__}: {e}"

    try:
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed, None

    except Exception as e:
        return None, f"Could not parse model response: {type(e).__name__}: {e}"


def calculate_y_dtick(max_y):
    max_y = int(max_y) if max_y else 0

    if max_y <= 20:
        return 1
    if max_y <= 100:
        return 5
    if max_y <= 250:
        return 10
    if max_y <= 1000:
        return 50
    return 100


def apply_chart_title_style(fig):
    fig.update_layout(
        title_font=dict(
            color=CHART_TITLE_COLOR,
            size=CHART_TITLE_SIZE,
        ),
        title_x=0.01,
        title_xanchor="left",
    )
    return fig

# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Nature Sites Intelligence Platform</h1>
        <p>
            AI-assisted analysis of visitor feedback across Israeli nature and heritage sites.<br>
            The dashboard identifies recurring operational topics, sentiment patterns, and site-level pain points
            from processed review segments.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# Main tabs
# ============================================================

tab_overview, tab_sites, tab_topics, tab_agent, tab_demo = st.tabs(
    [
        "Overview",
        "Site Deep Dive",
        "Topic Analysis",
        "Ask the Agent",
        "Try It Yourself",
    ]
)

# ============================================================
# Overview tab
# ============================================================

with tab_overview:

    st.markdown('<div class="section-title">Overview Filters</div>', unsafe_allow_html=True)

    all_sites = sorted(site_topic_df["site_name_en"].dropna().unique().tolist())
    all_topics = sorted(site_topic_df["topic_label_en"].dropna().unique().tolist())

    f1, f2 = st.columns(2)

    with f1:
        selected_sites = st.multiselect(
            "Sites",
            options=all_sites,
            default=[],
            placeholder="All sites",
            key="overview_selected_sites",
        )

    with f2:
        selected_topics = st.multiselect(
            "Topics",
            options=all_topics,
            default=[],
            placeholder="All topics",
            key="overview_selected_topics",
        )

    f3, f4 = st.columns(2)

    with f3:
        min_mentions = st.slider(
            "Minimum topic mentions",
            min_value=1,
            max_value=max(1, int(site_topic_df["total_topic_reviews"].max())),
            value=1,
            key="overview_min_mentions",
        )

    with f4:
        top_n = st.slider(
            "Rows to show in rankings",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="overview_top_n",
        )

    filtered_site_topic = site_topic_df.copy()

    if selected_sites:
        filtered_site_topic = filtered_site_topic[
            filtered_site_topic["site_name_en"].isin(selected_sites)
        ]

    if selected_topics:
        filtered_site_topic = filtered_site_topic[
            filtered_site_topic["topic_label_en"].isin(selected_topics)
        ]

    filtered_site_topic = filtered_site_topic[
        filtered_site_topic["total_topic_reviews"] >= min_mentions
    ].copy()

    filtered_site_topic = add_empirical_priority_context(
    filtered_site_topic,
    score_col="priority_score",
    prefix="view_priority",
    )

    total_sites = filtered_site_topic["site_name_en"].nunique()
    total_topics = filtered_site_topic["topic_id"].nunique()

    negative_mentions = int(filtered_site_topic["negative"].sum()) if "negative" in filtered_site_topic else 0
    neutral_mentions = int(filtered_site_topic["neutral"].sum()) if "neutral" in filtered_site_topic else 0
    positive_mentions = int(filtered_site_topic["positive"].sum()) if "positive" in filtered_site_topic else 0

    total_mentions = negative_mentions + neutral_mentions + positive_mentions
    negative_rate = negative_mentions / total_mentions if total_mentions else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Sites in view", number(total_sites), "Filtered site count")

    with col2:
        metric_card("Topics in view", number(total_topics), "Filtered topic count")

    with col3:
        metric_card("Topic mentions", number(total_mentions), "Aggregated review-topic mentions")

    with col4:
        metric_card("Negative share", pct(negative_rate), "Within current filters")

    st.markdown('<div class="section-title">Most mentioned topics</div>', unsafe_allow_html=True)

    topic_chart_mode = st.radio(
    "Topic chart style",
    options=[
        "Grouped Sentiment",
        "Stacked Sentiment",
    ],
    index=0,
    horizontal=True,
    key="overview_topic_chart_mode",
)

    if filtered_site_topic.empty:
        st.info("No rows match the current filters.")
    else:
        topic_hist = (
            filtered_site_topic
            .groupby(["topic_id", "topic_label_en", "topic_group_en"], dropna=False)
            .agg(
                negative=("negative", "sum"),
                neutral=("neutral", "sum"),
                positive=("positive", "sum"),
                total_topic_reviews=("total_topic_reviews", "sum"),
            )
            .reset_index()
        )

        topic_hist = topic_hist.sort_values(
            "total_topic_reviews",
            ascending=False,
        ).head(top_n)

        topic_order = topic_hist["topic_label_en"].tolist()

        topic_long = topic_hist.melt(
            id_vars=[
                "topic_id",
                "topic_label_en",
                "topic_group_en",
                "total_topic_reviews",
            ],
            value_vars=["negative", "neutral", "positive"],
            var_name="sentiment",
            value_name="count",
        )

        topic_long["topic_label_en"] = pd.Categorical(
            topic_long["topic_label_en"],
            categories=topic_order,
            ordered=True,
        )

        topic_long = add_sentiment_display_column(topic_long)

        barmode = "stack" if topic_chart_mode == "Stacked Sentiment" else "group"
        max_y = int(topic_hist["total_topic_reviews"].max()) if len(topic_hist) else 0

        fig = px.bar(
            topic_long,
            x="topic_label_en",
            y="count",
            color="sentiment_display",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            barmode=barmode,
           category_orders={
    "topic_label_en": topic_order,
    "sentiment_display": SENTIMENT_ORDER,
},
            hover_data=[
                "topic_group_en",
                "total_topic_reviews",
            ],
            labels=PLOTLY_LABELS,
            title = "",
        
        )

        fig.update_layout(
            height=560,
            margin=dict(l=20, r=20, t=55, b=140),
            xaxis_title="Topic",
            yaxis_title="Number of mentions",
            xaxis_tickangle=-45,
            legend_title_text="Sentiment",
        )

        fig.update_yaxes(
            tickformat=",d",
            dtick=calculate_y_dtick(max_y),
            rangemode="tozero",
        )
        
        fig = apply_plot_readability(fig)
        st.plotly_chart(fig, use_container_width=True)

        
        st.markdown('<div class="section-title">Site-topic ranking</div>', unsafe_allow_html=True)

        sort_metric = st.selectbox(
            "Sort ranking by",
            options=[
                "total_topic_reviews",
                "priority_score",
                "negative",
                "negative_rate",
                "net_sentiment_score",
            ],
            index=0,
            format_func=pretty_metric_name,
            key="site_topic_ranking_sort_metric",
            help="Choose how to rank site-topic combinations in the chart below.",
        )

        ranking_ascending = True if sort_metric == "net_sentiment_score" else False

        ranking_df = filtered_site_topic.sort_values(
            by=sort_metric,
            ascending=ranking_ascending,
        ).head(top_n)

        chart_df = ranking_df.copy()
        chart_df["site_topic"] = chart_df["site_name_en"] + " · " + chart_df["topic_label_en"]

        fig = px.bar(
            chart_df.sort_values(sort_metric, ascending=True),
            x=sort_metric,
            y="site_topic",
            orientation="h",
            hover_data={
            "negative": ":,d",
            "neutral": ":,d",
            "positive": ":,d",
            "total_topic_reviews": ":,d",
            "negative_rate": ":.1%",
            "net_sentiment_score": ":.2f",
            "priority_score": ":.2f",
            "view_priority_rank_label": True,
            "view_priority_severity_percentile": ":.0%",
            "view_priority_min": ":.2f",
            "view_priority_max": ":.2f",
        },
            labels=PLOTLY_LABELS,
            title = "",
        )

        fig.update_layout(
            height=max(450, 24 * len(chart_df)),
            margin=dict(l=20, r=20, t=55, b=20),
            xaxis_title=pretty_metric_name(sort_metric),
            yaxis_title="",
        )

        if sort_metric in ["negative", "total_topic_reviews"]:
            fig.update_xaxes(tickformat=",d")
        
        fig = apply_plot_readability(fig)
        st.plotly_chart(fig, use_container_width=True)

        table_cols = [
            "site_name_en",
            "topic_label_en",
            "negative",
            "neutral",
            "positive",
            "total_topic_reviews",
            "negative_rate",
            "net_sentiment_score",
            "priority_score",
        ]

        

    st.markdown('<div class="section-title">Sentiment mix across selected data</div>', unsafe_allow_html=True)

    sentiment_summary = pd.DataFrame(
    {
        "sentiment": ["Positive", "Neutral", "Negative"],
        "count": [positive_mentions, neutral_mentions, negative_mentions],
    }
)

    fig = px.pie(
    sentiment_summary,
    names="sentiment",
    values="count",
    color="sentiment",
    color_discrete_map=SENTIMENT_COLOR_MAP,
    category_orders={
        "sentiment": SENTIMENT_ORDER,
    },
    hole=0.48,
    labels=PLOTLY_LABELS,
    title = "",
)

    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    fig = apply_plot_readability(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Site deep dive tab
# ============================================================

with tab_sites:
    st.markdown('<div class="section-title">Site deep dive</div>', unsafe_allow_html=True)

    site_options = sorted(site_topic_df["site_name_en"].dropna().unique().tolist())

    default_site = site_options[0] if site_options else None

    selected_site = st.selectbox(
        "Choose a site",
        options=site_options,
        index=site_options.index(default_site) if default_site in site_options else 0,
        key="site_deep_dive_selector",
    )

    site_topic_detail = site_topic_df[
        site_topic_df["site_name_en"].eq(selected_site)
    ].copy()

    site_topic_detail = site_topic_detail.sort_values(
        ["priority_score", "negative", "total_topic_reviews"],
        ascending=[False, False, False],
    )

    site_row = site_stats_df[
        site_stats_df["site_name_en"].eq(selected_site)
    ].copy()

    if not site_row.empty:
        site_row = site_row.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        total_site_mentions = int(site_row.get("total_reviews_with_mentions", 0))
        site_negative_rate = float(site_row.get("negative_rate", 0))
        site_net = float(site_row.get("net_sentiment_score", 0))
        site_priority = float(site_row.get("priority_score", 0))

        with c1:
            metric_card("Mentions", number(total_site_mentions), selected_site)

        with c2:
            metric_card("Negative share", pct(site_negative_rate), "Site-level")

        with c3:
            metric_card("Net sentiment", f"{site_net:.2f}", "Positive minus negative share")

        with c4:
            metric_card(
                "Priority score",
                f"{site_priority:.2f}",
                priority_context_text(site_row, "site_priority"),
            )

    if site_topic_detail.empty:
        st.info("No topic data for this site.")
    else:
        sentiment_site_long = sentiment_long_format(
            site_topic_detail,
            id_cols=["topic_label_en"],
            total_col="total_topic_reviews",
        )
        sentiment_site_long = add_sentiment_display_column(sentiment_site_long)

        fig = px.bar(
            sentiment_site_long,
            x="count",
            y="topic_label_en",
            color="sentiment_display",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            orientation="h",
            category_orders={
            "sentiment_display": SENTIMENT_ORDER,
            },
            labels=PLOTLY_LABELS,
            title = "",
        )

        fig.update_layout(
            barmode="stack",
            height=max(480, 26 * site_topic_detail["topic_label_en"].nunique()),
            margin=dict(l=20, r=20, t=55, b=20),
            xaxis_title="Mentions",
            yaxis_title="Topic",
            legend_title_text="Sentiment",
        )

        fig.update_xaxes(tickformat=",d", rangemode="tozero")
        fig = apply_plot_readability(fig)
        st.plotly_chart(fig, use_container_width=True)

       

    st.markdown('<div class="section-title">AI site brief</div>', unsafe_allow_html=True)

    st.caption(
        "This summary is generated from aggregated topic-sentiment statistics only. "
        "Raw review text is not sent to the model."
    )

    brief_cache_key = f"site_brief::{selected_site}"

    if st.button("Generate AI site brief", type="primary", key="generate_ai_site_brief_for_selected_site"):
        st.session_state.pop(brief_cache_key, None)

        st.info(f"Generating AI brief for {selected_site}...")

        brief_container = st.container(border=True)

        with brief_container:
            brief_text = st.write_stream(
                stream_site_brief(selected_site, site_topic_detail)
            )

        if brief_text and str(brief_text).strip():
            st.session_state[brief_cache_key] = brief_text
        else:
            st.warning("The model call completed but returned no visible text.")

    elif brief_cache_key in st.session_state:
        st.container(border=True).markdown(st.session_state[brief_cache_key])


# ============================================================
# Topic analysis tab
# ============================================================

with tab_topics:
    st.markdown('<div class="section-title">Topic analysis</div>', unsafe_allow_html=True)

    topic_options = sorted(site_topic_df["topic_label_en"].dropna().unique().tolist())

    default_topic = (
    "Crowding"
    if "Crowding" in topic_options
    else topic_options[0]
        )   

    selected_topic_for_analysis = st.selectbox(
        "Choose a topic",
        options=topic_options,
        index=topic_options.index(default_topic) if default_topic in topic_options else 0,
        key="topic_analysis_selector",
    )

    topic_site_detail = site_topic_df[
        site_topic_df["topic_label_en"].eq(selected_topic_for_analysis)
    ].copy()

    topic_site_detail = add_empirical_priority_context(
    topic_site_detail,
    score_col="priority_score",
    prefix="topic_view_priority",
    )

    topic_site_detail = topic_site_detail.sort_values(
        ["priority_score", "negative", "total_topic_reviews"],
        ascending=[False, False, False],
    )

    topic_summary_row = topic_stats_df[
        topic_stats_df["topic_label_en"].eq(selected_topic_for_analysis)
    ].copy()

    if not topic_summary_row.empty:
        topic_summary_row = topic_summary_row.iloc[0]

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            metric_card(
                "Mentions",
                number(topic_summary_row.get("total_topic_reviews", 0)),
                selected_topic_for_analysis,
            )

        with t2:
            metric_card(
                "Negative share",
                pct(topic_summary_row.get("negative_rate", 0)),
                "Across all sites",
            )

        with t3:
            metric_card(
                "Net sentiment",
                f"{float(topic_summary_row.get('net_sentiment_score', 0)):.2f}",
                "Topic-level",
            )

        with t4:
            metric_card(
                "Priority score",
                f"{float(topic_summary_row.get('priority_score', 0)):.2f}",
                priority_context_text(topic_summary_row, "topic_priority", "Topic-level triage"),
            )

    if topic_site_detail.empty:
        st.info("No site-level data for this topic.")
    else:
        st.markdown(
            f'<div class="section-title">Sites most affected by: {selected_topic_for_analysis}</div>',
            unsafe_allow_html=True,
        )

        top_topic_sites = topic_site_detail.head(top_n).copy()

        fig = px.bar(
            top_topic_sites.sort_values("priority_score", ascending=True),
            x="priority_score",
            y="site_name_en",
            orientation="h",
            hover_data={
                "negative": ":,d",
                "neutral": ":,d",
                "positive": ":,d",
                "total_topic_reviews": ":,d",
                "negative_rate": ":.1%",
                "net_sentiment_score": ":.2f",
                "priority_score": ":.2f",
                "topic_view_priority_rank_label": True,
                "topic_view_priority_severity_percentile": ":.0%",
                "topic_view_priority_min": ":.2f",
                "topic_view_priority_max": ":.2f",
            },
            labels=PLOTLY_LABELS,
           title = "",
        )

        fig.update_layout(
            height=max(480, 26 * len(top_topic_sites)),
            margin=dict(l=20, r=20, t=55, b=20),
            xaxis_title="Priority score",
            yaxis_title="Site",
        )
        
        fig = apply_plot_readability(fig)
        st.plotly_chart(fig, use_container_width=True)

        topic_site_long = sentiment_long_format(
            top_topic_sites,
            id_cols=["site_name_en"],
            total_col="total_topic_reviews",
        )
        topic_site_long = add_sentiment_display_column(topic_site_long)

        fig = px.bar(
            topic_site_long,
            x="site_name_en",
            y="count",
            color="sentiment_display",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            barmode="group",
            category_orders={
            "sentiment_display": SENTIMENT_ORDER,
            },
            labels=PLOTLY_LABELS,
            title = "",
        )

        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=55, b=120),
            xaxis_title="Site",
            yaxis_title="Mentions",
            xaxis_tickangle=-45,
            legend_title_text="Sentiment",
        )

        fig.update_yaxes(tickformat=",d", rangemode="tozero")
        fig = apply_plot_readability(fig)
        st.plotly_chart(fig, use_container_width=True)

        


# ============================================================
# Site intelligence agent tab
# ============================================================

with tab_agent:
    initialize_agent_state()

    title_col, clear_col = st.columns([5, 1])

    with title_col:
        st.markdown(
            '<div class="section-title">Ask the Site Intelligence Agent</div>',
            unsafe_allow_html=True,
        )

    with clear_col:
        st.button(
            "Clear chat",
            key="clear_site_agent_chat",
            on_click=clear_site_agent_conversation,
            use_container_width=True,
        )

    st.caption(
        "Ask about official site rules, compare visitor feedback, or describe "
        "your constraints. Official facts and review-derived findings are kept separate."
    )

    readiness_issues = get_agent_readiness_issues()
    agent_ready = not readiness_issues

    if readiness_issues:
        st.warning("The live agent is not ready in this environment.")
        with st.expander("Setup diagnostics", expanded=True):
            for issue in readiness_issues:
                st.markdown(f"- {issue}")
            if AGENT_IMPORT_ERROR:
                st.caption(f"Import detail: {AGENT_IMPORT_ERROR}")

    selected_example = None

    with st.expander(
        "Try an example question",
        expanded=False,
    ):
        prompt_columns = st.columns(2)

        example_prompts = [
            {
                "title": "Family trip with a dog and water",
                "summary": (
                    "Which northern sites fit a family with a dog and offer either "
                    "a water view or permitted water access?"
                ),
                "prompt": (
                    "For a family with a dog, which site in northern Israel is "
                    "recommended if we also want either a water view or permitted "
                    "access to the water? Treat dog access as mandatory and water "
                    "as a flexible preference. Present the matching options in a "
                    "small table using only those criteria."
                ),
            },
            {
                "title": "Achziv or Caesarea for a family day trip?",
                "summary": (
                    "Which is better for water access, dogs, accessibility, parking, "
                    "restrooms, and family facilities?"
                ),
                "prompt": (
                    "Compare Achziv and Caesarea for a family day trip using only "
                    "these criteria: water setting, permitted water access, dog policy, "
                    "accessibility, parking, restrooms, and family facilities. Use a "
                    "compact table and finish with one recommendation."
                ),
            },
            {
                "title": "Can I bring a dog to Ein Gedi?",
                "summary": (
                    "Check the official dog rule and where visitors may enter the water."
                ),
                "prompt": (
                    "Can I bring a dog to Ein Gedi, and where is visitor water entry "
                    "permitted? Give only the relevant official rules and keep the "
                    "answer concise."
                ),
            },
            {
                "title": "Where can I picnic and use a BBQ?",
                "summary": (
                    "Find northern sites with picnic facilities and separate BBQ "
                    "permission from bonfire permission."
                ),
                "prompt": (
                    "Which sites in northern Israel allow BBQs and also provide picnic "
                    "facilities? Show up to four best options in a compact table with "
                    "only these rows: BBQ, bonfire, picnic facilities, and official page."
                ),
            },
        ]

        for index, example in enumerate(example_prompts):
            with prompt_columns[index % 2]:
                with st.container(border=True):
                    st.markdown(f"**{example['title']}**")
                    st.caption(example["summary"])

                    if st.button(
                        "Ask this question",
                        key=f"site_agent_example_{index}",
                        use_container_width=True,
                        disabled=not agent_ready,
                    ):
                        selected_example = example["prompt"]

    # Start compact so the input is visible immediately, then grow as the
    # conversation accumulates. Longer histories scroll inside the container.
    current_message_count = len(
        st.session_state[AGENT_MESSAGES_KEY]
    )

    if current_message_count == 0:
        chat_height = 170
    else:
        # Once a conversation exists, prioritize space for answers and tables.
        chat_height = 470

    chat_history = st.container(
        height=chat_height,
        border=True,
    )

    typed_prompt = st.chat_input(
        "Ask about a site, compare locations, or describe your constraints...",
        key="site_agent_chat_input",
        max_chars=1200,
        disabled=not agent_ready,
    )

    prompt = selected_example or typed_prompt

    def render_agent_message(message):
        role = message.get("role", "assistant")
        avatar = "🌿" if role == "assistant" else None

        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))

    if not prompt:
        with chat_history:
            messages = st.session_state[AGENT_MESSAGES_KEY]

            if not messages:
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(
                        "Ask me about a site, compare two locations, or describe "
                        "the conditions your visit must meet."
                    )
            else:
                for message in messages:
                    render_agent_message(message)

    else:
        question_count = int(
            st.session_state.get(AGENT_QUESTION_COUNT_KEY, 0)
        )

        if question_count >= AGENT_MAX_QUESTIONS_PER_SESSION:
            with chat_history:
                for message in st.session_state[AGENT_MESSAGES_KEY]:
                    render_agent_message(message)

                st.warning(
                    "This demo session reached its question limit. "
                    "Clear the chat to start a new session."
                )

        else:
            st.session_state[AGENT_MESSAGES_KEY].append(
                {"role": "user", "content": prompt}
            )

            answer = None

            with chat_history:
                for message in st.session_state[AGENT_MESSAGES_KEY]:
                    render_agent_message(message)

                with st.chat_message("assistant", avatar="🌿"):
                    activity_placeholder = st.empty()
                    activity_state = {"last": None}

                    def render_activity(message):
                        safe_message = escape(str(message).strip())
                        activity_placeholder.markdown(
                            f"""
                            <div class="agent-activity-line">
                                <span class="agent-activity-dot"></span>
                                <span>{safe_message}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    def on_agent_event(message):
                        message = str(message).strip()
                        if not message or message == activity_state["last"]:
                            return

                        activity_state["last"] = message
                        render_activity(message)

                    render_activity("Working on your request...")

                    try:
                        answer = get_site_intelligence_agent().ask(
                            prompt,
                            on_event=on_agent_event,
                        )

                    except Exception:
                        st.session_state.pop(AGENT_SESSION_KEY, None)
                        activity_placeholder.empty()
                        st.error(
                            "The agent could not complete this request. "
                            "Please try again or clear the conversation."
                        )

                    if answer:
                        # The colored activity line is intentionally transient.
                        activity_placeholder.empty()
                        st.markdown(answer)

            if answer:
                st.session_state[AGENT_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )
                st.session_state[AGENT_QUESTION_COUNT_KEY] = (
                    question_count + 1
                )

                # Re-render once so the chat container immediately adopts the
                # correct height for the newly stored conversation.
                st.rerun()

    remaining_questions = max(
        0,
        AGENT_MAX_QUESTIONS_PER_SESSION
        - int(st.session_state.get(AGENT_QUESTION_COUNT_KEY, 0)),
    )

    st.caption(
        f"{remaining_questions} demo questions remaining in this session · "
        "Verify time-sensitive operational rules on the linked official page."
    )


# ============================================================
# Try it yourself placeholder
# ============================================================

with tab_demo:
    st.markdown('<div class="section-title">Try It Yourself</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note-box">
        This live demo supports English visitor reviews using a multilingual LLM path.
        The historical dashboard was built from Hebrew reviews using a batch segmentation pipeline
        and a Hebrew sentiment model, so live-demo results may differ from the offline production pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    initialize_sample_review_state()

    col_review_title, col_suggest = st.columns([3, 1])

    with col_review_title:
        st.markdown("**Paste or edit an English visitor review**")

    with col_suggest:
        st.button(
            "Suggest another review",
            key="suggest_another_review",
            on_click=suggest_next_review,
            use_container_width=True,
        )

    example_review = st.text_area(
        label="Review text",
        key="live_review_text",
        height=80,
        label_visibility="collapsed",
        placeholder="Paste or edit an English visitor review...",
    )

    st.caption(
        f"Sample review {st.session_state.sample_review_index + 1} of {len(SAMPLE_REVIEWS)}. "
        "You can edit the text before analysis."
    )

    analyze_clicked = st.button(
        "Analyze review",
        type="primary",
        key="run_live_review_analyzer",
    )

    if analyze_clicked:
        log_placeholder = st.empty()

        # UI-only process trace. The backend currently runs one structured LLM call.
        for step_idx in range(len(LIVE_PROCESS_STEPS)):
            render_terminal_logs(log_placeholder, active_step=step_idx)
            time.sleep(0.10)

        live_result, live_error = analyze_live_review(example_review)

        render_terminal_logs(
            log_placeholder,
            active_step=len(LIVE_PROCESS_STEPS),
            done=True,
        )

        time.sleep(0.15)

        log_placeholder.empty()

        if live_error:
            st.warning(live_error)

        elif live_result:
            st.markdown('<div class="section-title">Live analysis result</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown("**Executive summary**")
                st.write(live_result.get("overall_summary", ""))

            with c2:
                sentiment = live_result.get("overall_sentiment", "mixed")
                st.markdown("**Overall sentiment**")
                st.markdown(f"`{str(sentiment).title()}`")

            segments = live_result.get("segments", [])

            if segments:
                st.markdown("**Detected topics and sentiment**")

                SENTIMENT_MARKDOWN_COLOR = {
                    "positive": "blue",
                    "neutral": "gray",
                    "negative": "red",
                }

                SENTIMENT_DISPLAY_LABEL = {
                    "positive": "Positive",
                    "neutral": "Neutral",
                    "negative": "Negative",
                }

                for idx, segment in enumerate(segments, start=1):
                    topic = str(segment.get("topic_label_en", "Other")).strip()
                    sentiment_raw = str(segment.get("sentiment", "neutral")).lower().strip()
                    segment_text = str(segment.get("segment_text", "")).strip()
                    reason = str(segment.get("short_reason", "")).strip()

                    sentiment_label = SENTIMENT_DISPLAY_LABEL.get(sentiment_raw, "Neutral")
                    sentiment_color = SENTIMENT_MARKDOWN_COLOR.get(sentiment_raw, "gray")

                    with st.container(border=True):
                        header_left, header_right = st.columns([4, 1])

                        with header_left:
                            st.markdown(
                                f"""
                                <div style="
                                    font-weight: 760;
                                    text-decoration: underline;
                                    text-underline-offset: 4px;
                                    margin-bottom: 0.25rem;
                                ">
                                    Segment {idx} · {topic}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with header_right:
                            st.markdown(
                                f":{sentiment_color}[**{sentiment_label}**]"
                            )

                        if segment_text:
                            st.markdown(
                                f"> “{segment_text}”"
                            )

                        if reason:
                            st.markdown(
                                f"*{reason}*"
                            )

                sentiment_counts = (
                    pd.DataFrame(segments)
                    .assign(sentiment=lambda x: x["sentiment"].str.title())
                    .groupby("sentiment")
                    .size()
                    .reset_index(name="count")
                )

                fig = px.bar(
                    sentiment_counts,
                    x="sentiment",
                    y="count",
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                    category_orders={
                        "sentiment": SENTIMENT_ORDER,
                    },
                    labels=PLOTLY_LABELS,
                    title="",
                )

                fig.update_layout(
                    height=330,
                    margin=dict(l=20, r=20, t=15, b=40),
                    xaxis_title="Sentiment",
                    yaxis_title="Number of Segments",
                    showlegend=False,
                )

                fig.update_yaxes(tickformat=",d", dtick=1, rangemode="tozero")
                fig.update_layout(title_text="")

                fig = soften_sparse_bars(
                    fig,
                    n_categories=sentiment_counts["sentiment"].nunique(),
                    grouped=False,
                )
                fig = apply_plot_readability(fig)
                st.plotly_chart(fig, use_container_width=True)

            actions = live_result.get("recommended_actions", [])

            if actions:
                st.markdown("**Recommended actions**")
                for action in actions:
                    st.markdown(f"- {action}")

            caveat = str(live_result.get("caveat", "")).strip()

            if caveat:
                st.caption(caveat)