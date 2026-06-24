# Nature Sites Intelligence Platform

An end-to-end visitor-feedback intelligence product for Israeli nature, heritage, and recreation sites.

The project combines a historical NLP pipeline, interactive analytics, structured official-site information, and a tool-calling decision-support agent in one Streamlit application.

## Live Demo

[https://nature-sites-intelligence.streamlit.app/](https://nature-sites-intelligence.streamlit.app/)

## What the Product Does

The application supports two complementary workflows:

1. **Operational intelligence**  
   Analyze recurring visitor-experience topics, sentiment patterns, and site-level pain points.

2. **Visitor decision support**  
   Ask natural-language questions about official site rules, facilities, comparisons, and recommendations.

Examples include:

- Can I bring a dog to Ein Gedi?
- Which northern sites allow BBQs and provide picnic facilities?
- Which site is a better fit for a family with a dog that also wants access to water?
- Compare Achziv and Caesarea for water access, accessibility, and family facilities.
- What do visitors say about cleanliness or crowding at a selected site?

## Main Application Areas

### Overview

Explore topic frequency, sentiment distribution, and site-topic rankings across the available sites.

### Site Deep Dive

Inspect one site in detail:

- total topic mentions
- negative share
- net sentiment
- operational priority score
- topic-level sentiment distribution
- AI-generated operational brief based on aggregated statistics

### Topic Analysis

Select one operational topic and identify which sites are most affected by it.

### Ask the Site Intelligence Agent

The agent combines two distinct evidence layers:

- **Official site facts:** rules, facilities, dog access, water setting and permitted entry, accessibility, parking, restrooms, picnic areas, BBQ and bonfire rules, camping, admission, and official links.
- **Visitor-feedback intelligence:** topic-level sentiment, evidence volume, representative excerpts, and review-based experience patterns.

The agent follows several product rules:

- Official requirements determine whether a site is eligible.
- Visitor reviews may be used as a secondary tie-breaker among eligible sites.
- Review count is not treated as popularity or quality.
- Answers include only the dimensions requested by the user.
- Internal ranking scores are hidden unless analytics are explicitly requested.
- When several sites share the same criteria, the answer uses an adaptive Markdown table:
  - the larger dimension is placed in rows;
  - the smaller dimension is placed in columns;
  - ties place criteria in rows and sites in columns.
- Broader comparisons are offered only after the concise answer, with the proposed additional criteria listed in advance.

The chat interface includes:

- a compact empty-state conversation area
- a larger scrollable history after the first exchange
- example-question cards with plain-language descriptions
- a single transient activity line while tools are running
- no persisted execution trace in the final answer

### Try It Yourself

Paste or edit an English visitor review and receive:

- topic-specific segmentation
- closed-schema topic classification
- segment-level sentiment
- a concise review summary
- recommended operational actions

This is an English-compatible live demonstration path. The historical production dataset was processed primarily from Hebrew reviews.

## Agent Architecture

```text
User question
    ↓
LLM selects deterministic tools
    ↓
Python / SQLite tool layer
    ├── resolve site names
    ├── filter by official requirements
    ├── retrieve official site facts
    ├── retrieve site-topic statistics
    ├── compare review-derived patterns
    ├── rank eligible sites
    └── retrieve representative evidence
    ↓
Grounded, scope-limited answer
```

The language model selects tools and explains their outputs. Filtering, statistics, comparisons, and rankings are calculated by deterministic code rather than by the model.

## Data Pipeline

The historical dataset was processed through the following stages:

1. Normalize site names.
2. Segment each review into topic-specific units.
3. Assign each segment to a closed topic schema.
4. Classify sentiment at the segment level.
5. Merge duplicate review-topic-sentiment mentions.
6. Aggregate statistics by site, topic, and sentiment.
7. Extract structured information from public official site pages.
8. Build deterministic agent tools and curated evidence tables.
9. Generate public dashboard-ready Parquet and SQLite assets.

## Dataset and Public Agent Database

The public agent database contains:

- 71 sites
- 25 visitor-experience topics
- 1,775 site-topic combinations
- aggregated review statistics
- structured official-site information
- a limited set of short representative review excerpts

The public repository does **not** contain:

- the full raw review corpus
- reviewer identities
- full original review text
- internal review identifiers
- private scraping outputs
- local development paths

Representative excerpts are included only to illustrate aggregate patterns and do not contain reviewer metadata.

## Metrics

### Negative Share

$$\frac{\text{negative}}{\text{negative + neutral + positive}}$$

### Net Sentiment Score

$$\frac{\text{positive - negative}}{\text{negative + neutral + positive}}$$


Range: $-1$ to $1$.

### Priority Score

$$\log(\text{1 + negative}) × \text{negative_share}$$

This is an operational triage metric that combines negative-feedback volume with negative concentration. It is not presented as statistical ground truth.

## Technology Stack

- Python
- Streamlit
- SQLite
- Pandas
- Plotly
- PyArrow / Parquet
- OpenAI API
- Responses API tool calling
- LLM-based topic segmentation
- Transformer-based sentiment analysis

## Repository Structure

```text
app/
  streamlit_app.py

agent/
  __init__.py
  agent_tools.py
  site_intelligence_agent.py

data/
  public/
    nip_agent_public.db
    site_topic_sentiment_stats_public.parquet
    site_sentiment_stats_public.parquet
    topic_sentiment_stats_public.parquet
    app_filters.json
    app_metadata.json
    site_name_translation_table.csv
    topic_translation_table.csv

scripts/
  build_public_agent_db.py

.streamlit/
  secrets.toml.example

requirements.txt
README.md
```

## Local Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "your-key"
```

Run the application from the repository root:

```powershell
python -m streamlit run app/streamlit_app.py
```

Do not commit the real secrets file.

## Deployment

The app is designed for Streamlit Community Cloud.

Configure `OPENAI_API_KEY` in the app's Streamlit Secrets settings. The public SQLite and Parquet assets are committed with the application; private batch files and local secrets remain excluded.

## Reliability and Product Constraints

- Official and operational details may change and should be confirmed on the linked official page before a visit.
- Review-derived findings describe patterns in the stored dataset, not real-time conditions.
- Review volume is treated as evidence volume, not site popularity.
- Missing official information is reported as unconfirmed rather than inferred.
- The public interface is English-first.
- Representative Hebrew evidence may be translated by the answering model.
- The dashboard is intended for decision support and exploratory analysis, not as a formal statistical audit.

## Project Goal

The project demonstrates how an NLP workflow can become a complete data product:

- large-scale unstructured-text processing
- structured extraction
- topic and sentiment modeling
- operational analytics
- public-safe data engineering
- deterministic tool use
- grounded recommendations
- interactive deployment
