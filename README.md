# Nature Sites Intelligence Platform

AI-assisted dashboard for analyzing visitor feedback across Israeli nature and heritage sites.

The system processes visitor reviews into topic-specific segments, assigns sentiment, aggregates site/topic-level pain points, and provides an interactive dashboard for operational insight.

## Live Demo

[https://nature-sites-intelligence.streamlit.app/](https://nature-sites-intelligence.streamlit.app/)

## Project Overview

This project demonstrates an end-to-end visitor-feedback intelligence workflow:

- LLM-based segmentation of free-text visitor reviews
- Closed-schema topic classification
- Segment-level sentiment analysis
- Aggregated site/topic analytics
- Interactive Streamlit dashboard
- AI-generated operational summaries for selected sites
- English-compatible live demo path for ad-hoc review analysis

The dashboard is designed for operational users who need to quickly identify recurring issues, compare sites, and prioritize improvements.

## Core Features

### 1. Overview Dashboard

Shows the most frequently mentioned visitor-experience topics and sentiment distribution across the selected data.

### 2. Site Deep Dive

Allows filtering by site and inspecting:

- Topic-level sentiment distribution
- Site-level negative share
- Net sentiment score
- Priority score
- AI-generated site brief based on aggregated statistics

### 3. Topic Analysis

Allows selecting a topic and identifying which sites are most affected by that issue.

### 4. Try It Yourself

Provides an English-compatible live demo where users can paste or cycle through sample visitor reviews and receive:

- Topic-specific segmentation
- Topic classification
- Sentiment classification
- Recommended operational actions

## Methodology

The historical dataset was processed using a batch pipeline:

1. Normalize site names
2. Segment each review into topic-specific units
3. Assign each segment to a closed topic schema
4. Run sentiment analysis at segment level
5. Merge duplicate review-topic-sentiment mentions
6. Aggregate metrics by site, topic, and sentiment
7. Build public dashboard-ready tables
8. Generate AI summaries from aggregated statistics only

## Metrics

### Negative Share


$$\frac{\text{negative}}{\text{negative} + \text{neutral} + \text{positive}}$$

Represents the share of negative sentiment mentions within a selected site/topic scope.

### Net Sentiment Score

$$\frac{\text{positive} - \text{negative}}{\text{negative} + \text{neutral} + \text{positive}}$$

Range: -1 to 1.
Higher values indicate more positive sentiment concentration. Lower values indicate more negative concentration.

### Priority Score

$$\log(1 + \text{negative}) \times \text{negative rate}$$

Used as an operational triage metric. It combines complaint volume with the concentration of negative feedback, so large sites do not dominate rankings purely because they receive more reviews.

## Data Privacy

Raw visitor review text is not included in this public repository.

The public dashboard uses only aggregated, precomputed statistics and public-safe metadata.

The live demo uses user-provided or sample English reviews and is separate from the historical Hebrew batch pipeline.

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- PyArrow / Parquet
- OpenAI API
- LLM-based topic segmentation
- Transformer-based sentiment analysis

## Repository Structure

```text
app/
  streamlit_app.py

data/
  public/
    site_topic_sentiment_stats_public.parquet
    site_sentiment_stats_public.parquet
    topic_sentiment_stats_public.parquet
    app_filters.json
    app_metadata.json
    site_name_translation_table.csv
    topic_translation_table.csv

requirements.txt
README.md
```

## Notes and Limitations

- The historical pipeline was optimized for Hebrew visitor reviews.
- The public live demo uses an English-compatible LLM path, so results may differ from the offline Hebrew pipeline.
- The dashboard is intended for operational prioritization and exploratory analysis, not as a formal statistical audit.
- Public data files exclude raw reviews and expose only aggregated outputs.

## Project Goal

The goal of this project is to demonstrate how modern LLM workflows can turn unstructured visitor feedback into an operational intelligence product: topic extraction, sentiment analysis, prioritization, dashboarding, and live AI-assisted interpretation.
