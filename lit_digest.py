#!/usr/bin/env python3
"""
Weekly literature digest for arXiv / bioRxiv / journals.

- Fetches RSS feeds
- Filters by keywords
- Ranks by semantic similarity to canonical papers
- Writes Markdown digest
- Optionally posts top-N to Slack via incoming webhook

Configurable in the CONFIG section below.
"""

import os
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import feedparser
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util


# ==========================
# ======== CONFIG ==========
# ==========================

# How many days back to look
LOOKBACK_DAYS = 7

# Max items to consider per feed before filtering/ranking
MAX_ITEMS_PER_FEED = 200

# Number of top papers to include in the digest
TOP_K = 60

# Number of top papers to optionally post to Slack
TOP_K_SLACK = 15

# Where to save the markdown digest
OUTPUT_DIR = "./digests"

# Optional Slack incoming webhook URL (set as env var or paste string here)
SLACK_WEBHOOK_URL = os.getenv("LIT_DIGEST_SLACK_WEBHOOK", "").strip()

# ---- Feeds ----
# Add / edit as you like. URLs are RSS / Atom endpoints.

FEEDS = [
    # arXiv (examples)
    {
        "name": "arXiv q-bio",
        "url": "https://export.arxiv.org/rss/q-bio",
    },
    {
        "name": "arXiv cs.LG",
        "url": "https://export.arxiv.org/rss/cs.LG",
    },

    # bioRxiv – you should adjust to your preferred subject feeds.
    # Check biorxiv "RSS" page for more granular subjects if you want.
    {
        "name": "bioRxiv Genomics+Bioinformatics",
        "url": "https://connect.biorxiv.org/biorxiv_xml.php?subject=genomics+bioinformatics",
    },

    # Journals – many have RSS links in their “Alerts” or “RSS” pages.
    # Put whatever you care about here.
    {
        "name": "Genome Research",
        "url": "https://genome.cshlp.org/rss/current.xml",
    },
    # {
    #     "name": "Genome Biology",
    #     "url": "https://genomebiology.biomedcentral.com/articles/most-recent/rss",
    # },
    {
        "name": "Nature Genetics",
        "url": "https://www.nature.com/ng.rss",
    },
    {
        "name": "Nature Methods",
        "url": "https://www.nature.com/nmeth.rss",
    },
    # {
    #     "name": "Cell Genomics",
    #     "url": "https://www.cell.com/cell-genomics/rss",
    # },
    {
        "name": "Nature Biotechnology",
        "url": "https://www.nature.com/nbt/current_issue/rss/",
    },
]

# ---- Keyword filters ----
# If INCLUDE_KEYWORDS is non-empty, keep items that match at least one of them in
# title or summary. Matching is case-insensitive simple substring.

INCLUDE_KEYWORDS = [
    "single-cell",
    "scRNA-seq",
    "scrna",
    "perturbation",
    "multi-omic",
    "multi omic",
    "multiomics",
    "spatial transcriptomics",
    "phosphoproteomic",
    "chromatin accessibility",
    "ATAC-seq",
    "gene regulatory",
    "foundation model",
    "transformer",
    "cardiomyopathy",
    "single cell"
]

# If any of these appear, drop the item.
EXCLUDE_KEYWORDS = [
    "plant",
    "fungus",
    "yeast",
    "microbiome",
    "bacterial community",
    "ecology",
    "behavioral",
    "mouse model"  # remove if you do want mice
]

# ---- Canonical papers ----
# These seed the semantic similarity – think of them as "prototypes"
# for what you care about. Titles + short summaries is enough.

CANONICAL_PAPERS = [
    {
        "title": "Single-cell multi-omics and perturbation profiling of human cardiomyocytes",
        "summary": "Integration of scRNA-seq, chromatin accessibility, and proteomics "
                   "to understand cardiomyopathy genetics and perturbation responses."
    },
    {
        "title": "Foundation models for single-cell gene expression and perturbation prediction",
        "summary": "Large-scale pretraining on single-cell transcriptomes and use of "
                   "in silico perturbations to predict gene regulatory responses."
    },
    {
        "title": "Transfer learning enables predictions in network biology.",
        "summary": "Context-aware, attention-based model Geneformer pretrained on ~30M "
                   "single-cell transcriptomes to learn gene network dynamics."
    },
    {
        "title": "scGPT: towards building a foundation model for single-cell multi-omics",
        "summary": "using a generative pretrained transformer trained on over 30M cells."
    },
    {
        "title": "Tahoe-x1: scaling perturbation-trained single-cell foundation models. ",
        "summary": "Tx1 is pretrained on 200M+ perturbation-rich scRNA profiles and "
                   "fine-tuned for cancer-relevant prediction tasks."
    },
]


# Sentence-transformer model (small but decent)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# ==========================
# ======= DATA MODEL ========
# ==========================

@dataclass
class Paper:
    title: str
    summary: str
    link: str
    published: datetime
    source: str
    score: float = math.nan  # semantic similarity score (filled later)


# ==========================
# ====== HELPERS ===========
# ==========================

def strip_html(text: str) -> str:
    """Very small HTML stripper for summaries."""
    # We keep it minimal; if you want, swap for 'beautifulsoup4'.
    import re
    return re.sub(r"<[^>]+>", "", text or "")


def parse_datetime(entry: Dict[str, Any]) -> datetime:
    """Try to get a timezone-aware datetime for the entry."""
    if "published_parsed" in entry and entry["published_parsed"]:
        return datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)
    if "updated_parsed" in entry and entry["updated_parsed"]:
        return datetime(*entry["updated_parsed"][:6], tzinfo=timezone.utc)
    # Fallback: now
    return datetime.now(timezone.utc)


def passes_keyword_filters(paper: Paper) -> bool:
    text = f"{paper.title} {paper.summary}".lower()

    if INCLUDE_KEYWORDS:
        if not any(k.lower() in text for k in INCLUDE_KEYWORDS):
            return False

    if EXCLUDE_KEYWORDS:
        if any(k.lower() in text for k in EXCLUDE_KEYWORDS):
            return False

    return True


def fetch_feed(feed: Dict[str, str]) -> List[Paper]:
    print(f"Fetching feed: {feed['name']}  ({feed['url']})")
    parsed = feedparser.parse(feed["url"])
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)

    papers: List[Paper] = []

    for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
        published = parse_datetime(entry)
        if published < cutoff:
            continue

        title = entry.get("title", "").strip()
        summary = strip_html(entry.get("summary", "")).strip()
        link = entry.get("link", "").strip() or feed["url"]

        if not title:
            continue

        paper = Paper(
            title=title,
            summary=summary,
            link=link,
            published=published,
            source=feed["name"],
        )

        if passes_keyword_filters(paper):
            papers.append(paper)

    print(f"  -> kept {len(papers)} items after filters")
    return papers


def rank_papers(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Build canonical embedding
    canon_texts = [c["title"] + ". " + c.get("summary", "") for c in CANONICAL_PAPERS]
    canon_emb = model.encode(canon_texts, convert_to_tensor=True)
    canon_emb = canon_emb.mean(dim=0, keepdim=True)  # average canonical embedding

    # Encode paper texts
    texts = [p.title + ". " + p.summary for p in papers]
    paper_emb = model.encode(texts, convert_to_tensor=True)

    sims = util.cos_sim(paper_emb, canon_emb).cpu().numpy().reshape(-1)

    for p, s in zip(papers, sims):
        p.score = float(s)

    # Sort by score descending, then by recency
    papers_sorted = sorted(
        papers,
        key=lambda p: (p.score, p.published),
        reverse=True
    )
    return papers_sorted


def format_paper_md(p: Paper) -> str:
    date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
    score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
    summary = p.summary or "_No abstract/summary available._"
    summary = textwrap.shorten(summary, width=600, placeholder="…")

    return textwrap.dedent(f"""
    ### [{p.title}]({p.link})
    - Source: **{p.source}**  
      Date: {date_str} · Relevance score: {score_str}

    {summary}
    """)


def build_markdown_digest(papers: List[Paper]) -> str:
    now = datetime.now(timezone.utc)
    header = f"# Weekly Literature Digest\n\nGenerated on {now:%Y-%m-%d %H:%M UTC}\n"
    intro = textwrap.dedent(f"""
    Time window: last {LOOKBACK_DAYS} days  
    Feeds: {', '.join(f['name'] for f in FEEDS)}  

    Ranked by semantic similarity to your canonical papers and filtered by keywords.
    """)

    body_parts = []
    for i, p in enumerate(papers, start=1):
        body_parts.append(f"---\n\n**#{i}**\n")
        body_parts.append(format_paper_md(p))

    return header + "\n" + intro + "\n\n" + "".join(body_parts)
    

def save_markdown(md: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    fname = f"digest_{now:%Y-%m-%d}.md"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved digest to {path}")
    return path

def build_html_digest(papers: List[Paper]) -> str:
    """Return a full HTML document string for the digest."""
    now = datetime.now(timezone.utc)

    # simple inline CSS so the file is standalone
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        max-width: 900px;
        margin: 2rem auto;
        padding: 0 1rem 3rem;
        line-height: 1.5;
        color: #111827;
        background-color: #f9fafb;
    }
    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.25rem;
    }
    .meta {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .paper {
        background: #ffffff;
        margin: 1rem 0 1.5rem;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }
    .paper h2 {
        font-size: 1.1rem;
        margin: 0 0 0.25rem;
    }
    .paper a {
        color: #2563eb;
        text-decoration: none;
    }
    .paper a:hover {
        text-decoration: underline;
    }
    .paper .info {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .paper .summary {
        font-size: 0.95rem;
        white-space: pre-wrap;
    }
    .index {
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.25rem;
    }
    """

    header = f"""
    <h1>Weekly Literature Digest</h1>
    <div class="meta">
        Generated on {now:%Y-%m-%d %H:%M UTC}<br>
        Time window: last {LOOKBACK_DAYS} days<br>
        Feeds: {", ".join(f["name"] for f in FEEDS)}
    </div>
    """

    paper_blocks = []
    for i, p in enumerate(papers, start=1):
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
        summary = p.summary or "No abstract/summary available."
        # Optionally truncate extremely long summaries:
        summary = textwrap.shorten(summary, width=1200, placeholder="…")

        block = f"""
        <div class="paper">
          <div class="index">#{i}</div>
          <h2><a href="{p.link}" target="_blank" rel="noopener noreferrer">
              {p.title}
          </a></h2>
          <div class="info">
            Source: <strong>{p.source}</strong> ·
            Date: {date_str} ·
            Relevance score: {score_str}
          </div>
          <div class="summary">{summary}</div>
        </div>
        """
        paper_blocks.append(block)

    body = header + "\n".join(paper_blocks)

    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Weekly Literature Digest</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>{css}</style>
    </head>
    <body>
    {body}
    </body>
    </html>
    """

    return html


def save_html(html: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    fname = f"digest_{now:%Y-%m-%d}.html"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML digest to {path}")
    return path


def post_to_slack(papers: List[Paper]) -> None:
    if not SLACK_WEBHOOK_URL:
        print("No Slack webhook configured; skipping Slack notification.")
        return

    top = papers[:TOP_K_SLACK]
    if not top:
        print("No papers to post to Slack.")
        return

    lines = ["*Weekly literature digest – top hits*"]
    for p in top:
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        score_str = f"{p.score:.3f}"
        line = f"• <{p.link}|{p.title}>  _(source: {p.source}, {date_str}, score {score_str})_"
        lines.append(line)

    text = "\n".join(lines)

    resp = requests.post(
        SLACK_WEBHOOK_URL,
        json={"text": text},
        timeout=10,
    )
    if resp.status_code // 100 == 2:
        print("Posted top papers to Slack.")
    else:
        print(f"Slack post failed: {resp.status_code} {resp.text}")


# ==========================
# ========= MAIN ===========
# ==========================

def main():
    all_papers: List[Paper] = []
    for feed in FEEDS:
        try:
            all_papers.extend(fetch_feed(feed))
        except Exception as e:
            print(f"Error fetching {feed['name']}: {e}")

    if not all_papers:
        print("No papers found after filtering.")
        return

    ranked = rank_papers(all_papers)
    top = ranked[:TOP_K]

    html = build_html_digest(top)
    save_html(html)
    post_to_slack(top)  # Slack still uses the text summary we build there


if __name__ == "__main__":
    main()
