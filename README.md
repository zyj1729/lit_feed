# lit_feed

Customize recent literature of interest feed. Lit_feed will pull most recent papers from the journals you selected, filter by keywords, rank them by semantic relevance to the seed papers you set, and feed the results in an HTML file.

It is one script with no database and no API key. Each run writes an HTML digest
into `digests/`: papers new since the last run at the top, ranked by how close they
are to the seed papers you define, each tagged with coloured topic labels.

## Install

Needs Python 3.10 or newer.

```bash
git clone https://github.com/zyj1729/lit_feed.git
cd lit_feed

python3 -m venv .venv
source .venv/bin/activate
pip install feedparser requests numpy sentence-transformers torch
```

The virtualenv is not optional on Homebrew Python or recent Debian/Ubuntu, where a
bare `pip install` refuses to run at all.

## Run

```bash
python lit_feed.py
```

That is the whole interface — no arguments, no subcommands. The first run downloads
a ~90 MB embedding model and caches it; everything after that is local. Works on a
headless server.

## Output

Two files per day, both rewritten if you run twice:

```
digests/
  digest_2026-08-19.html         open this one in a browser
  digest_2026-08-19.email.html   send this one if you mail the digest
```

Same papers, different packaging. The email copy is built for mail clients: 600px
tables, inline styles, shorter abstracts. The browser copy also carries a hidden
comment listing every paper it has seen, which is how the next run knows what is
new — so **keep the `digests/` folder.** Delete it and the next digest treats
everything as unseen.

Each digest has *Today's Feed* (papers you have not been shown before) and
*Previous Feed* (everything from the last 30 days you have already seen).

## Make it yours

Everything you would want to change is a list near the top of `lit_feed.py`, each
marked with a comment you can search for.

**`# ---- Canonical papers ----`** is the one that matters most. `CANONICAL_PAPERS`
holds a few papers that represent what you care about — a title plus a sentence or
two each. They are averaged into one reference point, and every incoming paper is
scored by similarity to it. Replace these with papers from your own field and the
ranking follows.

**`# ---- Keyword filters ----`** decides what is even considered.
`INCLUDE_KEYWORDS` — a paper must match at least one. `EXCLUDE_KEYWORDS` — any
single match drops it. Case-insensitive substrings against title and abstract.

**`# ---- Feeds ----`** is the source list. Add any RSS or Atom feed:

```python
{"name": "Nature Methods", "url": "https://www.nature.com/nmeth.rss"}
```

**`# ---- Topic tags ----`** sets the coloured labels. Each entry is a label, a
colour, and the keywords that trigger it:

```python
("spatial", "#A36186", ["spatial transcriptomics"]),
```

## Settings

Constants at the top of `lit_feed.py`:

| Constant | Default | What it does |
|---|---|---|
| `LOOKBACK_DAYS` | `30` | How far back to look, and how long a paper stays in Previous Feed |
| `TODAY_TOP_K` | `40` | Max papers in Today's Feed |
| `PREV_TOP_K` | `30` | Max papers in Previous Feed |
| `TODAY_MIN_SCORE` | `0.30` | Similarity floor — raise it if the digest feels noisy |
| `EMAIL_ABSTRACT_CHARS` | `300` | Abstract length in the email copy |
| `TAG_MAX` | `3` | Labels shown per paper |
| `OUTPUT_DIR` | `"./digests"` | Where digests go |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Any sentence-transformers model works |

Optional environment variables: `LIT_DIGEST_SLACK_WEBHOOK` to post the top papers to
Slack, `LIT_CROSSREF_EMAIL` to identify yourself to Crossref for a faster rate limit.

## Default sources

_arXiv q-bio_, _arXiv cs.LG_, _bioRxiv Genomics+Bioinformatics_, _preprints.org_,
_Nature_, _Cell_, _Science_, _Nature Methods_, _Nature Genetics_,
_Nature Biotechnology_, _Genome Research_, _Oxford Bioinformatics_,
_PLOS Computational Biology_

Twelve are RSS. preprints.org is read through the Crossref API instead, because its
own feeds block scripted clients. Some publishers serve XML the standard parser
chokes on, so the script retries those with a browser User-Agent automatically.
