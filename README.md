# lit_feed

Customize recent literature of interest feed. Lit_feed will pull most recent papers from the journals you selected, filter by keywords, rank them by semantic relevance to the seed papers you set, and feed the results in an HTML file.

Everything lives in one script, `lit_feed.py`. There is no package to install, no
database, no API key. Each run writes a digest into `digests/` — one copy to keep,
one shaped for email — and, if you set a webhook, posts the top hits to Slack.

## What it does

```
RSS / Atom feeds  ·  Crossref API        (the FEEDS list)
        │
        ▼
keyword filter          include-any, then exclude-any, on title + abstract
        │
        ▼
semantic ranking        cosine similarity to the mean embedding of your
        │               canonical seed papers (local sentence-transformers)
        ▼
digests/digest_YYYY-MM-DD.html          archive + the script's memory
digests/digest_YYYY-MM-DD.email.html    the copy to mail
        │
        └──▶ optional Slack post of the top 15
```

The script reads only metadata the source already hands it — title, authors, link,
date, abstract — whether that arrives as an RSS entry or a Crossref record. It never
fetches a publisher page or a PDF, so paywalled journals cost you nothing beyond a
shorter abstract.

## Installation

Requires **Python 3.10 or newer** (the script uses `str | None` annotations).

```bash
git clone https://github.com/zyj1729/lit_feed.git
cd lit_feed

python3 -m venv .venv
source .venv/bin/activate

pip install feedparser requests numpy sentence-transformers torch
python lit_feed.py
```

Those are exactly the packages `lit_feed.py` imports; everything else it uses is in
the standard library. `torch` arrives with `sentence-transformers` and is the large
download — the rest is small.

The virtualenv is not ceremony: on Homebrew Python and recent Debian/Ubuntu, a bare
`pip install` refuses to run at all with `error: externally-managed-environment`.

The first run downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
and caches it. The model runs locally — on GPU or Apple-silicon MPS if torch finds
one, otherwise CPU — so there is nothing to pay for and no key to configure.

## Daily use

```bash
python lit_feed.py
```

That is the whole interface. The script takes no arguments and no subcommands, so
anything you type after the filename is ignored. Run it from the repo directory, or
from wherever you want `digests/` to live.

It runs fine on a headless server — none of its dependencies need a display or a
browser.

## Output

Two files per day, both overwritten if you run twice in a day:

```
digests/
  digest_2026-08-18.html         archive, and the script's memory
  digest_2026-08-18.email.html   what you send to people
```

Both hold the same papers. They differ in what they are for.

**The archive copy** ends its body with a machine-readable comment
(`<!-- DIGEST_KEYS_JSON ... -->`) listing every paper the run knew about. That
comment is the script's entire memory, and most of the file's size. The next run
reads the most recent archive that is not today's, drops anything older than
`LOOKBACK_DAYS`, and uses the rest to decide what counts as new. **Keep
`digests/`** — delete it and the next digest treats everything as unseen.

**The email copy** carries no such comment, is laid out in tables at a fixed 600px
with every style inline, and truncates abstracts to `EMAIL_ABSTRACT_CHARS`. Mail
clients are the reason for each of those: Gmail clips a message over roughly 102 KB,
Outlook renders with Word's engine and has no flexbox, and several clients drop
`<style>` blocks entirely — a class-only stylesheet degrades to unstyled black text.
Send this file, not the archive.

Each digest has two sections:

- **Today's Feed** — papers published today or yesterday (UTC) that did not appear
  in the previous digest, scoring at least `TODAY_MIN_SCORE`, capped at
  `TODAY_TOP_K`. In the email these are full cards.
- **Previous Feed** — papers that were already in the previous digest, capped at
  `PREV_TOP_K`. In the email these sit inside a `<details>` element as a compact
  one-line list. Note `<details>` only actually collapses in Apple Mail and
  Thunderbird; Gmail, Outlook and Yahoo render it already open, which is why the
  content is a short list rather than full cards.

Yesterday is included in Today's Feed on purpose: feeds publish at different hours
and a strict same-day cut drops papers that land just after midnight UTC.

Papers are identified by a canonicalized link — scheme and host lowercased, trailing
slash and URL fragment dropped. An entry with no link of its own falls back to its
feed's `url`, so several link-less entries from one feed collapse into a single
remembered paper. Ranking happens over the union of today's items and the papers
recovered from the last digest, so scores stay comparable across the two sections.

The script also contains a Markdown digest builder, but nothing calls it.

## Topic tags

Each paper carries up to `TAG_MAX` coloured chips, matched from the `TAGS` table by
case-insensitive substring on title plus abstract. Tags are presentational only —
`INCLUDE_KEYWORDS` still decides what gets into the digest, and a paper matching no
tag simply shows none.

Tags are derived at render time rather than stored, so editing `TAGS` re-labels the
whole back catalogue on the next run instead of leaving old digests on an old
vocabulary.

Colours are Okabe-Ito for colour-vision safety, three of them darkened from the
published hexes so white 10px chip text clears WCAG AA at 4.5:1. Chip text colour is
picked per background by luminance, so a light tag gets dark text automatically. Keep
that in mind if you add one.

## Authors

Feeds disagree about author metadata more than you would expect, so the script
normalizes per source:

| Shape | Feeds | Handling |
|---|---|---|
| A real list on the entry | Nature family, PLOS | used directly |
| One string, `Given Family, Given Family` | arXiv, Cell | split on commas |
| One string, `Surname, I. I., Surname, I. I.` | bioRxiv | re-paired, flipped to `I. I. Surname` |
| Structured `given` / `family` | Crossref sources | joined |
| Nothing at all | Science, Oxford Bioinformatics | no author line shown |

Long lists render as `First Author, …, Last Author`. Note `entry.author` is the
*last* author on some feeds and the *entire list* on others, which is why the script
never reads it uniformly.

Affiliations are deliberately not collected. They are absent from every feed, and
absent from Crossref and OpenAlex for arXiv and preprints.org specifically — between
them a large share of a typical digest — so the field would be blank more often than
filled.

## Slack (optional)

```bash
export LIT_DIGEST_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
python lit_feed.py
```

It posts the top `TOP_K_SLACK` (15) items as one message — the new items if there are
any, otherwise the top of the full ranked list. With the variable unset the script
prints `No Slack webhook configured; skipping Slack notification.` and carries on.

## Configuration

Everything is a constant at the top of `lit_feed.py`, in the `CONFIG` section. Edit
the file directly.

| Constant | Default | What it does |
|---|---|---|
| `LOOKBACK_DAYS` | `30` | Ignore entries older than this, and prune remembered papers past it |
| `MAX_ITEMS_PER_FEED` | `200` | Entries read per feed before filtering. A feed entry can raise its own budget with `max_items` (preprints.org uses `3000`) |
| `TODAY_TOP_K` | `40` | Max papers in Today's Feed |
| `PREV_TOP_K` | `30` | Max papers in Previous Feed |
| `TODAY_MIN_SCORE` | `0.30` | Similarity floor for Today's Feed |
| `TOP_K_SLACK` | `15` | Papers per Slack message |
| `EMAIL_ABSTRACT_CHARS` | `300` | Abstract length in the email copy; what keeps cards near-uniform height |
| `TAG_MAX` | `3` | Chips shown per paper |
| `OUTPUT_DIR` | `"./digests"` | Where digests are written and read back from |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Ranking model; any sentence-transformers model works |
| `CROSSREF_ROWS` | `500` | Records per Crossref page |
| `CROSSREF_PAUSE_SEC` | `1.0` | Pause between Crossref pages |
| `CROSSREF_RETRIES` | `4` | Retries before giving up on a Crossref page |

Two settings come from the environment:

| Variable | Effect |
|---|---|
| `LIT_DIGEST_SLACK_WEBHOOK` | Slack incoming-webhook URL. Unset means no Slack post. |
| `LIT_CROSSREF_EMAIL` | Optional. Your email, added to the Crossref User-Agent to use their faster "polite pool" rate limit. |

`TOP_K = 60` is still defined but currently has no effect — the line that applied it
is commented out. `TODAY_TOP_K` and `PREV_TOP_K` are the caps that actually run.

## Journals default

_arXiv q-bio_, _arXiv cs.LG_, _bioRxiv Genomics+Bioinformatics_, _preprints.org_,
_Nature_, _Cell_, _Science_, _Nature Methods_, _Nature Genetics_,
_Nature Biotechnology_, _Genome Research_, _Oxford Bioinformatics_,
_PLOS Computational Biology_

Thirteen sources. The Oxford entry is named `Bioinformatics` in the code.

Twelve are RSS or Atom. **preprints.org is read through the Crossref REST API**
instead, by DOI prefix `10.20944`: its own RSS endpoints answer `403` to scripted
clients, while Crossref carries the same records — title, abstract, authors, posting
date — with no bot wall. Crossref also filters by date server-side, so that source
sees the whole `LOOKBACK_DAYS` window rather than whatever the latest N entries
happen to be. If Crossref rate-limits mid-run, the script keeps the pages it already
has and says so rather than losing the source.

Two publishers serve XML that `feedparser`'s own fetcher rejects, returning zero
entries — Genome Research sends a duplicate attribute, Nature Biotechnology an
invalid token. When a feed comes back empty the script refetches with `requests` and
a browser User-Agent, which both of those then parse, printing
`(recovered N entries via browser User-Agent)` when that path is used.

## Customization

Four blocks in `lit_feed.py`, each marked with a comment you can search for:

- **Feeds** (`# ---- Feeds ----`) — the `FEEDS` list. The `name` is the source label
  in the digest. Two entry shapes are supported:

  ```python
  # RSS or Atom
  {"name": "Nature Methods", "url": "https://www.nature.com/nmeth.rss"}

  # A preprint server or publisher indexed by Crossref, by DOI prefix
  {"name": "preprints.org", "url": "https://www.preprints.org/",
   "type": "crossref", "prefix": "10.20944", "max_items": 3000}
  ```

  For a Crossref entry the `url` is never fetched — it is only the display link and
  the fallback identity for a record with no URL of its own. To add another Crossref
  source, copy that entry and change `name` and `prefix`.

- **Keyword filters** (`# ---- Keyword filters ----`) — `INCLUDE_KEYWORDS` and
  `EXCLUDE_KEYWORDS`. Matching is case-insensitive substring against title plus
  abstract. A paper must hit at least one include keyword, and any single exclude
  keyword drops it. Leaving `INCLUDE_KEYWORDS` empty skips the include test only —
  the exclude list still applies.

- **Topic tags** (`# ---- Topic tags ----`) — the `TAGS` list, one
  `(label, colour, [keywords])` per chip. Kept separate from the filters on purpose,
  so you can admit a paper without also labelling it.

- **Canonical papers** (`# ---- Canonical papers ----`) — `CANONICAL_PAPERS`, the
  seed papers that define relevance. Title plus a couple of sentences is enough; they
  are embedded and averaged into a single reference vector. This is the setting that
  most changes the ranking.

## Limitations

Worth knowing before you trust the ranking:

- **Abstracts only.** The score comes from the title and the abstract. Abstracts
  systematically oversell the numbers, the honest limitations, and whether a result
  survives out of distribution. Use the digest to decide what to open, not what to
  believe.
- **Feed quality varies.** In practice some journals put a full abstract in RSS while
  others give a line or two, which leaves the ranker very little to work with. Worth
  checking your own feeds rather than assuming.
- **The broad sources are noisy.** arXiv cs.LG and preprints.org are general-purpose,
  so plenty of off-target work slips past `INCLUDE_KEYWORDS`. Narrow the include list
  or extend `EXCLUDE_KEYWORDS` if it crowds out biology.
- **Tags are substring matches, not judgements.** A paper that mentions a keyword in
  passing gets the chip.
- **`TOP_K_SLACK` is a hard cut.** The Slack message shows 15 items. Anything below
  that is in the HTML digest, not missing.
