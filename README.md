# lit_feed
Customize recent literature of interest feed. Lit_feed will pull most recent papers from the journals you selected, filter by keywords, rank them by semantic relevance to the seed papers you set, and feed the results in an HTML file. 


## Installation

```bash
git clone https://github.com/zyj1729/lit_feed.git
pip install feedparser requests sentence-transformers torch

cd lit_feed
python lit_feed.py
```

## Journals default
_arXiv q-bio_

_arXiv cs.LG_

_bioRxiv Genomics+Bioinformatics_

_Nature_

_Cell_

_Science_

_Nature Methods_

_Nature Genetics_

_Nature Biotechnology_

_Genome Research_

_Oxford Bioinformatics_

_PLOS Computational Biology_

## Customization

You can customize the journals to include in **Feeds** section, keywords to include or exclude in **Keywords filters** section, seed paper of relevance in **Canonical papers** section. 

