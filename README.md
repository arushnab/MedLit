# Biomedical Literature Review — Multi-Agent Pipeline

**Sanofi R&D AI Engineering Internship Case Study**

---

## Overview

A lightweight, two-agent pipeline that retrieves relevant PubMed Central articles for a biomedical query and generates concise abstractive summaries with keywords. Built to run entirely on CPU with no API keys required.

```
User Query
    │
    ▼
Retriever Agent  ──►  Pinecone Vector DB  ──►  Top-K Articles
                                                      │
                                                      ▼
                                          Summarizer Agent
                                                      │
                                                      ▼
                                          Structured Report
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install boto3 pandas sentence-transformers pinecone \
            transformers keybert torch sentencepiece pydantic

# 2. Set your Pinecone API key
export PINECONE_API_KEY="your-key-here"

# 3. Open and run the notebook
jupyter notebook notebook.ipynb
```

---

## Notebook Structure

| Section                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| **1. Setup & Data Access** | Connect to PMC S3, download metadata CSV, fetch and parse XML articles |
| **2. Retriever Agent**     | Generate embeddings, index to Pinecone, query by semantic similarity   |
| **3. Summarizer Agent**    | FLAN-T5-base abstractive summarization + KeyBERT keyword extraction    |
| **4. Report Generation**   | Run all 3 case study queries end-to-end, save structured report        |

---

## Design Choices & Tradeoffs

### Retriever Agent

**Embedding model: `all-MiniLM-L6-v2`**
Chosen for its balance of speed and semantic quality. Produces 384-dimensional dense vectors and runs efficiently on CPU. Encodes both title and abstract together (`"{title}. {abstract}"`) to give the model richer context per article than abstract alone.

**Vector database: Pinecone**
Managed vector database with millisecond retrieval via approximate nearest neighbour search. Scales to millions of vectors with no latency increase — a key advantage over brute-force cosine similarity. Cosine similarity is used as the distance metric since all vectors are L2-normalized, making dot product equivalent and fast.

**Why not BM25 or TF-IDF?**
Keyword-based methods like BM25 and TF-IDF rely on exact term overlap and word frequency. They have no understanding of meaning — 'adverse events' would never match 'side effects' — which makes them poorly suited for biomedical literature where terminology varies widely between authors and papers.

---

### Summarizer Agent

**Model: `google/flan-t5-base`**
FLAN-T5-base is an instruction-tuned T5 variant trained to follow natural language instructions. This makes it well-suited for abstractive summarization — it rephrases rather than copies text — unlike BART which was trained on news compression and tends to extract sentences verbatim.

**Model selection process:**
Three models were evaluated empirically on the same abstract:

| Model                  | Output Quality | Issue                                             |
| ---------------------- | -------------- | ------------------------------------------------- |
| `facebook/bart-base`   | Poor           | Copied sentences directly, truncated mid-sentence |
| `google/flan-t5-small` | Poor           | Too small to follow instructions reliably         |
| `google/flan-t5-large` | Poor           | Hallucinated fake paper titles and URLs           |
| `google/flan-t5-base`  | Acceptable     | Abstractive, occasionally misspells medical terms |

`flan-t5-base` was selected as the best CPU-friendly option. In production, this would be replaced by a larger model or API-based LLM (e.g. Claude, GPT-4) for higher quality output.

**Keyword extraction: KeyBERT**
BERT-based keyphrase extraction using cosine similarity between candidate n-grams and the full abstract embedding. Extracts 1–2 word biomedical terms without requiring any labelled training data.

**Post-processing:**
A `clean_summary()` function trims output at the last complete sentence to avoid mid-sentence truncation artifacts from the model hitting its token limit.

---

### Data Pipeline

**Source: PMC Open Access (oa_comm) on S3**
4.8 million open-access articles available publicly via unsigned S3 requests (no AWS credentials required). Articles are stored as XML files at `oa_comm/xml/all/{accession_id}.xml`. A metadata CSV at `oa_comm/txt/metadata/csv/oa_comm.filelist.csv` provides an index of all available articles including AccessionID and PMID.

**Article selection**
Articles are pre-filtered by keyword against the citation field to scope the corpus to biomedical topics relevant to Sanofi (vaccines, oncology, computational biology). 300 candidates are sampled with `random_state=42` for reproducibility, and the loop stops once 100 articles with abstracts longer than 100 characters are collected. The keyword filter introduces selection bias — articles using different terminology may be missed — but is a practical tradeoff for a prototype corpus.

**Caching**
Fetched articles are saved to `articles.json` to avoid re-downloading from S3 on every run. This file is ephemeral in Colab and will not survive session restarts.

---

## Creative Extensions

**Confidence Warning**
Each query's top retrieval score is compared against a threshold of 0.4. If the best match scores below this, a warning is printed and included in the report, signalling that the query may be outside the current corpus. This threshold was set empirically and would be calibrated against labelled examples in production.

**Categorizer Agent**
Each retrieved article is classified into a study type (Clinical Trial, Review Paper, Case Study, Computational Study, Observational Study) using rule-based keyword matching against the title and abstract. Standard biomedical methodology terminology is consistent enough across authors for keyword matching to work reliably here.

---

## Limitations

- **Corpus size**: Only 100 articles are indexed. Retrieval quality would improve significantly with a larger and more diverse corpus.
- **Keyword pre-filtering**: The topic filter biases the corpus and may miss relevant articles using non-standard terminology.
- **Summarisation quality**: FLAN-T5-base occasionally misspells medical terms, produces incomplete sentences, or fails to capture the most important finding. A production system would use a larger model.
- **No GPU**: The pipeline runs on CPU, making summarisation slow (~3–5 seconds per article). Parallelisation or GPU acceleration would be required at scale.
- **Single-category classification**: The categoriser returns the first matching category only. A paper matching multiple types (e.g. a computational clinical trial) will be miscategorised.

---

## Production Improvements

- Replace keyword pre-filtering with full-corpus indexing and let semantic search determine relevance
- Use a domain-specific embedding model (e.g. `PubMedBERT`) trained on biomedical literature
- Replace FLAN-T5-base with a production LLM via API (e.g. Claude, GPT-4) for higher quality summaries
- Parallelise summarisation across retrieved articles to reduce latency

---

## Files

| File                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `Sanofi_Case_Study.ipynb` | Main notebook — run top to bottom              |
| `articles.json`           | Cached article corpus (generated on first run) |
| `report.txt`              | Human-readable output report                   |
| `report.json`             | Structured results for downstream use          |
