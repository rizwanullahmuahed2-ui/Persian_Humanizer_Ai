# Persian text humanization manuscript

[Read the paper (PDF)](Persian_Text_Humanization_Paper.pdf) · [BibTeX citation](CITATION.bib) · [File manifest](manifest.json)

**Design and Development of a Retrieval-Augmented Generation-Based Persian Text Humanization Chatbot Using Large Language Models**

**Authors, in printed order:** Rizwanullah Muahed; Abdul Awal Quraishi.

**Affiliation:** Department of Computer Science, Kabul Polytechnic University, Kabul, Afghanistan.

**Status:** Unpublished manuscript supplied by the project owner. University approval was previously reported by the owner; this upload does not include a separate approval certificate. No journal, publication date or DOI is assigned here.

The 21-page PDF is preserved exactly as supplied. Its original upload filename was `Persian Text Humanization Paper(1).pdf`; only the repository filename was simplified. The manifest records its SHA-256 fingerprint. Citation metadata comes from the first page, not the generic author field in the PDF properties.

## What the paper describes

The paper studies natural Persian/Dari rewriting that preserves meaning, facts and the input's register. It describes retrieval-guided prompting with a locally hosted Llama 3.1 8B model, multilingual embeddings, ChromaDB, LangGraph and FastAPI, without model fine-tuning.

The manuscript identifies an LLM-only versus LLM+RAG comparison, retrieval-depth comparisons and held-out evaluation excluding self-retrieval as work still needed. The paper's reported metrics should be read with those limits.

## Relationship to the current code

| Component | Supplied manuscript | Current repository |
|---|---|---|
| Corpus | 3,173 paired examples | 3,511 paired examples |
| Retrieval | One embedding-similarity search; no reranker | Up to 20 candidates, optional cross-encoder reranking, up to four retrieved examples |
| Style selection | Register is a property of the input, not a conversion target | One natural-rewriting workflow; no user-selectable styles or production style filter |
| Generation | Local Llama 3.1 8B through Ollama | Ollama by default, with an optional OpenAI provider |

The manuscript documents a related earlier system version. Current automated tests verify software behavior with controlled generation; they do not reproduce or independently confirm the paper's reported experiments. See the [project README](../README.md) for current setup and behavior.
