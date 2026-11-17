# Persian rewriting research

The current revision is **Retrieval-Guided Persian Rewriting: An Artifact Audit and a Two-Model Controlled Study**.

**Sole author:** Rizwanullah Muahed. **Instructor and report recipient:** Abdul Awal Quraishi; he is not a coauthor.

- [Current article PDF](current/Article.pdf) / [Word](current/Article.docx)
- [Complete Dari report to the instructor](current/Report_to_Supervisor_Dari.pdf)
- [Complete evidence package](current/Research_Package.zip)
- [Single-author citation](CITATION.bib) / [Current metadata](current/metadata.json)
- [Experiment code and raw outputs](revision/experiments/) / [Corpus analysis and split](revision/corpus_analysis/)
- [Detailed Dari delivery guide](current/README.md)

The current article contains 25 cited sources and seven figures. The completed cloud experiment has 800 main responses and a separate 80-response numeral diagnostic. The report distinguishes automatic diagnostics, model-assisted inspection, and uncollected human ratings.

## Offline evidence verification

From the repository root:

```bash
python -m pip install numpy==2.3.5
python research/verify_research.py
```

No API key or inference call is needed. The checker verifies saved response hashes, returned model snapshots, paired inputs, retrieval, recorded metrics, and current authorship.

## Historical supplied manuscript

[The original 21-page PDF](Persian_Text_Humanization_Paper.pdf) is retained byte-for-byte. Its printed byline included Abdul Awal Quraishi. The author has explicitly corrected that attribution: the current manuscript has one author, Rizwanullah Muahed, and the instructor is the report recipient. The old printed byline is historical evidence, not the current citation record. [Historical fingerprint and correction](manifest.json).

The application code remains a separate prototype; the research harness under revision/ documents the controlled API experiment. Original corpus and code provenance is pinned to commit feb0f282430b13277c43dd32ca93ceec13c837df in the article.
