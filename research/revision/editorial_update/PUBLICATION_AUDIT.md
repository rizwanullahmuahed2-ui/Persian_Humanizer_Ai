# Publication audit

Read-only audit of the pre-publication state on 2026-11-17. This report is the only file changed by the auditor. The authorized correction is: **Rizwanullah Muahed is the sole current author; Abdul Awal Quraishi is the teacher and recipient of the revision report.** Do not infer a contribution, approval, funding statement, or journal acceptance from that role.

## Current attribution and metadata requiring correction

| Source | Observed issue | Precise treatment |
|---|---|---|
| `/workspace/persian_humanization_revision/manuscript/Revised_Manuscript_EN.md` | Line 3 lists two authors. The declarations say names are retained from the supplied manuscript and “Both authors should confirm…”. Availability also refers to documentation “by the authors”. | Sole-author title block and declarations; teacher identified only in an appropriate acknowledgement/role statement if included. Keep journal-policy declarations unresolved where facts have not been supplied. |
| `/workspace/persian_humanization_revision/report/Revision_Report_Dari.md` | Opening `نویسندگان` line lists both names as authors. | Identify the current sole author and identify عبدالاول قریشی as استاد / دریافت‌کنندهٔ گزارش. Singularize current-author declarations where applicable. |
| `/workspace/persian_humanization_revision/build_documents.py` | `build()` sets article DOCX core author to both names; report core author is the generic “Article revision report”. | Set both documents' creator to Rizwanullah Muahed. Keep teacher/recipient separate from the creator property. This is the builder that would otherwise restore the incorrect article metadata. |
| `/workspace/Persian_Humanizer_Ai/research/README.md` | Front matter presents printed names as its author listing, links a two-author BibTeX file without historical labeling, and says citation metadata comes from the original first page. | Lead with the sole current author, link the revised manuscript/report, and put the unchanged original PDF in an explicitly historical subsection. Explain that its printed names are historical and do not establish current authorship. |
| `/workspace/Persian_Humanizer_Ai/research/CITATION.bib` | Key and author field cite Muahed and Quraishi together and point at the original PDF. | Provide a current single-author revision citation with the revised title and URL. If retaining an original-manuscript entry, give it a distinctly historical key and explicit historical note; do not offer it as the current recommended citation. Do not invent a publication date, venue, DOI, or acceptance. |
| `/workspace/Persian_Humanizer_Ai/research/manifest.json` | Generic `authors` contains both names and notes treat printed order as citation authority. | Make current author/teacher roles explicit; move original printed names to an `authors_as_printed` or equivalent historical field. Preserve original PDF filename, page count, fingerprint, and `pdf_modified: false`. |
| `/workspace/Persian_Humanizer_Ai/README.md` | “The unpublished manuscript … includes its authors, citation and version notes” routes readers only to the older manuscript. | Link the current revision and reproduction instructions; distinguish the unchanged historical PDF from the current single-author revision. |

No other active builder inspected (`finalize_report.py`, `editorial_update/finish_copyedit.py`, `editorial_update/integrate_sources_figures.py`) hardcodes the second author. They operate on the existing Markdown rather than restoring the opening author block. `editorial_update/refs.json` author arrays describe cited works and should not be changed for this correction.

Regenerate the following derived current documents from corrected sources, rather than patching binary files independently:

- `manuscript/Revised_Manuscript_EN.{html,docx,pdf}`.
- `report/Revision_Report_Dari.{html,docx,pdf}`.
- `delivery/Persian_Humanization_Revised_Article.{docx,pdf}` and `delivery/Persian_Humanization_Article_25Refs_7Figures.{docx,pdf}`.
- `delivery/Persian_Humanization_Revision_Report_Dari.{docx,pdf}` and `delivery/Persian_Humanization_Updated_Report_Dari.pdf`.
- The newly published package snapshot, ZIP, checksums, completion certificate, build receipt and verification receipt that describe those regenerated artifacts. Existing package trees contain copied old metadata and must not be presented as the corrected package unchanged.

The current article DOCX and both article aliases all contained the two-name `dc:creator` when audited. The report DOCX creator was generic. The generated current PDFs had titles but no `Author` property reported by `pdfinfo`; their visible title blocks still require regeneration.

## Historical and frozen material

Preserve the original PDF byte for byte:

`research/Persian_Text_Humanization_Paper.pdf` — 21 pages; SHA-256 `88e5a8f5ef5025d4452e297fd91f29a97382f1be1440d44efb9472c7106ae8c3` (computed during this audit).

Do not rewrite `original_extracted.txt`, extracted original figures, `editorial_update/pre_copyedit.md`, the existing research archives, frozen plans/specifications/seals, request scripts/harness sealed by the plans, raw provider responses/errors, corpus/splits, embeddings, or recorded research outputs to remove historical names or paths. `pre_copyedit.md` has the old two-name title block and two-author declaration; historical context belongs in a README/sidecar, not in a silent edit to that snapshot. `methods_audit.md` also describes the original PDF's two printed authors in its historical critique. Preserve that observation as historical.

The files and seals recorded by `editorial_update/EXPERIMENT_FILES_BEFORE_EDITORIAL_UPDATE.json` provide the research-preservation baseline. Compare against it after editorial/publication work. Original `/workspace/...` paths inside frozen plans must remain unchanged.

## Publication layout and offline verification

Recommended GitHub layout:

```text
research/
  README.md, CITATION.bib, manifest.json
  Persian_Text_Humanization_Paper.pdf       # historical, unchanged
  revision/
    README.md                              # current author, documents, offline command
    manuscript/, report/, editorial_update/
    experiments/, corpus_analysis/, code_audit/
    build_documents.py, build_delivery.py, finalize_report.py
    ...                                    # selected reviewable source/audit files
    verify_publication.py                  # wrapper, outside frozen research
    PUBLICATION_MANIFEST.json
  evidence/
    Persian_Humanizer_Ai/data/
      persian_humanizer_dataset.csv          # exact frozen source copy, optional
```

Do not recursively copy `delivery/Persian_Humanization_Package` or old ZIPs into `research/revision`; those duplicate the repository, old current documents and runtime data. A separately generated downloadable package can retain its existing sibling-folder layout, with its own manifest and verifier.

The existing `delivery/verify_package.py` assumes its own directory contains `PACKAGE_MANIFEST.json`, `persian_humanization_revision/`, and `Persian_Humanizer_Ai/`. Copying it unchanged into `research/revision` will not work. Both analysis programs already support `--artifact-root`, which replaces the original `/workspace` prefix without altering frozen plans. A publication wrapper should create a temporary mapping directory containing:

```text
<temp>/persian_humanization_revision -> <checkout>/research/revision
<temp>/Persian_Humanizer_Ai         -> <checkout>
```

Then run the main analyzer with explicit `--directory`, `--secondary-directory`, `--artifact-root <temp>`, and a temporary `--out`; run the numeric analyzer with the same root and explicit `--main-diagnostics <main-out>/response_diagnostics.csv`. Only Python and NumPy are needed. Do not load the generator, harness, credentials, or models. Expected totals: 800 main successful responses, 80 targeted numeric responses, 400 preserved unavailable-model errors, 100 main query groups and 10 independent numeric source groups.

If future source CSV edits must not break verification, point `<temp>/Persian_Humanizer_Ai` at the minimal frozen `research/evidence/Persian_Humanizer_Ai` directory instead. Inspection of all three plans shows that the only dependency outside the revision tree for response analysis is `data/persian_humanizer_dataset.csv`; a full application snapshot or Chroma database is unnecessary. Include `train_set.csv` and relevant audited source separately only if reproducing the broader repository/corpus audit. Current repository tests are distinct from reproducing the saved research.

## Secret and staging audit

No credential file was opened, including `/workspace/.env`. All `.env*` filenames were excluded from content scanning, including `.env.example`. The scan inspected 5,500 regular files outside `.git`, `.venv` and `__pycache__`, plus ZIP/DOCX members. Outputs contained only filenames, sizes, and boolean regex flags. Results for OpenAI-style tokens, GitHub tokens, AWS access-key IDs, private-key headers and literal bearer credentials: **all false**. No regular file reached 45 MiB; this also puts all inspected files below GitHub's 100 MiB hard limit. This is a pattern check, not a proof that arbitrary strings cannot contain credentials.

Existing delivery ZIPs each contain 12 runtime/cache files: the tracked Chroma DB/index files and `experiments/__pycache__/analyze_api_results.cpython-312.pyc`. They also include `.env.example`, a named template that was not read. No real `.env` file was found under the revision tree. The application repository itself has `.git`, which must not be copied.

`build_delivery.py` currently copies nearly every revision child and only explicitly forbids `.env` and `.env.local`; it would accept `.env.production`, a future `.git`, or an unexpected runtime directory. Its tracked-repository snapshot copies Chroma data despite the repository ignore rule because those files are already tracked. For the new publication package, use a selected-source allowlist, reject every `.env*` except an explicitly approved example template, reject `.git`/`.venv`/cache/model/Chroma runtime directories, and inspect ZIP members after building. These packaging safeguards do not require changing the original tracked runtime artifacts or frozen research.

Stage only named publication paths. Before committing, inspect `git diff --cached --name-only`, reject secret/runtime path names and files at or above the chosen size ceiling, and rerun redacted regex scans on staged bytes and archive members. Avoid `git add /workspace` or an unreviewed recursive copy of the revision parent. Recompute publication hashes only after regeneration and successful offline verification.
