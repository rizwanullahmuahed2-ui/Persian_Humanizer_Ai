# Reproduce the saved-response analysis

This analysis makes no API or model call and reads no credential file. It imports no generator or harness implementation. Python and NumPy are required. `ANALYSIS_SPEC.json` was sealed before response inspection. `SPEC_AMENDMENT.json` records the later availability-based replacement of the unavailable nano generation endpoint; the original 400 HTTP 404 error records remain in the first archive.

From the `persian_humanization_revision` directory in the original workspace:

```sh
python experiments/analyze_api_results.py --directory experiments/api_study_v1 --secondary-directory experiments/api_study_secondary --out verification_analysis
```

For an extracted package, use its top-level folder as `--artifact-root`. That folder must contain both `persian_humanization_revision` and `Persian_Humanizer_Ai`. For example, from its `persian_humanization_revision` folder:

```sh
python experiments/analyze_api_results.py --directory experiments/api_study_v1 --secondary-directory experiments/api_study_secondary --artifact-root .. --out verification_analysis
```

The analyzer maps the original `/workspace/` prefix to that supplied package root for reading source and embedding artifacts. It does not modify any frozen plan, prompt, hash, raw response, or original path recorded in the plan. When no relocation is supplied, the original paths are used.

The scripts verify original and secondary plan seals, source and embedding file hashes, source-only embedding linkage, deterministic retrieval and random examples, identical queries and prompts between models, raw output hashes, returned snapshots, normal termination, response IDs, and usage. Error and missing records are reported explicitly. Final summaries require all 800 successful study responses plus the 400 documented unavailable-model records; no missing output is synthesized or silently dropped.

`response_diagnostics.csv` compares each output with its source and with the **unvalidated associated rewrite**, separately. These lexical and numerical diagnostics do not constitute human naturalness ratings or a semantic-accuracy estimate. Paired bootstrap intervals use 100 independent query groups; 800 generated responses are not 800 independent source questions.
