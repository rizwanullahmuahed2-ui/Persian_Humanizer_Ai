# Retrieval-Guided Persian Rewriting: An Artifact Audit and a Two-Model Controlled Study

**Rizwanullah Muahed; Abdul Awal Quraishi**  
Department of Computer Science, Kabul Polytechnic University, Kabul, Afghanistan  
Corresponding author: rizwanullahmuahed2@gmail.com


## Abstract

Example retrieval can guide Persian rewriting without model adaptation, but its benefit must be separated from the effect of providing examples. We audit a rewriting prototype and conduct a paired four-condition study with two fixed model snapshots. The repository contains 3511 source–rewrite pairs, inconsistent training/index files, and a prompt formatter whose effective example count differs from the requested retrieval depth. Unicode-aware processing also exposes the sensitivity of corpus-overlap scores to Persian tokenization. After constructing text-disjoint training/development/test partitions of 2438/371/702 records, we select 100 test inputs from distinct groups. GPT-4.1 mini and GPT-4o mini each generate rewrites with zero examples, one or four semantic examples, and four random examples, yielding 800 verified responses. A separate numerical-coverage module adds 80 responses for ten test sources. Four semantic examples increase output–associated-reference ROUGE-L F1 over zero examples by 0.0892 and 0.0915. Their advantages over random examples are 0.0049 and 0.0092, with paired 95% bootstrap intervals spanning zero. References include register shifts and unverified preservation, so these values measure lexical agreement rather than naturalness or factual accuracy. Model-assisted case inspection identifies event changes, register drift, and qualifier omission. The study demonstrates reference convergence under example prompting while leaving semantic selection's advantage over random examples unestablished on that outcome. Versioned artifacts, raw responses, and independent verification support reproduction.

**Keywords:** Persian rewriting; Dari; retrieval-augmented generation; example retrieval; reproducibility; evaluation; Unicode tokenization.

## 1. Introduction

An automatic rewrite can be grammatically fluent while changing a deadline, weakening an obligation, introducing an unsupported statistic, or shifting a formal message into casual speech. A useful Persian rewriting system must therefore improve expression subject to preservation constraints. We use *naturalization* for this objective: making wording more natural while retaining the input's propositional content, communicative force, register, and language variety. The term does not imply that a system's output has been established to be human-authored.

Example retrieval is a plausible mechanism. Given an input, an embedding model retrieves similar source texts and their associated rewrites; the generator receives those pairs as demonstrations. Unlike knowledge-grounded question answering, the retrieved material mainly demonstrates a transformation rather than supplying facts to include in the answer. This distinction affects both evaluation and system design: a retrieved rewrite can introduce an inappropriate register or factual addition even when its source is semantically similar to the query.

The prototype examined here combines multilingual sentence embeddings, ChromaDB, LangGraph, FastAPI, and a Llama 3.1 8B model served through Ollama. The earlier manuscript described a working application and selected examples but did not isolate the effect of retrieval. It also interpreted low lexical overlap as successful humanization. Such an inference is not justified: low overlap can result from acceptable paraphrasing, unsupported changes, unrelated output, or script-incompatible tokenization. Conversely, a small edit may be sufficient for an already clear sentence.

This revision first establishes what the available artifacts support. Rather than carrying forward historical figures as confirmed experimental results, it audits the corpus, index, prompt path, and metric definitions. It then tests retrieval's contribution in a controlled API study using the same inputs and preservation instruction across conditions within each model. Three research questions organize the work:

- **RQ1—Artifact consistency:** Do the available corpus, training file, stored index, and prompt construction describe a consistent and reproducible experiment?
- **RQ2—Measurement validity:** How sensitive are the reported lexical descriptions to Persian text processing, and what conclusions can those descriptions support?
- **RQ3—Observable generation effects:** Under a fixed generator and preservation instruction, how do semantic example selection and retrieval depth affect copying, source-wording retention, associated-reference agreement, and output length relative to zero examples and random examples?

RQ1 and RQ2 are addressed by the artifact and corpus audits; RQ3 is addressed by the completed paired generation study. The broader application goal remains meaning-preserving naturalness, but the measured generation outcomes are deliberately narrower. The contributions are an artifact-level empirical audit, an explicit Persian text-processing analysis, and a controlled test of semantic selection against both an instruction-only baseline and a random-example control. Tool integration is an engineering contribution rather than evidence of improved language quality.

## 2. Related work and scope

Transformer architectures underpin the encoding and generation components [1], while few-shot language modeling shows how demonstrations can guide a task at inference time without parameter updates [2]. For rewriting, the question is whether the chosen demonstrations offer more useful guidance than examples selected without semantic matching.

Lewis et al. introduced retrieval-augmented generation for knowledge-intensive tasks using a training and fine-tuning recipe [3]. A frozen-generator application with retrieved rewrite examples is a different implementation choice; it should not be presented as evidence that the original RAG method required no fine-tuning. Li et al.'s Delete–Retrieve–Generate framework also places retrieval within text transformation rather than exclusively in factual question answering [4].

Instruction-based rewriting without parameter adaptation predates this application. Reif et al. demonstrate arbitrary style transfer with large language models using prompting [5]. Consequently, avoiding fine-tuning alone is not a sufficient novelty claim. The relevant question is whether retrieving demonstrations provides a measurable benefit for a clearly specified Persian rewriting objective.

Example selection has been studied directly in in-context learning. Liu et al. compare semantically retrieved examples with random selection for GPT-3 [6], and Rubin et al. learn a prompt retriever using language-model feedback [7]. Min et al. show that demonstration format and input distribution can explain important effects in classification and multiple-choice tasks [8]. These findings motivate controls in the present study; they do not establish that semantic selection will help Persian rewriting or that inaccurate rewrite targets are harmless.

The representation and retrieval choices also require a distinction. Sentence-BERT encodes sentences separately for efficient similarity comparison [9], and multilingual knowledge distillation extends this approach across languages [10]. Dense passage retrieval, by contrast, learns query–passage representations for question answering [11]. Here the retrieved pairs illustrate transformations: they are not sources of additional facts to insert into a rewrite.

Persian formality transfer and normalization provide useful related resources, but their objectives differ from register-preserving naturalization. Falakaflaki and Shamsfard study informal-to-formal transformation and discuss weaknesses of overlap-only evaluation [12]. Their critique does not justify reversing ROUGE into a metric to minimize. Tajalli et al. provide an informal–formal corpus, with a later published description in 2025 [13]. Rasooli et al. address standardization of colloquial Persian [14]. A corpus designed to change register is not automatically a gold standard for preserving register.

Evaluation literature separates naturalness, content preservation, and the intended style behavior [15,16]. For the present objective, the style behavior is preservation, unless a separate conversion task is explicitly requested. ROUGE [17], chrF [18], and contextual similarity measures such as BERTScore [19] supply complementary diagnostic information; none certifies factual equivalence or acceptable register by itself. Human evaluation requires documented instructions, sampling, and actual ratings rather than an inferred score from output length [16].

The Llama 3 family makes local deployment possible [20], but Persian and Dari are not among the eight languages listed as officially supported in the Llama 3.1 model card [21]. Their quality in this application must be measured. A user interface accepting Dari input does not establish that the demonstration corpus contains representative, independently validated Dari material.

## 3. Task definition and evidence design

### 3.1 Preservation before stylistic change

Let an input be $x$, a retrieval corpus be $D$, the ordered demonstration set be $R_k(x;D)$, and the generator be $G_\theta$. A generated rewrite is

$$ y_k = G_\theta(P(x,R_k(x;D));\lambda). $$

Here $P$ is the complete prompt and $\lambda$ contains decoding and runtime settings. The target is increased naturalness subject to preservation of named entities, quantities and units, dates and times, negation, conditions, obligations, certainty, register, and dialect. These are distinct evaluation dimensions rather than a single weighted heuristic score.

Dataset documentation should explain how examples were collected, who produced them, and what uses they support, as emphasized by the datasheet approach [22]. The corpus column names `ai` and `human` are retained as schema fields. In this study they mean *source* and *associated rewrite*. The available files do not document per-record generator provenance, human authorship, annotator identity, or dialect verification. We therefore do not treat the column name `human` as proof of authorship or semantic correctness.

Figure 1 shows the executed retrieval and prompting path. Only training sources enter the searchable pool. Their associated rewrites become demonstrations after selection, while the query's own associated rewrite is withheld from the prompt.

![Figure 1. Architecture of the controlled API experiment. Source-only embeddings support training-pool retrieval; the four conditions supply zero, one, or four demonstrations to the same preservation prompt. The query's associated reference is used only for post-generation diagnostics.](figures/new_runtime_architecture.png)

### 3.2 Audited snapshot

The audit uses repository `rizwanullahmuahed2-ui/Persian_Humanizer_Ai` at commit `feb0f282430b13277c43dd32ca93ceec13c837df`. The supplied 21-page PDF is byte-identical to the PDF in that snapshot. The full corpus SHA-256 is `44e9912ce119bf5014165087af10b5ad7ce675ef8e3c17e90cd9031cecf8d39f`. Separate scripts inspect CSV contents, read SQLite metadata without executing serialized index objects, and examine reachable Git history.

The search scope is the available checkout and its seven reachable commits. All seven contain the same 3511-row full corpus. No exact 3173-pair version or generation-output archive was identified within that scope. This does not establish that such material never existed elsewhere. Historical measurements are consequently retained as unverified historical reports, not silently relabeled as results for the current corpus.

### 3.3 Text normalization and overlap

For reproducible comparisons, the audit applies Unicode NFKC normalization, maps Arabic kaf and yeh variants to Persian forms, replaces zero-width non-joiners with spaces, removes tatweel and combining marks, maps Unicode decimal digits to ASCII digits, case-folds, and collapses whitespace. Punctuation is otherwise retained for duplicate detection. Lexical tokens are maximal Unicode alphanumeric spans, excluding underscores. This is an explicit operational choice, not a claim that alternative Persian tokenizers are equivalent.

For each source–rewrite pair, ROUGE-1 and ROUGE-2 use multiset unigram/bigram overlap and ROUGE-L uses the longest common token subsequence. If $m$ is the overlap count, pairwise F1 is $2m/(|a|+|b|)$; empty-token cases score zero. Reported corpus values are macro averages over records. The implementation uses no stemming and is not assumed numerically identical to every ROUGE package.

Metric reporting needs to identify preprocessing and configuration rather than relying on a score name alone [23]. A second diagnostic reproduces the unstemmed core of Google's default ROUGE tokenizer: lowercasing followed by retention of ASCII letters and digits [24]. This comparison assesses implementation sensitivity on the current corpus. Because the historical evaluation script is unavailable in the audited artifacts, it cannot establish which tokenizer produced the earlier manuscript's scores.

### 3.4 Duplicate groups and a new split

Rows are joined into connected groups whenever their normalized source or rewrite equals either field of another row. Group identifiers are hashes of sorted unique texts. A salted hash assigns each entire group to training, development, or test using thresholds 0.70 and 0.80. This creates a new deterministic split; it does not reconstruct an unrecorded historical test set. Exact normalized text overlap across all three partitions is checked explicitly.

The split is intended to prevent direct text leakage. It does not exclude semantic paraphrases, common templates, topic overlap, or possible model-pretraining exposure. It is not stratified to guarantee adequate test coverage for each rare metadata label. Corpus-level analysis has already examined the text; the test partition is therefore a newly frozen project holdout, not an untouched external benchmark.

## 4. Completed audit results

### 4.1 Corpus and version consistency

**Table 1. Manuscript description and audited repository artifacts. Counts concern records, not independent generated outputs.**

| Artifact | Earlier manuscript | Audited snapshot |
|---|---|---|
| Full paired corpus | 3173 pairs | 3511 pairs |
| `style` labels | 10 categories | 22 distinct strings |
| `transformation` labels | 8 types | 43 distinct strings |
| Committed training file | No verifiable split supplied | 3322 pairs; 130 absent from full corpus |
| V2 stored index | Described as corpus index | 1265 records; 113 pairs absent from full corpus |
| Retrieval/generation archive | Summary charts and selected examples | No matching generation-output archive identified |

The full corpus has no blank cells in its five recorded fields and no duplicate source–rewrite pairs after normalization. It has 3505 unique normalized sources: six source groups contain two different associated rewrites each. There are two duplicate-rewrite groups. Multiple references for a source need not be invalid; they must, however, remain in the same partition and their labels and preservation behavior need inspection.

The metadata mixes stylistic properties (`casual`), domains (`medical`, `legal`), and compound labels (`medical_simple`). It is not a uniform taxonomy of register. The largest label is `casual`, with 1702/3511 records (48.48%). The largest transition string is `formal_to_casual`, with 1581 records (45.03%). Examples explicitly collected to change register cannot be assumed to demonstrate within-register preservation.

The training file is not a simple subset of the current full corpus. Its pair intersection with that corpus is 3192; 130 training pairs are outside the full file and 319 full-corpus pairs are absent from training. The latter 319 records are an algebraic set difference, not a recovered historical test set. Subtracting a training count from a full count is therefore not a valid way to reconstruct the experiment.

![Figure 2. Audited corpus composition and source–rewrite measurement sensitivity. The bars describe stored records; they are not generator accuracy or naturalness results.](figures/corpus_audit.png)

### 4.2 Index and effective prompt depth

The legacy SQLite index stores 383 records; the V2 index stores 1265. The legacy index document field concatenates the source and associated rewrite for all 383 records. The V2 document field equals the source for all 1265 records, but 113 V2 metadata pairs do not match the current full corpus. Neither stored collection documents a distance metric in the inspected collection metadata. A dimension of 384 alone does not identify the embedding revision or establish cosine search. A fresh experiment must build an explicitly configured index or cosine matrix from the frozen training partition.

The current code requests up to 20 retrieval candidates and optionally applies an English-trained cross-encoder before selecting up to four. Separately, `format_examples_for_prompt` discards pairs whose source or rewrite exceeds 80 characters and retains at most two surviving pairs. When none remains, `build_prompt` inserts a fixed demonstration. Thus, three quantities must be logged: candidate count, selected retrieval count, and actual prompt demonstration count. They are not interchangeable.

This behavior also invalidates a seemingly straightforward baseline implemented by passing an empty example list to the production prompt builder. That path is a fixed-example condition, not zero-shot prompting. The research harness therefore constructs a genuine zero-example prompt without the fallback. It does not present this new harness as an exact replay of the earlier implementation.

### 4.3 Tokenization changes the numerical picture

**Table 2. Recomputed macro source–associated-rewrite F1 on all 3511 records. Higher or lower values are not interpreted as naturalness.**

| Tokenization | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Explicit Unicode-aware processing | 0.4557 | 0.2643 | 0.4409 |
| ASCII-only diagnostic | 0.0313 | 0.0155 | 0.0312 |

The ASCII-only diagnostic produces empty token lists for both sides of 3287 pairs, approximately 93.62% of the corpus. An identical Persian-only sentence also has zero lexical score under this diagnostic because its tokens are discarded. This demonstrates why implementation checks must precede interpretation. It does not reproduce or explain the historical ROUGE-1 F1 value of 0.018: the audited corpus and computation are explicitly different and the historical script is missing.

The revised interpretation removes both the claim that low ROUGE proves successful rewriting and the suggestion of a universal 0.5 quality threshold. Source–rewrite overlap describes lexical retention in the stored corpus. Generated-output–reference overlap would answer a different question and requires actual generated outputs and trustworthy references. Neither direction alone establishes factual preservation.

### 4.4 Length and numerical diagnostics

Unicode-token counts average 8.45 for sources and 8.53 for associated rewrites. The mean per-pair rewrite/source length ratio is 1.144, while its median is 0.909 and its range is 0.091–10.0. A near-one mean therefore does not show that every rewrite preserves information; both strong shortening and expansion occur. These are properties of the corpus, not measured generation behavior.

Ninety-four pairs contain a digit expression on at least one side. Thirty have differing digit-token multisets. This is a review flag rather than an error count: number words can become digits, and a time such as 16:00 can become 4 p.m. Conversely, a matching number does not guarantee a matching relation, unit, or obligation.

**Table 3. Traceable corpus examples illustrating why references require inspection. Record identifiers follow the exported diagnostic sheet; no error prevalence is inferred from these selected cases.**

| Diagnostic sheet record | Observed source → associated rewrite | Interpretation |
|---|---|---|
| 122 | “two-year warranty” → “2-year warranty” | Digit mismatch can be a benign representation change. |
| 3023 | “8 to 16” → “8 to 4 in the afternoon” | Time representation changes; interpretation requires context. |
| 1061 | “profitable in the long term” → a new “5 years” claim | The new specific duration is not supplied by the source. |
| 1062 | “increases satisfaction” → “92% of users are more satisfied” | The source provides no basis for the added statistic. |
| 1638 | “do not miss this class” → “only 10 places remain” | A new scarcity assertion is introduced. |

The earlier manuscript's five displayed outputs also do not justify categorical preservation claims. Its medical rewrite adds “again” to an instruction to consult a doctor, introducing a possible prior-visit presupposition; its business example changes wording of an attendance obligation. These observations motivate explicit checks of pragmatic force. They are qualitative inspections of displayed examples, not a blinded human study or a measured error rate for the generator.

### 4.5 Retrieval labels and the new split

A same-label fraction should be named *label agreement at k*, not simultaneously precision and recall. If relevance were defined solely as matching a metadata label, then

$$ P@k(q)=\frac{\sum_{d\in R_k(q)}\mathbb{1}[\ell_d=\ell_q]}{k},\qquad
Recall@k(q)=\frac{\sum_{d\in R_k(q)}\mathbb{1}[\ell_d=\ell_q]}{|\{d\in D_{train}:\ell_d=\ell_q\}|}. $$

Recall requires a relevant-pool denominator and is undefined for labels with no relevant training item. Neither metric directly measures whether an example helps a rewrite preserve meaning. The current corpus's majority-label proportion is 48.48%; its expected same-label agreement under uniformly random selection excluding self is 25.95%. These are distinct descriptive baselines. They must not be used to reinterpret the old approximately 0.50 retrieval score as if it were computed on the same corpus and split.

The new grouping procedure yields 3499 connected groups, with largest size two, partitioned as follows.

**Table 4. Newly constructed deterministic group split; no generation results are implied.**

| Partition | Records | Groups | Role |
|---|---:|---:|---|
| Training | 2438 | 2430 | Demonstration retrieval pool |
| Development | 371 | 369 | Implementation checks and declared configuration choices |
| Test | 702 | 700 | Frozen paired comparison inputs |

Normalized text intersections are zero between every partition pair. Near-duplicate and template-family leakage remain unmeasured, and small labels cannot support reliable per-label conclusions merely because a split exists.

## 5. Controlled generation study

### 5.1 Inputs, retrieval, and conditions

The generation study uses 100 test inputs selected before output inspection through seeded SHA-256 ordering, retaining at most one record from each connected text group. The same 100 inputs appear in all four conditions and both models. The sampling seed is 2026; the generation seed is zero. This is a project holdout from the audited corpus, not an external benchmark.

Source fields alone are encoded using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at immutable revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. The model produces 384-dimensional normalized vectors. Retrieval uses exact cosine similarity against the 2438 training sources, with stable identifier tie-breaking. Test references are never embedded, retrieved, or supplied in prompts. The newly computed vectors replace the inconsistent production indices for this experiment; no reranker or register filter is applied.

**Table 5. Executed conditions; each contains 100 responses per model.**

| Condition | Demonstrations actually supplied | Comparison purpose |
|---|---|---|
| LLM-0 | None; no fallback example | Instruction-only baseline |
| Semantic-1 | One nearest training source–rewrite pair | Low-depth semantic guidance |
| Semantic-4 | Four nearest training source–rewrite pairs | Retrieval-depth comparison |
| Random-4 | Four training pairs from a fixed query-specific random seed | Providing examples without semantic selection |

All conditions share the same Persian preservation instruction, which requests natural wording while retaining meaning, numbers, dates, names, units, negation, conditions, certainty, register, and dialect. Examples are explicitly subordinate to the input's register. The full unmodified example pairs are supplied: the production formatter's two-example limit and fixed fallback are absent. The exact prompts, ordered example identifiers, and effective counts of zero, one, or four demonstrations are stored in the frozen request plans.

Semantic-4 versus Random-4 controls example count, although individual demonstration lengths differ. Mean prompt lengths are 162.09, 198.13, 321.60, and 324.44 provider tokens for LLM-0, Semantic-1, Semantic-4, and Random-4, respectively. These values are identical across the two models in the recorded usage. The comparison therefore evaluates the configured retrieval policy and its selected examples; it does not isolate semantic matching from every possible length or example-composition effect.

![Figure 3. Evidence flow for the corpus audit, frozen split, paired API generation, and output analysis. Generation and automatic analysis have been completed; human naturalness assessment is outside the reported outcomes.](figures/evaluation_flow.png)

### 5.2 Models, access deviation, and execution records

The completed comparisons use `gpt-4.1-mini-2025-04-14` and `gpt-4o-mini-2024-07-18` through the OpenAI Chat Completions API. Both use temperature zero, seed zero, one completion per request, and a maximum of 256 completion tokens. No parameter adaptation is performed. Each model–condition cell is generated separately, giving 800 fresh provider response identifiers rather than reused outputs. The 800 responses terminate normally; identical text in different responses is still counted as distinct generation, not as independent evidence beyond its query group.

The original frozen plan paired GPT-4.1 mini with `gpt-4.1-nano-2025-04-14`. Although the latter's metadata endpoint was accessible, all 400 generation requests returned HTTP 404. Those errors remain in the original archive and are excluded from the rewriting results. After a successful generation preflight, GPT-4o mini was declared as an availability-based replacement in a separate plan. Its 400 prompts, query identifiers, retrieved examples, and settings match the corresponding primary-model conditions exactly. The availability amendment was recorded before the replacement outputs were inspected. It changes the model comparison, not the original sample or failed records.

The resulting scientific request archive contains 1200 HTTP attempts: 800 successful generations and 400 preserved access failures. Connectivity preflights are recorded separately and do not belong to the sample. The study logs raw provider responses, request hashes, model snapshots, returned identifiers, usage, finish reasons, and client wall times. An independent checker reconstructs the train-only retrieval and paired design and verifies the raw-response hashes and usage. This is a new cloud-based experiment, not a reproduction of the earlier local Llama execution or evidence of offline deployment performance.

### 5.3 Outcomes and statistical analysis

The output analysis uses the explicit Unicode processing from Section 3.3. It measures normalized exact copying of the source, output–source ROUGE, output–associated-reference ROUGE, and the ratio of output to source token counts. The associated rewrite is an unvalidated corpus reference, especially where its recorded transformation changes register; reference agreement is therefore a descriptive outcome rather than task accuracy. Similarly, more or less source overlap is not unconditionally better. Already clear wording may need little change, while an incorrect paraphrase can retain many words.

Paired resampling has a well-established role in comparing NLP systems evaluated on shared inputs [25]. The analysis specification was sealed before response contents were inspected, with a separate availability amendment for the replacement model. Paired differences use 100 query groups and 10000 fixed-seed bootstrap resamples, retaining the matched model–condition values for each query. The percentile 95% intervals are descriptive and unadjusted for multiple comparisons. One generation per cell is available, so the intervals do not estimate variability over repeated API generations. Four conditions and two models do not turn the 100 inputs into 800 independent test questions.

## 6. Generation results

### 6.1 Lexical retention, reference agreement, and length

**Table 6. Automatic output diagnostics on the same 100 query groups. ROUGE-L values are macro F1; length is the mean output/source token ratio; copies are normalized exact matches out of 100.**

| Model | Condition | Source ROUGE-L | Reference ROUGE-L | Length ratio | Copies |
|---|---|---:|---:|---:|---:|
| GPT-4.1 mini | LLM-0 | 0.6251 | 0.3621 | 1.1876 | 1 |
| GPT-4.1 mini | Semantic-1 | 0.6128 | 0.4088 | 1.1753 | 0 |
| GPT-4.1 mini | Semantic-4 | 0.6437 | 0.4513 | 1.1347 | 1 |
| GPT-4.1 mini | Random-4 | 0.6574 | 0.4464 | 1.0687 | 1 |
| GPT-4o mini | LLM-0 | 0.6086 | 0.3647 | 1.1844 | 1 |
| GPT-4o mini | Semantic-1 | 0.6319 | 0.3813 | 1.1931 | 0 |
| GPT-4o mini | Semantic-4 | 0.6571 | 0.4562 | 1.1033 | 1 |
| GPT-4o mini | Random-4 | 0.6974 | 0.4470 | 1.0759 | 4 |

![Figure 4. Output–source and output–associated-reference ROUGE-L across the four conditions and two models. Each mean uses the same 100 query groups. These are lexical-agreement diagnostics, not naturalness or factual-accuracy scores.](figures/new_generation_outcomes.png)

Four semantic examples move the generated wording closer to the associated reference than zero examples for both models. Random examples produce a similar change. Exact copying is uncommon under every condition, but its rarity alone does not demonstrate useful rewriting. The 800 successful responses contain 574 distinct output strings, reflecting some identical rewrites across conditions or models; the independent checker verifies 800 distinct provider response identifiers.

The mean length ratio decreases from 1.1876 to 1.1347 for GPT-4.1 mini and from 1.1844 to 1.1033 for GPT-4o mini when moving from LLM-0 to Semantic-4. Their paired differences are −0.0528 [−0.0976, −0.0082] and −0.0811 [−0.1268, −0.0372], respectively. These changes concern expansion relative to the input, not verified information retention.

### 6.2 Does semantic selection outperform random examples?

**Table 7. Mean paired ROUGE-L differences and unadjusted percentile 95% bootstrap intervals over 100 query groups.**

| Model | Contrast | Source ROUGE-L difference [95% CI] | Reference ROUGE-L difference [95% CI] |
|---|---|---|---|
| GPT-4.1 mini | Semantic-4 minus LLM-0 | 0.0186 [−0.0192, 0.0541] | 0.0892 [0.0611, 0.1191] |
| GPT-4.1 mini | Semantic-4 minus Random-4 | −0.0137 [−0.0473, 0.0192] | 0.0049 [−0.0221, 0.0317] |
| GPT-4o mini | Semantic-4 minus LLM-0 | 0.0485 [0.0117, 0.0851] | 0.0915 [0.0591, 0.1263] |
| GPT-4o mini | Semantic-4 minus Random-4 | −0.0403 [−0.0792, −0.0022] | 0.0092 [−0.0232, 0.0427] |

The reference-overlap intervals for Semantic-4 minus LLM-0 are above zero in both models. In contrast, the Semantic-4 minus Random-4 intervals include zero for both models. The results thus provide evidence that example prompting changes agreement with the corpus rewrite distribution, but do not establish an advantage of semantic selection over random examples on this outcome. An interval spanning zero is not proof of equivalence; more precise estimates or a different dataset could produce a different result.

For GPT-4o mini, source overlap is lower with Semantic-4 than Random-4, with a paired interval below zero. This is a wording-retention difference, not evidence that one condition preserves meaning better. Given the corpus's register shifts and unsupported additions, interpreting reference convergence as improved naturalization would repeat the measurement error identified in the audit.

![Figure 5. Paired differences in agreement with the associated corpus rewrite. Error bars are descriptive, unadjusted 95% percentile bootstrap intervals over the same 100 query groups. They are not intervals for human naturalness or semantic accuracy.](figures/paired_reference_contrasts.png)

The distributions in Figure 6 complement the means: responses can expand or shorten even when the average length ratio is near one. A ratio of one indicates equal token counts, not semantic equivalence.

![Figure 6. Per-query output/source token-ratio distributions for the same 100 inputs in each model–condition cell. The horizontal reference at one marks equal length. Variation in length is descriptive and does not identify omissions or unsupported additions by itself.](figures/new_length_distributions.png)

### 6.3 Execution cost and numerical coverage

The main study uses 201252 prompt tokens and 11556 completion tokens, totaling 212808, with zero cached prompt tokens reported. At the recorded uncached input/output prices, the successful generations have an estimated token cost of USD 0.06805390. This is a usage-based estimate, not an invoice or a measure of end-to-end deployment cost; local embedding computation, development, and preflights are separate.

None of the 100 sampled sources or their 800 outputs contains a digit expression under the declared diagnostic. Consequently, the numerical-preservation denominator is zero: a rate is undefined, not 100%. To address that coverage gap, a separately declared posthoc module selected all ten source-digit records in the test partition, each from a distinct group absent from the main sample. The same four conditions and two snapshots generated 80 additional normal-stop responses. This targeted module is not pooled into a representative error rate for the main sample.

**Table 8. Digit-expression multiset flags in the separate diagnostic module. Each cell contains ten source-digit inputs. Flags indicate a representation mismatch, not an adjudicated factual error.**

| Model | LLM-0 | Semantic-1 | Semantic-4 | Random-4 |
|---|---:|---:|---:|---:|
| GPT-4.1 mini | 0 | 0 | 1 | 0 |
| GPT-4o mini | 1 | 1 | 1 | 1 |

![Figure 7. Numerical coverage and digit-expression flags. The main sample has zero source-digit inputs among 100 queries; the separate targeted module has ten among ten. Each heatmap cell uses ten inputs. All five flags concern the same 24-hour to 12-hour time representation change, rather than five established factual errors.](figures/new_numerical_diagnostics.png)

All five flags concern the same input, `PH-c77b21b3af14e7f13b04`: working hours written as 8–17 become 8 to 5 with an afternoon/evening marker. This is consistent with a 24-hour to 12-hour representation change rather than five numerical errors. Conversely, a matching digit multiset would not establish preservation of units, relations, or non-numerical meaning. The module used 20552 prompt and 1607 completion tokens, with an estimated uncached successful-generation cost of USD 0.00741800. Across both modules there are 880 distinct successful provider response identifiers and 1280 scientific HTTP attempts, including the 400 retained unavailable-model failures. The estimated successful-generation token cost is USD 0.07547190; connectivity preflights are excluded.

### 6.4 Model-assisted case inspection

Eight of the main query groups were selected by a fixed hash rule before their responses were read for case inspection. All eight model–condition outputs per group were inspected, yielding 64 cases. This is an explicitly model-assisted qualitative analysis, not human ratings, inter-rater validation, or an estimate of population error frequency. The accompanying notes retain exact source/output quotations and request identifiers.

Three cases clarify what the automatic outcomes miss. For `PH-8430ee192d142acff80c`, GPT-4o mini without examples changes “this message was not sent” to “this message did not reach the destination”; failure to send and failure to arrive are not equivalent events. For `PH-0ba65753f93d3382d459`, both Semantic-4 outputs render a formal annual-inspection sentence using the colloquial verb form *beshan*, and GPT-4.1 mini also uses the colloquial *ye* (“a/one”). The selected demonstrations contain colloquial forms, making demonstration influence a plausible reading, though this case alone does not isolate its cause. For `PH-70f7edba57adffbb1749`, GPT-4o mini in Semantic-4 changes an explicitly urgent board-meeting item into a statement of priority, omitting the explicit urgency qualifier. These examples motivate semantic and register assessment while leaving the quantitative claims limited to the measured diagnostics.

## 7. Discussion

The artifact audit and paired study address different parts of the evidence chain. The audit shows how a stale index, a hidden fixed example, a formatter that reduces demonstration count, and script-incompatible tokenization can undermine reproducibility. The controlled study then measures actual generation under explicit conditions rather than attributing corpus statistics to a deployed model.

The principal generation finding is the importance of the random-example control. A zero-example comparison alone would show higher reference overlap with four semantically selected demonstrations in both models. The random-example condition reveals that a similar shift occurs without semantic selection. This limits the claim that retrieval itself improves the measured outcome, even before considering whether the associated references define the intended naturalization task.

Corpus alignment remains important. Demonstrations that simplify legal wording, add persuasive details, or convert formal text to casual speech can be suitable for another task while conflicting with preservation. Improving retrieval similarity does not correct a misleading demonstration target. The observed changes in lexical agreement should therefore motivate validation of the demonstration pool rather than automatic optimization of overlap scores.

The Persian/Dari distinction also requires care. These experiments use the current mixed Persian corpus; they do not establish naturalness for Afghan Dari readers. Lexical marker counts in the application include shared words and substring matches and are not calibrated dialect accuracy. Language variety should be assessed against the input, with documented material and judgments from relevant speakers.

## 8. Limitations

The audit covers one repository snapshot and its reachable history; it does not recover the historical 3173-pair corpus or reproduce the original Llama measurements. The paired cloud experiment uses two small models from one provider, one embedding model, short inputs, and one response per condition. It does not establish cross-provider generalization, local-runtime performance, or robustness on long documents.

Normalized grouping prevents direct shared-text leakage but not semantic or template overlap. The metadata mixes styles and domains, rare labels have small samples, and the associated references have not been validated as gold outputs for within-register naturalization. Semantic-4 and Random-4 match demonstration count but not every aspect of length or content. Bootstrap intervals describe query sampling under this configuration and do not incorporate repeated-generation variability.

No blinded human naturalness ratings or population-level factual-preservation accuracy are reported. The automatic outcomes answer the stated observable-generation question; they do not establish that the text is more natural, more human-authored, or factually equivalent. The posthoc numeric module addresses a coverage gap in the main sample but cannot supply a representative estimate for all Persian rewriting. Source permissions, per-record provenance, and dialect coverage also remain incompletely documented.

## 9. Conclusion

A reproducible study of Persian rewriting requires consistent artifacts, script-aware measurement, and a control for providing examples without semantic retrieval. The completed audit identifies corpus/index mismatches and a hidden fixed-example baseline; explicit Unicode processing corrects the interpretation of low-overlap descriptions. In 800 actual responses from two model snapshots, four semantic demonstrations increase associated-reference overlap relative to zero examples, but their advantage over four random demonstrations is not established by the paired confidence intervals. The measured effect is therefore attributable to demonstration prompting more broadly, while a benefit of semantic selection remains unconfirmed on this outcome. These findings support evidence-traceable evaluation and validated demonstration resources rather than treating overlap change as proof of meaning-preserving naturalness.

## Data, code, and declarations

**Availability.** The audited source files are publicly accessible at [the project repository](https://github.com/rizwanullahmuahed2-ui/Persian_Humanizer_Ai/tree/feb0f282430b13277c43dd32ca93ceec13c837df). The revision package includes corpus diagnostics, group split files, embedding metadata, frozen request plans, raw API responses, availability errors, independent analysis code, and checksums. Public access alone does not establish a dataset license, record-level authorship, or consent; those provenance and reuse questions require documentation by the authors.

**Generation and assessment.** The reported cloud generation and automatic analyses have been executed. They are separate from the earlier local prototype. No blinded human ratings are claimed. Original PDF figures and files remain historical material and are not relabeled as new experimental evidence.

**Author declarations.** Author names and affiliation are retained from the supplied manuscript. Both authors should confirm funding, contributions, competing interests, any applicable ethics declaration, and AI-assistance disclosure under the selected journal's policy.

## References

[1] A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, vol. 30, 2017. https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[2] T. B. Brown et al., “Language Models are Few-Shot Learners,” Advances in Neural Information Processing Systems, vol. 33, pp. 1877–1901, 2020. https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html

[3] P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” *NeurIPS*, vol. 33, pp. 9459–9474, 2020. https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

[4] J. Li, R. Jia, H. He, and P. Liang, “Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer,” *NAACL-HLT*, pp. 1865–1874, 2018. https://aclanthology.org/N18-1169/

[5] E. Reif, D. Ippolito, A. Yuan, A. Coenen, C. Callison-Burch, and J. Wei, “A Recipe for Arbitrary Text Style Transfer with Large Language Models,” *ACL, Volume 2: Short Papers*, pp. 837–848, 2022. https://aclanthology.org/2022.acl-short.94/

[6] J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin, and W. Chen, “What Makes Good In-Context Examples for GPT-3?” DeeLIO, pp. 100–114, 2022. https://aclanthology.org/2022.deelio-1.10/

[7] O. Rubin, J. Herzig, and J. Berant, “Learning To Retrieve Prompts for In-Context Learning,” NAACL-HLT, pp. 2655–2671, 2022. https://aclanthology.org/2022.naacl-main.191/

[8] S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer, “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?” EMNLP, pp. 11048–11064, 2022. https://aclanthology.org/2022.emnlp-main.759/

[9] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” EMNLP-IJCNLP, pp. 3982–3992, 2019. https://aclanthology.org/D19-1410/

[10] N. Reimers and I. Gurevych, “Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation,” EMNLP, pp. 4512–4525, 2020. https://aclanthology.org/2020.emnlp-main.365/

[11] V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense Passage Retrieval for Open-Domain Question Answering,” EMNLP, pp. 6769–6781, 2020. https://aclanthology.org/2020.emnlp-main.550/

[12] P. Falakaflaki and M. Shamsfard, “Formality Style Transfer in Persian,” arXiv:2406.00867, 2024. https://arxiv.org/abs/2406.00867

[13] V. Tajalli, M. Shamsfard, and F. Kalantari, “Developing an Informal-Formal Persian Corpus: Highlighting the Differences between Two Writing Styles,” *AbjadNLP*, pp. 44–53, 2025. https://aclanthology.org/2025.abjadnlp-1.6/

[14] M. S. Rasooli, F. Bakhtyari, F. Shafiei, M. Ravanbakhsh, and C. Callison-Burch, “Automatic Standardization of Colloquial Persian,” arXiv:2012.05879, 2020. https://arxiv.org/abs/2012.05879

[15] R. Mir, B. Felbo, N. Obradovich, and I. Rahwan, “Evaluating Style Transfer for Text,” *NAACL-HLT*, pp. 495–504, 2019. https://aclanthology.org/N19-1049/

[16] E. Briakou, S. Agrawal, K. Zhang, J. Tetreault, and M. Carpuat, “A Review of Human Evaluation for Style Transfer,” *GEM*, pp. 58–67, 2021. https://aclanthology.org/2021.gem-1.6/

[17] C.-Y. Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,” *Text Summarization Branches Out*, pp. 74–81, 2004. https://aclanthology.org/W04-1013/

[18] M. Popović, “chrF: character n-gram F-score for automatic MT evaluation,” *WMT*, pp. 392–395, 2015. https://aclanthology.org/W15-3049/

[19] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating Text Generation with BERT,” *ICLR*, 2020. https://arxiv.org/abs/1904.09675

[20] A. Grattafiori et al., “The Llama 3 Herd of Models,” arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783

[21] Meta, “Llama 3.1 8B Instruct: Model Card,” model documentation. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

[22] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford, “Datasheets for Datasets,” Communications of the ACM, 2021. Author version: https://arxiv.org/abs/1803.09010

[23] M. Post, “A Call for Clarity in Reporting BLEU Scores,” WMT, pp. 186–191, 2018. https://aclanthology.org/W18-6319/

[24] Google Research, “ROUGE tokenizer,” implementation source, `rouge/tokenize.py`. https://github.com/google-research/google-research/blob/master/rouge/tokenize.py

[25] P. Koehn, “Statistical Significance Tests for Machine Translation Evaluation,” EMNLP, pp. 388–395, 2004. https://aclanthology.org/W04-3250/
