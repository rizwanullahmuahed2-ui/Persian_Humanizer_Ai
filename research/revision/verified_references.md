# Verified reference and evaluation audit

Verification scope: the uploaded 21-page PDF and primary publication records, paper text, model card and implementation source. This file reports a literature/source audit, not a new generation experiment or human evaluation. Retrieval dates and live repository content may change; pin versions for an experiment.

## Immediate changes needed

1. Remove every inference that lower ROUGE establishes better humanization. Distinguish source–reference corpus overlap, output–source edit distance, and output–reference agreement. None establishes naturalness or factual preservation alone.
2. Investigate Persian tokenization before reusing Table V. The Google `rouge-score` default tokenizer filters to ASCII letters and digits. Pure Persian identical sentences can therefore produce empty token lists and zero overlap. This is a concrete failure mode, not proof that the historical computation used that implementation.
3. Correct the novelty comparison: zero-shot style transfer without fine-tuning already exists; original Lewis RAG involved fine-tuning. Neither tool integration nor avoiding fine-tuning alone establishes novelty.
4. Treat Persian/Dari use of Llama 3.1 as application-specific experimental evidence. Persian is absent from the official eight-language supported list.
5. Report naturalness, meaning/fact preservation and register preservation separately. A generic embedding score cannot certify preserved negation, obligations, dates or numerical relations.

## Primary sources and precise permitted uses

### 1. Reif et al. (original reference [5])

Emily Reif, Daphne Ippolito, Ann Yuan, Andy Coenen, Chris Callison-Burch, and Jason Wei. 2022. *A Recipe for Arbitrary Text Style Transfer with Large Language Models*. ACL, Volume 2: Short Papers, pp. 837–848. DOI: 10.18653/v1/2022.acl-short.94.

The work presents instruction-based, augmented zero-shot rewriting without model fine-tuning or target-style exemplars. Cite it when describing the no-retrieval baseline and related prompting methods. It prevents claiming that earlier style-transfer work uniformly requires supervised adaptation. It does not establish performance on this paper's Dari collection. [Primary record](https://aclanthology.org/2022.acl-short.94/)

### 2. Falakaflaki and Shamsfard (original [6])

Parastoo Falakaflaki and Mehrnoush Shamsfard. 2024. *Formality Style Transfer in Persian*. arXiv:2406.00867v1.

Fa-BERT2BERT converts informal text to formal text using supervised modeling. Section 5.2, PDF pp. 10–11, explains that overlapping unchanged words can reward insufficient stylistic conversion and proposes register-specific and syntactic measures. This supports supplementing lexical scores; it does **not** support minimizing ROUGE. Its reported reference-based ROUGE-L comparisons reward higher scores. Its objective differs from within-register naturalization, so use an appropriately limited comparison. [Record](https://arxiv.org/abs/2406.00867) · [Full paper](https://arxiv.org/pdf/2406.00867)

### 3. Tajalli corpus (original [7], newer published version)

The original citation is valid: Vahide Tajalli, Fateme Kalantari, and Mehrnoush Shamsfard. 2023. *Developing an Informal-Formal Persian Corpus*. arXiv:2308.05336. [Original record](https://arxiv.org/abs/2308.05336)

A later publication is available: Vahide Tajalli, Mehrnoush Shamsfard, and Fateme Kalantari. 2025. *Developing an Informal-Formal Persian Corpus: Highlighting the Differences between Two Writing Styles*. Proceedings of the 1st Workshop on NLP for Languages Using Arabic Script, pp. 44–53. The author order differs between records: preserve the order of the cited version. It reports 50000 aligned sentence pairs. Neither this resource's existence nor its metadata verifies the authorship, license or dialect composition of the separate 3173-pair collection. [Published primary record](https://aclanthology.org/2025.abjadnlp-1.6/)

### 4. Rasooli et al. (original [15])

Mohammad Sadegh Rasooli, Farzane Bakhtyari, Fatemeh Shafiei, Mahsa Ravanbakhsh, and Chris Callison-Burch. 2020. *Automatic Standardization of Colloquial Persian*. arXiv:2012.05879.

The work standardizes Iranian colloquial Persian using sequence-to-sequence modeling and synthetic parallel material. It is related to normalization, not direct evidence of within-register humanization or Dari quality. The original reference's title, author order and identifier match the primary record. [Primary record](https://arxiv.org/abs/2012.05879)

### 5. Llama 3 family and 3.1 model card (original [12])

Aaron Grattafiori et al. 2024. *The Llama 3 Herd of Models*. arXiv:2407.21783. Use `A. Grattafiori et al.` or the full author list, not `and et al.`. The current arXiv record is v3. [Primary paper](https://arxiv.org/abs/2407.21783)

The Llama 3.1 8B Instruct model card lists English, German, French, Italian, Portuguese, Hindi, Spanish and Thai as supported languages. Persian/Dari is not listed; this does not mean generation is impossible, but its adequacy must be evaluated locally. Record exact model variant, model digest, quantization and runtime rather than relying only on the mutable Ollama tag. Describe weights using their actual license rather than assuming an unrestricted open-source license. [Official model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### 6. Google ROUGE implementation

Google Research, `rouge/tokenize.py`: the default tokenization patterns retain ASCII `a-z` and `0-9`. `RougeScorer` uses `DefaultTokenizer` when no custom tokenizer is supplied. A Persian-aware tokenizer must be explicitly supplied and tested, including identical Persian inputs, mixed digits, negation and unrelated outputs. A stemming toggle alone does not fix script deletion. This is implementation evidence, not a diagnosis of the unavailable historical script. [Tokenizer source](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py) · [Scorer source](https://raw.githubusercontent.com/google-research/google-research/master/rouge/rouge_scorer.py)

### 7. chrF

Maja Popović. 2015. *chrF: character n-gram F-score for automatic MT evaluation*. WMT, pp. 392–395. DOI: 10.18653/v1/W15-3049.

Character n-gram overlap provides an additional script-friendly reference-agreement diagnostic with less dependence on word segmentation. Cite the original metric; report actual implementation and parameters. It remains an overlap measure and cannot by itself prove naturalness or meaning preservation. [Primary record](https://aclanthology.org/W15-3049/)

### 8. BERTScore

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. *BERTScore: Evaluating Text Generation with BERT*. ICLR; arXiv:1904.09675 (first posted 2019).

BERTScore aligns contextual token representations instead of requiring exact lexical identity. The released package supports multilingual BERT. For Persian specify the encoder, layer, package/model versions, IDF use and baseline rescaling. Its results are contextual-similarity proxies, not conclusive tests of factual equivalence or humanlike style; compare fact-sensitive perturbations as well. [Primary paper](https://arxiv.org/abs/1904.09675) · [Author implementation](https://github.com/Tiiiger/bert_score)

### 9. Human evaluation protocol

Eleftheria Briakou, Sweta Agrawal, Ke Zhang, Joel Tetreault, and Marine Carpuat. 2021. *A Review of Human Evaluation for Style Transfer*. GEM, pp. 58–67. DOI: 10.18653/v1/2021.gem-1.6.

The review covers 97 style-transfer papers and identifies underspecified evaluation protocols. Use it to motivate explicit instructions, sampling, assessed dimensions and annotator reporting. For this study distinguish naturalness from content preservation and register match. Keep any author inspection or AI assistance accurately labeled; no agreement statistic exists until ratings are actually collected. [Primary record](https://aclanthology.org/2021.gem-1.6/)

### 10. Separate quality dimensions

Remi Mir, Bjarke Felbo, Nick Obradovich, and Iyad Rahwan. 2019. *Evaluating Style Transfer for Text*. NAACL-HLT, Volume 1, pp. 495–504. DOI: 10.18653/v1/N19-1049.

The paper separates style-transfer intensity, content preservation and naturalness, showing tradeoffs rather than a single quality axis. For the present task the style objective is register preservation, so adapt the dimensions explicitly rather than importing a direction-of-transfer score as humanization. Its experiments use sentiment material, not validation of this paper's Persian/Dari outputs. [Primary record](https://aclanthology.org/N19-1049/)

### 11. Original RAG (original [1])

Patrick Lewis et al. 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 33, pp. 9459–9474.

The original research explicitly presents and evaluates a fine-tuning recipe. Therefore Table I's blanket “Fine-tuning required ×” for this study is misleading. The current pipeline can use a frozen generator and example retrieval, but that is a different implementation choice. RAG's results on knowledge-intensive tasks do not establish improvements on this humanization task. The manuscript's seven-author list omits several authors without an `et al.` marker; shorten consistently. [Primary publication](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

### 12. Original ROUGE (original [24])

Chin-Yew Lin. 2004. *ROUGE: A Package for Automatic Evaluation of Summaries*. Text Summarization Branches Out, pp. 74–81.

Keep this as the metric's original citation. Specify target and prediction roles when reporting precision and recall and distinguish reference agreement from source retention. The publication does not provide a universal 0.5 threshold or license interpreting low overlap as successful naturalization. [Primary record](https://aclanthology.org/W04-1013/)

## Suggested replacement evaluation paragraph

“Lexical metrics describe agreement with a reference or retention of source wording; their direction and interpretation depend on which texts are compared. We do not treat low ROUGE as evidence of humanization. We compute reference-based overlap using a Persian-aware tokenizer and report character-based overlap as a complementary diagnostic. Contextual similarity, numeral and entity consistency, negation and obligation checks, and a documented manual assessment address different aspects of preservation and naturalness. Improvements attributable to retrieval require paired comparisons against the same generator without examples and with randomly selected examples.”

This paragraph is a proposed method. Do not write “we compute” or claim completed assessment in a results section until those operations have actually been run.
