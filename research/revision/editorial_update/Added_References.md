# Verified additions to the reference list

The current 14 references can be expanded to 25 with these 11 relevant sources. Proposed IDs 15–25 are temporary: renumber all citations in first-use order after editorial integration. No manuscript was modified by this reference audit.

Each entry below was checked against the primary ACL Anthology, NeurIPS publication record or author-posted arXiv record. The metadata omitted in a specific note was not independently verified.

## Proposed [15] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” EMNLP-IJCNLP, pp. 3982–3992, 2019. https://aclanthology.org/D19-1410/

**Use:** Related work / semantic retrieval method. Sentence-level embeddings make cosine comparison practical without jointly encoding every candidate pair. This supports the representation choice, not a claim that the present encoder has been validated for Dari naturalness.

**Suggested integration:** Our retriever uses the sentence-embedding formulation introduced by Sentence-BERT [15], where an input and candidate source texts are encoded separately and compared by cosine similarity.

## Proposed [16] Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation

N. Reimers and I. Gurevych, “Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation,” EMNLP, pp. 4512–4525, 2020. https://aclanthology.org/2020.emnlp-main.365/

**Use:** Related work / semantic retrieval method. The paper extends sentence embeddings across languages through aligned representations. It explains a multilingual retrieval design but does not verify dialect quality of the current corpus.

**Suggested integration:** Multilingual sentence-embedding work extends this shared-space approach through knowledge distillation [16]; the present study uses a fixed pretrained multilingual encoder rather than training a task-specific retriever.

## Proposed [17] Attention Is All You Need

A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, vol. 30, 2017. https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

**Use:** Related work, concise generation background. Introduces the Transformer attention architecture underlying contemporary generation and encoding systems. Keep the role limited to architectural background; it supplies no Persian humanization result.

**Suggested integration:** Transformer architectures provide the basis for the generation and encoding components [17], while few-shot prompting allows task examples to be supplied at inference time [18].

## Proposed [18] Language Models are Few-Shot Learners

T. B. Brown et al., “Language Models are Few-Shot Learners,” Advances in Neural Information Processing Systems, vol. 33, pp. 1877–1901, 2020. https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html

**Use:** Related work, in-context learning. Demonstrates few-shot application without gradient updates during task interaction. It supports prompting as a distinct adaptation mechanism, not equivalence to fine-tuning or guaranteed naturalness.

**Suggested integration:** Brown et al. showed that language models can perform tasks using instructions and demonstrations without task-time gradient updates [18].

## Proposed [19] What Makes Good In-Context Examples for GPT-3?

J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin, and W. Chen, “What Makes Good In-Context Examples for GPT-3?” DeeLIO, pp. 100–114, 2022. https://aclanthology.org/2022.deelio-1.10/

**Use:** Related work / random-example control rationale. Tests semantic-neighbor demonstration selection against random sampling on other NLP tasks. It makes the same control relevant here without implying the earlier advantage transfers to Persian rewriting.

**Suggested integration:** Liu et al. found that semantic example selection can improve GPT-3 performance over random selection on several tasks [19], motivating a direct comparison rather than assuming the same benefit for Persian rewriting.

## Proposed [20] Learning To Retrieve Prompts for In-Context Learning

O. Rubin, J. Herzig, and J. Berant, “Learning To Retrieve Prompts for In-Context Learning,” NAACL-HLT, pp. 2655–2671, 2022. https://aclanthology.org/2022.naacl-main.191/

**Use:** Related work, learned vs fixed retrieval. Learns prompt retrieval from an LM-based supervision signal for semantic parsing. The present work uses fixed embeddings and no retrieval training, so distinguish the two designs.

**Suggested integration:** Rubin et al. instead train a prompt retriever using language-model preferences [20]; our fixed cosine selector is a simpler mechanism whose utility must be measured separately.

## Proposed [21] Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer, “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?” EMNLP, pp. 11048–11064, 2022. https://aclanthology.org/2022.emnlp-main.759/

**Use:** Related work / Discussion of demonstration effect. Identifies label-space, input-distribution and formatting effects in classification/multiple-choice in-context learning. These findings motivate controls but do not show that false rewrite targets are harmless.

**Suggested integration:** Min et al. show that demonstration format and input distribution can explain important in-context effects in classification and multiple-choice tasks [21]. Their result supports examining what examples contribute, but does not establish that inaccurate rewrite demonstrations are safe.

## Proposed [22] Statistical Significance Tests for Machine Translation Evaluation

P. Koehn, “Statistical Significance Tests for Machine Translation Evaluation,” EMNLP, pp. 388–395, 2004. https://aclanthology.org/W04-3250/

**Use:** Section 5.3, paired resampling. Provides an NLP precedent for paired resampling when comparing systems on shared inputs. The current descriptive percentile intervals are not presented as a replication of its exact MT test or multiplicity-adjusted confirmatory inference.

**Suggested integration:** Following the paired-comparison principle used in machine-translation evaluation [22], we resample query groups while retaining the matched condition values for each input.

## Proposed [23] A Call for Clarity in Reporting BLEU Scores

M. Post, “A Call for Clarity in Reporting BLEU Scores,” WMT, pp. 186–191, 2018. https://aclanthology.org/W18-6319/

**Use:** Text processing/measurement transparency. Documents score variation caused by metric configuration and reference tokenization. It supports transparent preprocessing reports; this study does not claim to have calculated BLEU.

**Suggested integration:** The need to record preprocessing and metric configuration is well established in machine-translation evaluation [23]; here it is especially important because script filtering can erase Persian tokens.

## Proposed [24] Dense Passage Retrieval for Open-Domain Question Answering

V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense Passage Retrieval for Open-Domain Question Answering,” EMNLP, pp. 6769–6781, 2020. https://aclanthology.org/2020.emnlp-main.550/

**Use:** Related work, knowledge retrieval versus example retrieval. Establishes a learned dual-encoder passage-retrieval formulation for question answering. The present example retrieval is task-different and does not train or deploy the DPR model.

**Suggested integration:** Dense passage retrieval learns query–passage representations for question answering [24], whereas our retrieval pool supplies example transformations rather than new factual evidence.

## Proposed [25] Datasheets for Datasets

T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford, “Datasheets for Datasets,” Communications of the ACM, 2021. Author version: https://arxiv.org/abs/1803.09010

**Use:** Corpus provenance / data availability. Proposes documenting dataset motivation, composition, collection and intended uses. It supports identifying missing provenance rather than treating a column label as proof of human authorship.

**Suggested integration:** A dataset description should document how examples were collected, who produced them, and what uses they support, following the datasheet approach [25]. Those details are not recoverable from the column names alone.

**Verification note:** Author-posted arXiv v8 explicitly reports publication in CACM December2021. Publisherpage returned403; journalvolume/pages and publisherDOI intentionally omitted rather than claimed verified.

## Suggested organization rather than citation padding

1. Add a compact related-work paragraph connecting Transformer/few-shot prompting with semantic and learned demonstration selection [17–21].
2. Explain the actual sentence-representation choice with [15,16], while separating it from trained QA passage retrieval [24].
3. Cite [23] in the explicit text-processing section and [22] in paired bootstrap methods.
4. Use [25] where record-level provenance and intended use are discussed.

Do not claim the new sources independently validate the current Persian/Dari results. In particular, the Min et al. finding about randomized classification labels is not evidence that incorrect factual rewrite targets can be used safely.
