"""Add verified scholarship and evidence-backed figures; research records unchanged."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'manuscript/Revised_Manuscript_EN.md'

def main():
    text=PAPER.read_text()
    assert 'new_runtime_architecture.png' not in text,'Integration already applied'
    body,biblio=text.split('\n## References\n',1)
    refs={int(n):entry.strip()for n,entry in re.findall(r'^\[(\d+)\]\s+([\s\S]*?)(?=^\[\d+\]\s|\Z)',biblio,re.M)}
    added=json.loads((ROOT/'editorial_update/refs.json').read_text())['references']
    for item in added:refs[item['proposed_id']]=item['bibliography']
    assert len(refs)==25
    marker='## 2. Related work and scope\n'
    body=body.replace(marker,marker+'\nTransformer architectures underpin the encoding and generation components [17], while few-shot language modeling shows how demonstrations can guide a task at inference time without parameter updates [18]. For rewriting, the question is whether the chosen demonstrations offer more useful guidance than examples selected without semantic matching.\n')
    marker='Persian formality transfer and normalization provide useful related resources'
    addition='''Example selection has been studied directly in in-context learning. Liu et al. compare semantically retrieved examples with random selection for GPT-3 [19], and Rubin et al. learn a prompt retriever using language-model feedback [20]. Min et al. show that demonstration format and input distribution can explain important effects in classification and multiple-choice tasks [21]. These findings motivate controls in the present study; they do not establish that semantic selection will help Persian rewriting or that inaccurate rewrite targets are harmless.

The representation and retrieval choices also require a distinction. Sentence-BERT encodes sentences separately for efficient similarity comparison [15], and multilingual knowledge distillation extends this approach across languages [16]. Dense passage retrieval, by contrast, learns query–passage representations for question answering [24]. Here the retrieved pairs illustrate transformations: they are not sources of additional facts to insert into a rewrite.

'''
    assert marker in body;body=body.replace(marker,addition+marker)
    marker='### 3.2 Audited snapshot'
    figure='''Figure 1 shows the executed retrieval and prompting path. Only training sources enter the searchable pool. Their associated rewrites become demonstrations after selection, while the query's own associated rewrite is withheld from the prompt.

![Figure 1. Architecture of the controlled API experiment. Source-only embeddings support training-pool retrieval; the four conditions supply zero, one, or four demonstrations to the same preservation prompt. The query's associated reference is used only for post-generation diagnostics.](figures/new_runtime_architecture.png)

'''
    body=body.replace(marker,figure+marker)
    marker='The corpus column names `ai` and `human`'
    body=body.replace(marker,'Dataset documentation should explain how examples were collected, who produced them, and what uses they support, as emphasized by the datasheet approach [25]. '+marker)
    marker='A second diagnostic reproduces the unstemmed core'
    body=body.replace(marker,'Metric reporting needs to identify preprocessing and configuration rather than relying on a score name alone [23]. '+marker)
    body=body.replace('![Figure 1. Audited corpus','![Figure 2. Audited corpus').replace('![Figure 2. Evidence flow','![Figure 3. Evidence flow').replace('![Figure 3. Paired differences','![Figure 5. Paired differences')
    marker='The analysis specification was sealed before response contents were inspected'
    body=body.replace(marker,'Paired resampling has a well-established role in comparing NLP systems evaluated on shared inputs [22]. '+marker)
    marker='Four semantic examples move the generated wording closer'
    figure='''![Figure 4. Output–source and output–associated-reference ROUGE-L across the four conditions and two models. Each mean uses the same 100 query groups. These are lexical-agreement diagnostics, not naturalness or factual-accuracy scores.](figures/new_generation_outcomes.png)

'''
    body=body.replace(marker,figure+marker)
    marker='### 6.3 Execution cost and numerical coverage'
    figure='''The distributions in Figure 6 complement the means: responses can expand or shorten even when the average length ratio is near one. A ratio of one indicates equal token counts, not semantic equivalence.

![Figure 6. Per-query output/source token-ratio distributions for the same 100 inputs in each model–condition cell. The horizontal reference at one marks equal length. Variation in length is descriptive and does not identify omissions or unsupported additions by itself.](figures/new_length_distributions.png)

'''
    body=body.replace(marker,figure+marker)
    marker='All five flags concern the same input'
    figure='''![Figure 7. Numerical coverage and digit-expression flags. The main sample has zero source-digit inputs among 100 queries; the separate targeted module has ten among ten. Each heatmap cell uses ten inputs. All five flags concern the same 24-hour to 12-hour time representation change, rather than five established factual errors.](figures/new_numerical_diagnostics.png)

'''
    body=body.replace(marker,figure+marker)
    # Number bibliography entries by first in-text appearance, including grouped citations.
    pattern=re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
    order=[]
    for match in pattern.finditer(body):
        for number in map(int,match.group(1).split(',')):
            assert number in refs,number
            if number not in order:order.append(number)
    assert set(order)==set(refs),'Every reference must be used in context'
    mapping={old:new for new,old in enumerate(order,1)}
    body=pattern.sub(lambda m:'['+','.join(str(mapping[int(n)])for n in m.group(1).split(','))+']',body)
    final=body+'\n## References\n\n'+'\n\n'.join(f'[{mapping[n]}] {refs[n]}'for n in order)+'\n'
    assert len(re.findall(r'^\[\d+\]',final.split('\n## References\n')[1],re.M))==25
    assert [int(n)for n in re.findall(r'!\[Figure (\d+)\.',final)]==list(range(1,8))
    PAPER.write_text(final)
    (ROOT/'editorial_update/citation_number_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print(json.dumps({'references':25,'figures':7,'all_references_cited':True,'citation_order':'first appearance'}))

if __name__=='__main__':main()
