"""Independently verify and describe saved OpenAI responses; no network/client imports.

No generator or harness code is imported. Scores are automatic diagnostics and
must not be presented as human naturalness or semantic-accuracy measurements.
"""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata

HERE = Path(__file__).resolve().parent
SPEC = HERE / 'ANALYSIS_SPEC.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def filehash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def readjson(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def artifact_path(value, artifact_root=None):
    """Relocate frozen workspace paths without rewriting sealed artifacts."""
    path=Path(value)
    if artifact_root is not None and path.is_absolute():
        try:
            relative=path.relative_to('/workspace')
        except ValueError:
            raise ValueError('Frozen absolute path outside expected workspace prefix')from None
        return Path(artifact_root).resolve()/relative
    return path


def normalize(text):
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans({
        'ي': 'ی', 'ى': 'ی', 'ك': 'ک', '\u200c': ' ', 'ـ': ''})).casefold()
    text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                   for c in text if not unicodedata.combining(c))
    return ' '.join(text.split())


def tokens(text):
    return re.findall(r'[^\W_]+', normalize(text), flags=re.UNICODE)


def numeric_tokens(text):
    return Counter(re.findall(r'\d+(?:[٫.,:/]\d+)*', normalize(text)))


def ngrams(sequence, n):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


def overlap_f1(a, b):
    if not a or not b:
        return 0.0
    return 2 * sum((Counter(a) & Counter(b)).values()) / (len(a) + len(b))


def lcs_length(a, b):
    previous = [0] * (len(b)+1)
    for first in a:
        current = [0]
        for j, second in enumerate(b, 1):
            current.append(previous[j-1]+1 if first == second else
                           max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge(a, b):
    return {'rouge1_f1': overlap_f1(a, b),
            'rouge2_f1': overlap_f1(ngrams(a, 2), ngrams(b, 2)),
            'rougeL_f1': 2*lcs_length(a, b)/(len(a)+len(b)) if a and b else 0.0}


def descriptions(values):
    values = [x for x in values if x is not None]
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def load_and_verify_inputs(directory, secondary=False, artifact_root=None, numeric=False):
    import numpy as np
    directory = Path(directory).resolve()
    spec = readjson(SPEC)
    require(filehash(SPEC) == SPEC.with_suffix('.sha256').read_text().strip(),
            'Registered analysis specification changed')
    if secondary or numeric:
        amendment_path=HERE/'SPEC_AMENDMENT.json'
        amendment=readjson(amendment_path)
        require(filehash(amendment_path)==amendment_path.with_suffix('.sha256').read_text().strip(),
                'Availability amendment changed')
        require(amendment['parent_spec_sha256']==filehash(SPEC),'Amendment parent mismatch')
        spec={**spec,'models':[amendment['combined_success_models'][1]],
              'expected_jobs':amendment['expected_secondary_jobs'],
              'prices_usd_per_million_tokens_uncached':{
                  **spec['prices_usd_per_million_tokens_uncached'],
                  **amendment['prices_usd_per_million_tokens_uncached_extension']}}
    if numeric:
        numeric_path=HERE/'NUMERIC_DIAGNOSTIC_SPEC.json';numeric_spec=readjson(numeric_path)
        require(filehash(numeric_path)==numeric_path.with_suffix('.sha256').read_text().strip(),
                'Numeric diagnostic specification changed')
        require(numeric_spec['parent_spec_sha256']==filehash(SPEC)and
                numeric_spec['availability_amendment_sha256']==filehash(HERE/'SPEC_AMENDMENT.json'),
                'Numeric diagnostic specification parent mismatch')
        spec={**spec,'models':numeric_spec['models'],
              'expected_jobs':numeric_spec['expected_jobs'],
              'expected_queries':numeric_spec['expected_queries']}
    plan_path = directory/'plan.json'
    plan = readjson(plan_path)
    require(filehash(plan_path) == (directory/'plan.sha256').read_text().strip(),
            'Frozen generation plan changed')
    runner=HERE/('api_numeric_diagnostic.py'if numeric else
                 'api_secondary_experiment.py' if secondary else 'api_experiment.py')
    require(filehash(runner) == plan['script_sha256'],
            'Generation script changed after plan freeze')
    require(filehash(HERE/'harness.py') == plan['harness_sha256'],
            'Shared harness changed after plan freeze')
    require(set(plan['models']) == set(spec['models']) and
            set(plan['conditions']) == set(spec['conditions']), 'Unexpected study matrix')
    require(plan['query_count'] == spec['expected_queries'] and
            len(plan['jobs']) == spec['expected_jobs'], 'Unexpected study dimensions')
    proof = plan['provenance']
    for name in ('manifest', 'dataset', 'records'):
        require(filehash(artifact_path(proof[name+'_path'],artifact_root)) == proof[name+'_sha256'],
                'Frozen source changed: '+name)
    split = readjson(artifact_path(proof['manifest_path'],artifact_root))
    require(filehash(artifact_path(proof['dataset_path'],artifact_root)) == split['dataset_sha256'] and
            filehash(artifact_path(proof['records_path'],artifact_root)) == split['records_sha256'],
            'Split is not linked to frozen source')
    records = readcsv(artifact_path(proof['records_path'],artifact_root))
    by_id = {r['stable_id']: r for r in records}
    require(len(by_id) == len(records), 'Duplicate source IDs')
    all_ids = [i for part in split['splits'].values() for i in part]
    require(len(all_ids) == len(set(all_ids)) and set(all_ids) == set(by_id),
            'Split ID overlap or incomplete source coverage')
    seen = {}
    for part, ids in split['splits'].items():
        for sid in ids:
            row = by_id[sid]
            for text in (normalize(row['ai']), normalize(row['human'])):
                require(text not in seen or seen[text] == part,
                        'Normalized source/reference text leaked across partitions')
                seen[text] = part
    require(len(plan['query_ids']) == len(set(plan['query_ids'])) == spec['expected_queries'],
            'Queries are not distinct')
    require(all(i in split['splits']['test'] for i in plan['query_ids']),
            'Non-test query found')
    query_groups = [by_id[i]['group_id'] for i in plan['query_ids']]
    require(len(query_groups) == len(set(query_groups)) and
            query_groups == plan['query_group_ids'], 'Query groups are not distinct/aligned')
    eligible=split['splits']['test']
    if numeric:
        eligible=[sid for sid in eligible if re.search(r'\d',by_id[sid]['ai'])]
        require(len(eligible)==spec['expected_queries'],'Source-digit cohort changed')
    ordered = sorted(eligible, key=lambda sid: digest([plan['selection_seed'], sid]))
    representatives = {}
    for sid in ordered:
        representatives.setdefault(by_id[sid]['group_id'], sid)
    require(list(representatives.values())[:plan['query_count']] == plan['query_ids'],
            'Seeded query selection cannot be reconstructed')
    vector_path=artifact_path(plan['vectors'],artifact_root)
    metadata_path=artifact_path(plan['embedding_metadata'],artifact_root)
    require(filehash(vector_path) == plan['vectors_sha256'] and
            filehash(metadata_path) == plan['embedding_metadata_sha256'],
            'Embedding artifacts changed')
    metadata = readjson(metadata_path)
    require(metadata['model'] == plan['embedding_model'] and
            metadata['revision'] == plan['embedding_revision'], 'Embedding model mismatch')
    require(metadata['vectors_sha256'] == filehash(vector_path), 'Vector seal mismatch')
    with np.load(vector_path, allow_pickle=False) as source:
        vector_ids = source['ids'].tolist()
        matrix = source['vectors'].astype('float64')
    require(len(vector_ids) == len(set(vector_ids)) and set(vector_ids) == set(by_id),
            'Embedding IDs do not match records')
    require(matrix.ndim == 2 and matrix.shape[0] == len(vector_ids) and
            np.isfinite(matrix).all(), 'Invalid embedding matrix')
    norms = np.linalg.norm(matrix, axis=1)
    require((norms > 0).all(), 'Zero embedding vector')
    vectors = {sid: matrix[i]/norms[i] for i, sid in enumerate(vector_ids)}
    expected_text_hashes = {sid: hashlib.sha256(by_id[sid]['ai'].encode()).hexdigest()
                            for sid in vector_ids}
    require(expected_text_hashes == metadata['text_sha256'],
            'Embeddings are not linked to source-only text')
    ranks = {}
    random_examples = {}
    for sid in plan['query_ids']:
        ranks[sid] = sorted(split['splits']['train'],
                            key=lambda i: (-float(vectors[sid] @ vectors[i]), i))
        rng = random.Random(int(digest([plan['selection_seed'], sid, 'random4']), 16))
        random_examples[sid] = rng.sample(sorted(split['splits']['train']), 4)
    cells = Counter()
    for job in plan['jobs']:
        sid = job['query_id']; condition = job['condition']; model = job['model']
        require(job['payload_sha256'] == digest(job['payload']), 'Payload hash mismatch')
        require(job['job_id'] == digest([sid, condition, model, digest(job['payload'])]),
                'Job identifier cannot be reconstructed')
        require(sid in plan['query_ids'] and model in plan['models'] and
                condition in plan['conditions'], 'Job not in declared matrix')
        require(job['group_id'] == by_id[sid]['group_id'] and
                job['input'] == by_id[sid]['ai'], 'Query record not aligned')
        expected = [] if condition == 'LLM0' else ranks[sid][:1] if condition == 'semantic1' else ranks[sid][:4] if condition == 'semantic4' else random_examples[sid]
        require(job['retrieved_ids'] == expected, 'Demonstration selection not reproducible')
        require(len(job['retrieved_examples']) == len(expected), 'Demonstration count mismatch')
        for example, rid in zip(job['retrieved_examples'], expected):
            require(example['stable_id'] == rid and example['ai'] == by_id[rid]['ai'] and
                    example['human'] == by_id[rid]['human'], 'Demonstration text changed')
            require(math.isclose(example['cosine'], float(vectors[sid]@vectors[rid]),
                                 rel_tol=1e-9, abs_tol=1e-9), 'Similarity changed')
        payload = job['payload']
        require(payload['model'] == model and payload['temperature'] == 0 and
                payload['seed'] == 0 and payload['max_completion_tokens'] == 256 and
                payload['n'] == 1, 'Undeclared generation settings')
        require(len(payload['messages']) == 1 and payload['messages'][0]['role'] == 'user',
                'Undeclared messages')
        require(payload['messages'][0]['content'].count('ورودی نمونه ') == len(expected),
                'Effective demonstration count differs from declared condition')
        cells[(sid, model, condition)] += 1
    require(len(cells) == spec['expected_jobs'] and set(cells.values()) == {1},
            'Study matrix has duplicate or missing cells')
    for sid in plan['query_ids']:
        for condition in plan['conditions']:
            prompts = {j['payload']['messages'][0]['content'] for j in plan['jobs']
                       if j['query_id'] == sid and j['condition'] == condition}
            require(len(prompts) == 1, 'Prompt differs between model snapshots')
    return plan, spec, by_id


def read_outputs(directory, plan, spec, records):
    directory = Path(directory)
    output = []
    errors = []
    response_ids = []
    plan_sha = filehash(directory/'plan.json')
    for job in plan['jobs']:
        path = directory/'responses'/(job['job_id']+'.json')
        rawpath = directory/'responses'/(job['job_id']+'.raw.json')
        if not path.is_file():
            errors.append({'job_id': job['job_id'], 'problem': 'missing_parsed_record'})
            continue
        saved = readjson(path)
        if saved.get('status') != 'complete':
            errors.append({'job_id': job['job_id'], 'problem': 'recorded_error',
                           'http_status': saved.get('http_status'),
                           'error_type': saved.get('error_type')})
            continue
        require(rawpath.is_file() and filehash(rawpath) == saved['raw_sha256'],
                'Raw response missing or changed')
        require(saved['plan_sha256'] == plan_sha and
                saved['payload_sha256'] == job['payload_sha256'], 'Saved result changed request')
        require(all(saved[k] == job[k] for k in ('job_id','query_id','group_id','model','condition','retrieved_ids')),
                'Saved record metadata not aligned')
        raw = readjson(rawpath)
        require(raw['model'] == job['model'] == saved['returned_model'],
                'Returned model differs from frozen snapshot')
        require(len(raw['choices']) == 1 and raw['choices'][0]['finish_reason'] == 'stop',
                'Non-normal model termination')
        text = raw['choices'][0]['message']['content']
        require(isinstance(text, str) and text.strip() and text == saved['output'],
                'Empty or altered model output')
        usage = raw['usage']
        require(usage == saved['usage'] and all(type(usage.get(k)) is int and usage[k] >= 0
                for k in ('prompt_tokens','completion_tokens','total_tokens')), 'Usage changed or invalid')
        require(usage['total_tokens'] == usage['prompt_tokens']+usage['completion_tokens'],
                'Usage total inconsistent')
        require(usage['completion_tokens'] <= job['payload']['max_completion_tokens'],
                'Output exceeds frozen token cap')
        require(raw.get('id') and raw['id'] == saved['response_id'], 'Response identifier missing or changed')
        response_ids.append(raw['id'])
        price = spec['prices_usd_per_million_tokens_uncached'][job['model']]
        cost = (Decimal(usage['prompt_tokens'])*Decimal(price['input']) +
                Decimal(usage['completion_tokens'])*Decimal(price['output']))/Decimal(1000000)
        require(cost == Decimal(saved['uncached_cost_usd']), 'Recorded usage cost changed')
        source = job['input']; reference = records[job['query_id']]['human']
        at, rt, ot = tokens(source), tokens(reference), tokens(text)
        an, on = numeric_tokens(source), numeric_tokens(text)
        row = {'job_id': job['job_id'], 'query_id': job['query_id'],
               'group_id': job['group_id'], 'model': job['model'], 'condition': job['condition'],
               'source': source, 'unvalidated_reference': reference, 'output': text,
               'normalized_exact_copy': int(normalize(source) == normalize(text)),
               'source_tokens': len(at), 'reference_tokens': len(rt), 'output_tokens': len(ot),
               'output_source_token_ratio': len(ot)/len(at) if at else None,
               **{'source_'+k: v for k,v in rouge(ot, at).items()},
               **{'reference_'+k: v for k,v in rouge(ot, rt).items()},
               'source_has_digit_expressions': int(bool(an)),
               'either_has_digit_expressions': int(bool(an or on)),
               'digit_multiset_differs': int(an != on),
               'source_digit_expressions': json.dumps(an, ensure_ascii=False, sort_keys=True),
               'output_digit_expressions': json.dumps(on, ensure_ascii=False, sort_keys=True),
               'retrieved_ids': '|'.join(job['retrieved_ids']),
               'effective_examples': len(job['retrieved_ids']),
               'response_id': raw['id'], 'finish_reason': 'stop',
               'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
               'total_tokens': usage['total_tokens'],
               'cached_prompt_tokens': usage.get('prompt_tokens_details',{}).get('cached_tokens',0),
               'uncached_cost_usd': str(cost), 'wall_seconds': saved['wall_seconds'],
               'raw_sha256': saved['raw_sha256']}
        output.append(row)
    require(len(response_ids) == len(set(response_ids)),
            'Provider response ID reused; responses are not independent fresh calls')
    return output, errors


def summarize(rows, spec):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'],row['condition'])].append(row)
    summaries = []
    for (model,condition), subset in sorted(groups.items()):
        numeric_eligible = [r for r in subset if r['either_has_digit_expressions']]
        source_numeric = [r for r in subset if r['source_has_digit_expressions']]
        result = {'model':model,'condition':condition,'n_responses':len(subset),
                  'n_query_groups':len({r['group_id']for r in subset}),
                  'exact_copies':sum(r['normalized_exact_copy']for r in subset),
                  'exact_copy_fraction':statistics.mean(r['normalized_exact_copy']for r in subset),
                  'any_side_digit_eligible':len(numeric_eligible),
                  'digit_multiset_flags':sum(r['digit_multiset_differs']for r in numeric_eligible),
                  'digit_multiset_flag_rate_among_any_side_eligible':statistics.mean(r['digit_multiset_differs']for r in numeric_eligible)if numeric_eligible else None,
                  'source_digit_eligible':len(source_numeric),
                  'digit_multiset_flags_among_source_digit_eligible':sum(r['digit_multiset_differs']for r in source_numeric),
                  'prompt_tokens_total':sum(r['prompt_tokens']for r in subset),
                  'completion_tokens_total':sum(r['completion_tokens']for r in subset),
                  'cached_prompt_tokens_total':sum(r['cached_prompt_tokens']for r in subset),
                  'uncached_cost_usd':str(sum(Decimal(r['uncached_cost_usd'])for r in subset))}
        for metric in ['source_rouge1_f1','source_rouge2_f1','source_rougeL_f1',
                       'reference_rouge1_f1','reference_rouge2_f1','reference_rougeL_f1',
                       'output_source_token_ratio','prompt_tokens','completion_tokens','wall_seconds']:
            result[metric]=descriptions([r[metric]for r in subset])
        summaries.append(result)
    return summaries


def bootstrap(rows, spec):
    import numpy as np
    outputs=[]
    for model in spec['models']:
        cells={(r['query_id'],r['condition']):r for r in rows if r['model']==model}
        for baseline in ('LLM0','random4'):
            query_ids=sorted({r['query_id']for r in rows if r['model']==model and
                              (r['query_id'],'semantic4')in cells and(r['query_id'],baseline)in cells})
            require(len(query_ids)==spec['expected_queries'],'Paired comparison missing queries')
            for metric in spec['contrast_metrics']:
                differences=np.array([cells[(q,'semantic4')][metric]-cells[(q,baseline)][metric]
                                      for q in query_ids],dtype=float)
                require(np.isfinite(differences).all(),'Undefined paired metric')
                # One generator per contrast, same resampling indices for all metrics.
                rng=np.random.default_rng(spec['bootstrap']['seed'])
                indices=rng.integers(0,len(differences),size=(spec['bootstrap']['replicates'],len(differences)))
                estimates=differences[indices].mean(axis=1)
                lo,hi=np.percentile(estimates,[2.5,97.5],method='linear')
                outputs.append({'model':model,'contrast':'semantic4 minus '+baseline,'metric':metric,
                                'n_distinct_query_groups':len(query_ids),'mean_paired_difference':float(differences.mean()),
                                'percentile_95_ci_low':float(lo),'percentile_95_ci_high':float(hi),
                                'bootstrap_replicates':spec['bootstrap']['replicates'],
                                'interpretation':'Automatic diagnostic difference; not human quality or semantic accuracy.'})
    return outputs


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--secondary-directory',type=Path)
    parser.add_argument('--artifact-root',type=Path,
                        help='Extracted package root replacing the original /workspace prefix; sealed files are not edited')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();directory=args.directory.resolve();out=args.out or directory/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    plan,spec,records=load_and_verify_inputs(directory,False,args.artifact_root)
    rows,errors=read_outputs(directory,plan,spec,records)
    all_errors=errors
    unexpected_errors=errors
    expected_jobs=len(plan['jobs'])
    scientific_attempts=0
    for line in (directory/'attempts.jsonl').read_text().splitlines():
        event=json.loads(line)
        require(event['job_id']in{j['job_id']for j in plan['jobs']},'Unknown original attempt')
        scientific_attempts+=1
    additional_integrity={}
    if args.secondary_directory:
        amendment_path=HERE/'SPEC_AMENDMENT.json';amendment=readjson(amendment_path)
        secondary=args.secondary_directory.resolve()
        secondary_plan,secondary_spec,secondary_records=load_and_verify_inputs(secondary,True,args.artifact_root)
        require(secondary_plan['query_ids']==plan['query_ids'] and
                secondary_plan['query_group_ids']==plan['query_group_ids'],
                'Secondary study changed query/group selection')
        primary_prompts={(j['query_id'],j['condition']):j['payload']['messages']
                         for j in plan['jobs']if j['model']==amendment['combined_success_models'][0]}
        for job in secondary_plan['jobs']:
            require(job['payload']['messages']==primary_prompts[(job['query_id'],job['condition'])],
                    'Secondary study changed matched prompt')
            require(job['payload']['temperature']==0 and job['payload']['seed']==0 and
                    job['payload']['max_completion_tokens']==256,'Secondary settings changed')
        secondary_rows,secondary_errors=read_outputs(secondary,secondary_plan,secondary_spec,secondary_records)
        unavailable_jobs={j['job_id']for j in plan['jobs']if j['model']==amendment['retained_failed_model']}
        expected_errors=[e for e in errors if e['job_id']in unavailable_jobs and
                         e.get('problem')=='recorded_error'and e.get('http_status')==404]
        unexpected_errors=[e for e in errors if e not in expected_errors]+secondary_errors
        require(len(expected_errors)==amendment['expected_original_nano_http404'],
                'Unexpected unavailable-model error count')
        require(all(r['model']==amendment['combined_success_models'][0]for r in rows),
                'Original success set differs from availability amendment')
        rows+=secondary_rows;all_errors=errors+secondary_errors
        spec={**spec,'models':amendment['combined_success_models'],
              'prices_usd_per_million_tokens_uncached':secondary_spec['prices_usd_per_million_tokens_uncached']}
        for line in (secondary/'attempts.jsonl').read_text().splitlines():
            event=json.loads(line)
            require(event['job_id']in{j['job_id']for j in secondary_plan['jobs']},'Unknown secondary attempt')
            scientific_attempts+=1
        expected_jobs+=len(secondary_plan['jobs'])
        additional_integrity={'secondary_plan_sha256':filehash(secondary/'plan.json'),
                              'secondary_directory':str(secondary),
                              'availability_amendment_sha256':filehash(amendment_path),
                              'expected_unavailable_model_http404':len(expected_errors)}
    readiness={'status':'complete'if not unexpected_errors else 'incomplete','validated_responses':len(rows),
               'planned_jobs_all_archives':expected_jobs,'scientific_http_attempts':scientific_attempts,
               'recorded_error_count':len(all_errors),'unexpected_error_count':len(unexpected_errors),
               'all_recorded_errors_or_missing':all_errors,'unexpected_errors':unexpected_errors,
               'analysis_spec_sha256':filehash(SPEC),'plan_sha256':filehash(directory/'plan.json'),
               'analyzer_sha256':filehash(__file__),'human_ratings_created':0,**additional_integrity}
    (out/'READINESS.json').write_text(json.dumps(readiness,indent=2)+'\n')
    require(not unexpected_errors and len(rows)==spec['expected_jobs'],
            'Generation study incomplete; no full-results summary generated. See READINESS.json')
    summaries=summarize(rows,spec);contrasts=bootstrap(rows,spec)
    require(len({r['response_id']for r in rows})==len(rows),'Provider response IDs not unique')
    result={**readiness,'interpretation':spec['scope'],'query_groups':len({r['group_id']for r in rows}),
            'fresh_provider_response_ids':len({r['response_id']for r in rows}),
            'distinct_output_strings':len({r['output']for r in rows}),
            'normal_stop_responses':len(rows),'prompt_tokens':sum(r['prompt_tokens']for r in rows),
            'completion_tokens':sum(r['completion_tokens']for r in rows),
            'total_tokens':sum(r['total_tokens']for r in rows),
            'cached_prompt_tokens':sum(r['cached_prompt_tokens']for r in rows),
            'uncached_cost_usd_not_invoice':str(sum(Decimal(r['uncached_cost_usd'])for r in rows)),
            'summaries':summaries,'paired_contrasts':contrasts,
            'warnings':['Associated references are not validated gold for within-register naturalization.',
                        'Number-token flags are not an error rate; number words and equivalent representations require inspection.',
                        'Two small models from one provider do not establish broad cross-model or cross-provider generalization.',
                        'No human naturalness ratings or factual-preservation accuracy are computed.']}
    write_csv(out/'response_diagnostics.csv',rows)
    write_csv(out/'paired_contrasts.csv',contrasts)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in
                     ('summaries','paired_contrasts','all_recorded_errors_or_missing','unexpected_errors')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
