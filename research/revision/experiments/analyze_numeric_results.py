"""Verify the separate targeted numeral module; no API calls or human judgments."""
from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal
import argparse,json
from analyze_api_results import (HERE,filehash,readjson,readcsv,require,load_and_verify_inputs,
                                 read_outputs,numeric_tokens,write_csv,summarize)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--artifact-root',type=Path)
    parser.add_argument('--main-diagnostics',type=Path,
                        default=HERE/'combined_analysis/response_diagnostics.csv')
    args=parser.parse_args();directory=args.directory.resolve()
    out=args.out or directory/'analysis';out.mkdir(parents=True,exist_ok=True)
    spec_path=HERE/'NUMERIC_DIAGNOSTIC_SPEC.json';registered=readjson(spec_path)
    require(filehash(spec_path)==spec_path.with_suffix('.sha256').read_text().strip(),'Diagnostic spec seal mismatch')
    plan,spec,records=load_and_verify_inputs(directory,False,args.artifact_root,True)
    rows,errors=read_outputs(directory,plan,spec,records)
    require(not errors and len(rows)==registered['expected_jobs'],
            'Numeric module incomplete; do not summarize partial outcomes')
    require(len({r['query_id']for r in rows})==registered['expected_queries']and
            all(r['source_has_digit_expressions']for r in rows),'Wrong numeric cohort')
    if args.main_diagnostics.is_file():
        main_rows=readcsv(args.main_diagnostics)
        require(not({r['query_id']for r in rows}&{r['query_id']for r in main_rows}),
                'Targeted module overlaps the original random sample unexpectedly')
        require(not({r['response_id']for r in rows}&{r['response_id']for r in main_rows}),
                'Response identifier reused between studies')
    flagged=[]
    for row in rows:
        source_numbers=numeric_tokens(row['source']);output_numbers=numeric_tokens(row['output'])
        row['digit_expressions_added']=json.dumps(output_numbers-source_numbers,sort_keys=True,ensure_ascii=False)
        row['digit_expressions_deleted']=json.dumps(source_numbers-output_numbers,sort_keys=True,ensure_ascii=False)
        if row['digit_multiset_differs']:
            flagged.append({'job_id':row['job_id'],'query_id':row['query_id'],'group_id':row['group_id'],
                            'model':row['model'],'condition':row['condition'],'source':row['source'],
                            'output':row['output'],'source_digit_expressions':row['source_digit_expressions'],
                            'output_digit_expressions':row['output_digit_expressions'],
                            'digit_expressions_added':row['digit_expressions_added'],
                            'digit_expressions_deleted':row['digit_expressions_deleted'],
                            'interpretation':'Token mismatch requires inspection; not an adjudicated factual error.'})
    attempts=[json.loads(line)for line in(directory/'attempts.jsonl').read_text().splitlines()]
    require(all(a['job_id']in{j['job_id']for j in plan['jobs']}for a in attempts),'Unknown diagnostic attempt')
    result={'status':'complete','scope':registered['module'],'selection':registered['selection'],
            'analysis_spec_sha256':filehash(spec_path),'plan_sha256':filehash(directory/'plan.json'),
            'analyzer_sha256':filehash(__file__),'shared_independent_verifier_sha256':filehash(HERE/'analyze_api_results.py'),
            'queries':len(plan['query_ids']),'independent_source_groups':len(set(plan['query_group_ids'])),
            'validated_responses':len(rows),'normal_stop_responses':len(rows),
            'fresh_provider_response_ids':len({r['response_id']for r in rows}),
            'http_attempts_total':len(attempts),'recorded_errors':len(errors),
            'prompt_tokens':sum(r['prompt_tokens']for r in rows),
            'completion_tokens':sum(r['completion_tokens']for r in rows),
            'uncached_cost_usd_not_invoice':str(sum(Decimal(r['uncached_cost_usd'])for r in rows)),
            'summaries':summarize(rows,spec),'flagged_cases':flagged,
            'flagged_response_count':len(flagged),'flagged_distinct_queries':len({r['query_id']for r in flagged}),
            'human_ratings_created':0,'pooled_with_main_sample':False,
            'interpretation':registered['interpretation']}
    write_csv(out/'response_diagnostics.csv',rows)
    if flagged:write_csv(out/'numeric_flags.csv',flagged)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('summaries','flagged_cases')},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
