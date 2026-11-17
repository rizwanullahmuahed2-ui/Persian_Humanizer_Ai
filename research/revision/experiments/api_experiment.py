"""Frozen paired OpenAI study. Default is dry run. Credentials never enter artifacts."""
from pathlib import Path
import argparse, csv, hashlib, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from decimal import Decimal
from harness import canonical, digest, filehash, readjson, writejson, require, load_corpus, load_vectors, PROMPT

MODELS=('gpt-4.1-mini-2025-04-14','gpt-4.1-nano-2025-04-14')
CONDITIONS=('LLM0','semantic1','semantic4','random4')
PRICES={MODELS[0]:{'input':Decimal('.4'),'output':Decimal('1.6')},MODELS[1]:{'input':Decimal('.1'),'output':Decimal('.4')}}
API='https://api.openai.com/v1/chat/completions'

def safe_key(envfile):
    key=os.environ.get('OPENAI_API_KEY','')
    if not key:
        for line in Path(envfile).read_text(encoding='utf-8').splitlines():
            match=re.match(r'^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*$',line)
            if match:key=match[1].strip().strip('"\'');break
    require(key and '\n' not in key and '\r' not in key,'OPENAI_API_KEY unavailable or invalid; do not put a key in artifacts')
    return key

def maximum_cost(payload):
    # One input token cannot contain fewer than one UTF-8 byte. The allowance
    # covers role/message framing; actual billing uses provider usage below.
    tokens=sum(len(m['content'].encode('utf-8'))+64 for m in payload['messages'])+256
    price=PRICES[payload['model']]
    return (Decimal(tokens)*price['input']+Decimal(payload['max_completion_tokens'])*price['output'])/Decimal(1000000)

def prepare(manifest,vectors,metadata,out,limit=100,seed=2026,budget='2.00',dataset=None,models=MODELS,max_attempts=None):
    require(set(models)<=set(MODELS) and len(models)==len(set(models)),'Only explicitly pinned study models accepted')
    require(limit>0 and seed>=0,'Positive query limit and nonnegative seed required')
    m,rows,proof=load_corpus(manifest,dataset);vecs,meta=load_vectors(vectors,metadata,rows)
    require(len(m['splits']['train'])>=4,'Need at least four training records')
    ordered=sorted(m['splits']['test'],key=lambda sid:digest([seed,sid]))
    representatives={}
    for sid in ordered:representatives.setdefault(rows[sid]['group_id'],sid)
    query_ids=list(representatives.values())[:limit]
    require(len(query_ids)==limit,'Query limit exceeds test population')
    jobs=[]
    for sid in query_ids:
        ranked=sorted(m['splits']['train'],key=lambda i:(-float(vecs[sid]@vecs[i]),i))
        sampled=random.Random(int(digest([seed,sid,'random4']),16)).sample(sorted(m['splits']['train']),4)
        for condition in CONDITIONS:
            chosen=[] if condition=='LLM0' else ranked[:1] if condition=='semantic1' else ranked[:4] if condition=='semantic4' else sampled
            examples=[{'stable_id':i,'ai':rows[i]['ai'],'human':rows[i]['human'],'cosine':float(vecs[sid]@vecs[i])}for i in chosen]
            text='\n\n'.join(f'ورودی نمونه {n}: {e["ai"]}\nبازنویسی نمونه {n}: {e["human"]}'for n,e in enumerate(examples,1)) if examples else '(بدون مثال)'
            prompt=PROMPT.format(examples=text,source=rows[sid]['ai'])
            for model in models:
                payload={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':0,'max_completion_tokens':256,'n':1}
                jobs.append({'job_id':digest([sid,condition,model,digest(payload)]),'query_id':sid,'group_id':rows[sid]['group_id'],'condition':condition,'model':model,'input':rows[sid]['ai'],'retrieved_ids':chosen,'retrieved_examples':examples,'payload':payload,'payload_sha256':digest(payload),'maximum_attempt_cost_usd':str(maximum_cost(payload))})
    random.Random(seed).shuffle(jobs)
    count=len(jobs);attempt_cap=max_attempts if max_attempts is not None else count
    require(attempt_cap>=count,'Attempt cap smaller than planned jobs')
    ceiling=Decimal(str(budget));require(ceiling>0,'Positive budget required')
    worst=sum(Decimal(j['maximum_attempt_cost_usd'])for j in jobs)
    require(worst<=ceiling,f'Conservative planned cost {worst} exceeds ceiling {ceiling}; lower limit or explicitly review budget')
    plan={'schema_version':1,'study':'Paired OpenAI controlled rewriting; no Ollama replication claimed','status':'prepared_not_executed','script_sha256':filehash(__file__),'harness_sha256':filehash(Path(__file__).with_name('harness.py')),'provenance':proof,'vectors':str(Path(vectors).resolve()),'vectors_sha256':filehash(vectors),'embedding_metadata':str(Path(metadata).resolve()),'embedding_metadata_sha256':filehash(metadata),'embedding_model':meta['model'],'embedding_revision':meta['revision'],'models':list(models),'conditions':list(CONDITIONS),'selection_seed':seed,'generation_seed':0,'query_count':len(query_ids),'query_ids':query_ids,'query_group_ids':[rows[sid]['group_id']for sid in query_ids],'selected_from_test_population':len(m['splits']['test']),'selection':'Seeded SHA256 order; first record per distinct connected group; generated outputs not inspected before this selection','budget_ceiling_usd':str(ceiling),'max_http_attempts':attempt_cap,'conservative_planned_maximum_usd':str(worst),'prices_usd_per_million_tokens':{k:{n:str(v)for n,v in p.items()}for k,p in PRICES.items()},'jobs':jobs,'primary_preservation_reference':'original input, not automatically trusted human column','human_ratings_created':0,'generator_requests_made_at_preparation':0}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);require(not(out/'plan.json').exists(),'Output plan already exists; use existing frozen run or another directory')
    writejson(out/'plan.json',plan);(out/'plan.sha256').write_text(filehash(out/'plan.json')+'\n')
    return {'status':'prepared_not_executed','jobs':count,'queries':len(query_ids),'cost_upperbound_usd':str(worst),'ceiling_usd':str(ceiling)}

def verify(directory):
    d=Path(directory);p=readjson(d/'plan.json')
    require(filehash(d/'plan.json')==(d/'plan.sha256').read_text().strip(),'Plan seal mismatch')
    require(filehash(__file__)==p['script_sha256'],'API runner changed since freeze')
    require(filehash(Path(__file__).with_name('harness.py'))==p['harness_sha256'],'Shared harness changed since freeze')
    proof=p['provenance'];load_corpus(proof['manifest_path'],proof['dataset_path'])
    for k in ('manifest','dataset','records'):require(filehash(proof[k+'_path'])==proof[k+'_sha256'],'Frozen input changed: '+k)
    require(filehash(p['vectors'])==p['vectors_sha256'] and filehash(p['embedding_metadata'])==p['embedding_metadata_sha256'],'Embedding artifact changed')
    require(len(p['jobs'])<=p['max_http_attempts'],'Job count exceeds cap')
    for job in p['jobs']:
        require(job['payload_sha256']==digest(job['payload']),'Payload hash mismatch')
        require(job['model']==job['payload']['model'] and job['model'] in p['models'],'Model mismatch')
        require(job['condition'] in p['conditions'],'Condition mismatch')
        require(len(job['retrieved_ids'])=={'LLM0':0,'semantic1':1,'semantic4':4,'random4':4}[job['condition']],'Incorrect effective demonstration count')
    return p

def parse_success(raw,job):
    value=json.loads(raw);require(value.get('model')==job['model'],'Provider returned another model snapshot')
    choices=value.get('choices',[]);require(len(choices)==1,'Unexpected number of choices')
    require(choices[0].get('finish_reason')=='stop','Non-normal finish reason')
    text=choices[0].get('message',{}).get('content');require(isinstance(text,str)and text.strip(),'Empty output')
    usage=value.get('usage',{});require(all(isinstance(usage.get(k),int)and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')),'Usage missing')
    require(usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens'],'Usage inconsistent')
    require(usage['completion_tokens']<=job['payload']['max_completion_tokens'],'Completion token cap exceeded')
    price=PRICES[job['model']];cost=(Decimal(usage['prompt_tokens'])*price['input']+Decimal(usage['completion_tokens'])*price['output'])/Decimal(1000000)
    return {'output':text,'usage':usage,'finish_reason':choices[0]['finish_reason'],'response_id':value.get('id'),'system_fingerprint':value.get('system_fingerprint'),'returned_model':value['model'],'uncached_cost_usd':str(cost)}

def run(directory,envfile='/workspace/.env',execute=False,concurrency=4,max_retries=0):
    d=Path(directory);plan=verify(d);plan_sha=filehash(d/'plan.json')
    require(1<=concurrency<=4 and 0<=max_retries<=2,'Concurrency <=4 and retries <=2 required')
    if not execute:return {'status':'dry_run','queries':plan['query_count'],'jobs':len(plan['jobs']),'generator_calls':0,'conservative_cost_usd':plan['conservative_planned_maximum_usd']}
    key=safe_key(envfile);responses=d/'responses';responses.mkdir(exist_ok=True)
    ledger=d/'attempts.jsonl';history=[json.loads(x)for x in ledger.read_text().splitlines()]if ledger.exists()else[]
    lock=threading.Lock();budget=Decimal(plan['budget_ceiling_usd']);reserved=sum(Decimal(r['reserved_maximum_usd'])for r in history)
    require(len(history)<=plan['max_http_attempts'] and reserved<=budget,'Stored attempt/budget cap exceeded')
    pending=[];history_ids={x['job_id']for x in history}
    for job in plan['jobs']:
        rp=responses/(job['job_id']+'.json');rawp=responses/(job['job_id']+'.raw.json')
        if rp.exists():
            saved=readjson(rp)
            require(saved.get('plan_sha256')==plan_sha and saved.get('payload_sha256')==job['payload_sha256'],'Cached response does not match frozen request')
            require(saved.get('status')=='complete' and rawp.exists() and saved.get('raw_sha256')==filehash(rawp),'Cached error/incomplete/corrupt output: create a separately documented recovery run')
            parsed=parse_success(rawp.read_bytes(),job);require(parsed['output']==saved.get('output'),'Cached output differs from raw provider response')
            continue
        require(job['job_id']not in history_ids and not rawp.exists(),'Unresolved/orphan attempt; no silent duplicate generation')
        pending.append(job)
    def task(job):
        nonlocal reserved
        base={'job_id':job['job_id'],'query_id':job['query_id'],'group_id':job['group_id'],'model':job['model'],'condition':job['condition'],'plan_sha256':plan_sha,'payload_sha256':job['payload_sha256'],'retrieved_ids':job['retrieved_ids']}
        rp=responses/(job['job_id']+'.json');rawp=responses/(job['job_id']+'.raw.json')
        for retry in range(max_retries+1):
            # Check every mutation-sensitive frozen input before a request.
            require(filehash(d/'plan.json')==plan_sha,'Plan changed during execution')
            with lock:
                amount=Decimal(job['maximum_attempt_cost_usd'])
                require(len(history)<plan['max_http_attempts'],'HTTP attempt hard cap reached')
                require(reserved+amount<=budget,'Conservative budget ceiling reached')
                event={'job_id':job['job_id'],'retry_index':retry,'payload_sha256':job['payload_sha256'],'reserved_maximum_usd':str(amount),'started_unix':time.time()}
                with ledger.open('a',encoding='utf-8')as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
                history.append(event);reserved+=amount
            start=time.perf_counter();status=0;raw=None;request_id=None
            request=Request(API,data=canonical(job['payload']),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            try:
                with urlopen(request,timeout=120)as response:
                    status=response.status;request_id=response.headers.get('x-request-id');raw=response.read()
                rawp.write_bytes(raw);record={**base,'status':'complete','http_status':status,'request_id':request_id,'wall_seconds':time.perf_counter()-start,'raw_sha256':filehash(rawp),**parse_success(raw,job)}
                require(Decimal(record['uncached_cost_usd'])<=amount,'Actual usage cost exceeded reservation; stop and inspect')
                writejson(rp,record);return record
            except HTTPError as e:
                status=e.code;request_id=e.headers.get('x-request-id');e.close()
                # Never write or echo error bodies, which may contain credential details.
                error_type='provider_http_error'
            except (URLError,TimeoutError):error_type='network_or_timeout'
            except Exception as e:error_type='invalid_provider_response_'+type(e).__name__
            record={**base,'status':'error','http_status':status,'request_id':request_id,'wall_seconds':time.perf_counter()-start,'error_type':error_type,'retry_index':retry,'raw_sha256':filehash(rawp)if rawp.exists()else None}
            writejson(responses/(job['job_id']+f'.attempt{retry}.error.json'),record)
            if retry<max_retries and (status==429 or 500<=status<600):time.sleep(min(4,2**retry));continue
            writejson(rp,record);return record
    results=[]
    with ThreadPoolExecutor(max_workers=concurrency)as pool:
        futures={pool.submit(task,j):j for j in pending}
        for f in as_completed(futures):
            try:r=f.result();results.append(r)
            except Exception as e:
                # Bound-check failures are deliberately safe and contain no key.
                for item in futures:item.cancel()
                raise ValueError('Execution stopped by invariant/budget check; inspect frozen artifacts')from None
            print(json.dumps({'completed_this_invocation':len(results),'planned':len(pending),'condition':r['condition'],'model':r['model'],'status':r['status']}),flush=True)
    all_records=[readjson(responses/(j['job_id']+'.json'))for j in plan['jobs']]
    ok=[r for r in all_records if r['status']=='complete'];errors=[r for r in all_records if r['status']!='complete']
    receipt={'status':'complete'if not errors else 'completed_with_errors','planned_jobs':len(plan['jobs']),'complete_responses':len(ok),'errors':len(errors),'new_http_attempts_this_invocation':sum(1 for _ in results),'http_attempts_total':len(history),'conservative_reserved_usd':str(reserved),'actual_success_cost_uncached_usd':str(sum(Decimal(r['uncached_cost_usd'])for r in ok)),'budget_ceiling_usd':str(budget),'plan_sha256':plan_sha,'human_ratings_created':0}
    writejson(d/'completion.json',receipt);return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',required=True);p.add_argument('--vectors',required=True);p.add_argument('--embedding-metadata',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=100);p.add_argument('--seed',type=int,default=2026);p.add_argument('--budget',default='2.00');p.add_argument('--dataset');p.add_argument('--max-attempts',type=int)
    p=sub.add_parser('run');p.add_argument('--directory',required=True);p.add_argument('--env-file',default='/workspace/.env');p.add_argument('--execute',action='store_true');p.add_argument('--concurrency',type=int,default=4);p.add_argument('--max-retries',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.manifest,args.vectors,args.embedding_metadata,args.out,args.limit,args.seed,args.budget,args.dataset,max_attempts=args.max_attempts)
    else:result=run(args.directory,args.env_file,args.execute,args.concurrency,args.max_retries)
    print(json.dumps(result,ensure_ascii=False,indent=2))
