"""Reproducible corpus audit: Python standard library; no generator or network.

All overlap scores compare the CSV ai and human cells, NOT system outputs.
Numbers and lexical overlap are diagnostics, not meaning or human quality scores.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, csv, hashlib, json, math, os, re, statistics, unicodedata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]/'Persian_Humanizer_Ai'
FULL = REPO/'data/persian_humanizer_dataset.csv'
TRAIN = REPO/'data/train_set.csv'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(text):
    text=unicodedata.normalize('NFKC',text).translate(str.maketrans({'ي':'ی','ى':'ی','ك':'ک','\u200c':' ','ـ':''})).casefold()
    text=''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text if not unicodedata.combining(c))
    return ' '.join(text.split())
def tokens(text): return re.findall(r'[^\W_]+', norm(text), flags=re.UNICODE)
def ascii_tokens(text):
    # Google rouge/tokenize.py default core when use_stemmer=False.
    text = re.sub(r'[^a-z0-9]+', ' ', text.lower())
    return [t for t in re.split(r'\s+', text) if re.match(r'^[a-z0-9]+$', t)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);return r.fieldnames,list(r)
def writecsv(name,rows,fields=None):
    rows=list(rows)
    if not fields: fields=list(rows[0]) if rows else ['none']
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def f1(a,b):
    common=sum((Counter(a)&Counter(b)).values())
    return 2*common/(len(a)+len(b)) if a and b else 0.0
def ngrams(a,n):return [tuple(a[i:i+n]) for i in range(len(a)-n+1)]
def lcs(a,b):
    last=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b,1):current.append(last[j-1]+1 if x==y else max(current[-1],last[j]))
        last=current
    return last[-1]
def rouge(a,b):return {'r1_f1':f1(a,b),'r2_f1':f1(ngrams(a,2),ngrams(b,2)),'rl_f1':2*lcs(a,b)/(len(a)+len(b)) if a and b else 0.0}
def desc(values):
    values=sorted(values)
    def q(p):
        k=(len(values)-1)*p;lo=int(k);hi=math.ceil(k)
        return values[lo]*(hi-k)+values[hi]*(k-lo) if hi!=lo else values[lo]
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p05':q(.05),'p95':q(.95),'min':min(values),'max':max(values)} if values else {'n':0}
def digits(text):
    return Counter(re.findall(r'\d+(?:[٫.,:/]\d+)*', norm(text)))

def connected_groups(rows):
    """Keep repeated source/reference strings, including cross-role matches, together."""
    parent=list(range(len(rows)));sz=[1]*len(rows)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if sz[a]<sz[b]:a,b=b,a
        parent[b]=a;sz[a]+=sz[b]
    seen={}
    for i,r in enumerate(rows):
        for s in set([norm(r['ai']),norm(r['human'])]):
            if not s:continue
            if s in seen:union(i,seen[s])
            else:seen[s]=i
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    return groups


def main():
    global HERE,REPO,FULL,TRAIN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--out',type=Path,default=HERE)
    args=parser.parse_args()
    REPO=args.repo.resolve();HERE=args.out.resolve();HERE.mkdir(parents=True,exist_ok=True)
    FULL=REPO/'data/persian_humanizer_dataset.csv';TRAIN=REPO/'data/train_set.csv'
    names,rows=read(FULL);tnames,train=read(TRAIN)
    missing={k:sum(not(r.get(k)or'').strip() for r in rows)for k in names}
    index={k:defaultdict(list) for k in ['exact_input','normalized_input','exact_output','normalized_output','normalized_pair']}
    details=[];numflag=[]
    for i,r in enumerate(rows,2):
        a,h=r['ai'],r['human'];na,nh=norm(a),norm(h)
        for key,v in [('exact_input',a),('normalized_input',na),('exact_output',h),('normalized_output',nh),('normalized_pair',(na,nh))]:index[key][v].append(i)
        at,ht=tokens(a),tokens(h);aa,ha=ascii_tokens(a),ascii_tokens(h)
        d={'csv_line':i,'ai':a,'human':h,'style':r.get('style',''),'transformation':r.get('transformation',''),'ai_unicode_tokens':len(at),'human_unicode_tokens':len(ht),'human_to_ai_token_ratio':len(ht)/len(at)if at else None,'ai_ascii_tokens':len(aa),'human_ascii_tokens':len(ha),'normalized_identity':na==nh,**{'unicode_'+k:v for k,v in rouge(at,ht).items()},**{'ascii_'+k:v for k,v in rouge(aa,ha).items()}}
        na_num,nh_num=digits(a),digits(h)
        d.update(input_number_tokens=json.dumps(na_num,ensure_ascii=False),reference_number_tokens=json.dumps(nh_num,ensure_ascii=False),number_multisets_equal=na_num==nh_num)
        details.append(d)
        if na_num!=nh_num:numflag.append(d)
    dupstats={}
    duplines=[]
    for kind,groups in index.items():
        duplicates=[v for v in groups.values()if len(v)>1]
        dupstats[kind]={'unique_values':len(groups),'duplicate_groups':len(duplicates),'rows_in_duplicate_groups':sum(map(len,duplicates)),'excess_rows':sum(len(v)-1 for v in duplicates)}
        for ids in duplicates:duplines.append({'kind':kind,'group_size':len(ids),'csv_lines':'|'.join(map(str,ids))})
    conflicts=[]
    for a,ids in index['normalized_input'].items():
        outputs={norm(rows[i-2]['human'])for i in ids}
        styles={rows[i-2].get('style','')for i in ids}
        if len(outputs)>1:conflicts.append({'normalized_input':a,'rows':len(ids),'distinct_normalized_references':len(outputs),'styles':'|'.join(sorted(styles)),'csv_lines':'|'.join(map(str,ids))})
    # Infer historical row holdout only when training rows are a complete multiset subset.
    fields=names
    fullcounter=Counter(tuple(r.get(k,'')for k in fields)for r in rows)
    traincounter=Counter(tuple(r.get(k,'')for k in fields)for r in train)
    extra=traincounter-fullcounter
    remain=fullcounter-traincounter
    inferred=[]
    for values,n in remain.items():inferred +=[dict(zip(fields,values))]*n
    train_ai={norm(r['ai']) for r in train};train_h={norm(r['human'])for r in train}
    hold_overlap={'train_rows_not_in_full_multiset':sum(extra.values()),'inferred_complement_rows':len(inferred),'inferred_test_source_seen_in_train_rows':sum(norm(r['ai'])in train_ai for r in inferred),'inferred_test_reference_seen_in_train_rows':sum(norm(r['human'])in train_h for r in inferred)}
    groups=connected_groups(rows)
    assignments=[];by_split=defaultdict(list);group_counts=Counter()
    for ids in groups.values():
        texts=sorted({norm(rows[i][f])for i in ids for f in ['ai','human']})
        gid=hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        u=int(hashlib.sha256(('ph-journal-group-split-v1|'+gid).encode()).hexdigest()[:8],16)/2**32
        split='train' if u<.70 else 'development'if u<.80 else 'test'
        group_counts[split]+=1
        for i in ids:
            r=rows[i]
            stable_id='PH-'+hashlib.sha256((sha(FULL)+'|csv-line|'+str(i+2)).encode()).hexdigest()[:20]
            record_sha=hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assignments.append({'stable_id':stable_id,'group_id':gid,'source_csv_line':i+2,'source_record_sha256':record_sha,'split':split,'ai':r['ai'],'human':r['human'],'style':r.get('style',''),'transformation':r.get('transformation','')});by_split[split].append(i)
    alltexts={s:{norm(rows[i][f])for i in ids for f in ['ai','human']}for s,ids in by_split.items()}
    split_overlap={a+'__'+b:len(alltexts[a]&alltexts[b])for a in by_split for b in by_split if a<b}
    assert not any(split_overlap.values())
    splitstats={s:{'rows':len(ids),'groups':group_counts[s],'style_counts':dict(Counter(rows[i].get('style','')for i in ids)),'transformation_counts':dict(Counter(rows[i].get('transformation','')for i in ids))}for s,ids in by_split.items()}
    # Exact random-without-replacement same-label retrieval expectation excluding self.
    styles=Counter(r.get('style','')for r in rows);n=len(rows)
    random_agreement=sum(c*(c-1)for c in styles.values())/(n*(n-1))
    out={
      'scope':'Audit of repository CSV cells, not generated responses or a replication of historical 3173-pair experiment',
      'input_files':{'full':{'path':str(FULL),'sha256':sha(FULL),'rows':len(rows),'columns':names},'train':{'path':str(TRAIN),'sha256':sha(TRAIN),'rows':len(train),'columns':tnames}},
      'normalization':'NFKC; Arabic yeh/alef-maksura/kaf to Persian forms; ZWNJ to space; tatweel and combining marks removed; all Unicode decimals to ASCII; casefold and whitespace collapse. Punctuation otherwise retained.',
      'missing_or_blank_cells':missing,'style_counts':dict(styles),'transformation_counts':dict(Counter(r.get('transformation','')for r in rows)),
      'duplicate_counts':dupstats,'normalized_input_conflicts':{'groups':len(conflicts),'rows':sum(r['rows']for r in conflicts),'meaning':'Multiple distinct reference strings for same normalized source; this does not itself prove label invalidity.'},
      'normalized_identity_pairs':sum(d['normalized_identity']for d in details),
      'lengths_unicode_alphanumeric_tokens':{'source':desc([d['ai_unicode_tokens']for d in details]),'reference':desc([d['human_unicode_tokens']for d in details]),'reference_source_ratio':desc([d['human_to_ai_token_ratio']for d in details if d['human_to_ai_token_ratio']is not None])},
      'source_reference_rouge_f1_macro':{k:statistics.mean(d[k]for d in details)for k in details[0]if k.startswith(('unicode_r','ascii_r'))},
      'ascii_tokenizer_zero_token_rows':{'source':sum(not d['ai_ascii_tokens']for d in details),'reference':sum(not d['human_ascii_tokens']for d in details),'both':sum(not d['ai_ascii_tokens']and not d['human_ascii_tokens']for d in details)},
      'number_token_diagnostics':{'any_numbers_in_either':sum(bool(digits(r['ai'])or digits(r['human']))for r in rows),'source_with_numbers':sum(bool(digits(r['ai']))for r in rows),'reference_with_numbers':sum(bool(digits(r['human']))for r in rows),'different_multiset_rows':len(numflag),'caveat':'Differences can reflect spelling, units, benign rewriting or genuine error; unchanged tokens do not prove fact or meaning preservation.'},
      'train_complement_overlap':hold_overlap,
      'random_same_style_agreement_without_self':random_agreement,'majority_label_overall_agreement':max(styles.values())/n,
      'proposed_group_split':{'status':'New deterministic proposal, no experiment executed; not historical split','version':'ph-journal-group-split-v1','assignment':'SHA256 of sorted unique normalized texts in connected component; hashed salt mapped to train<0.70, development<0.80, test otherwise','group_count':len(groups),'largest_group':max(map(len,groups.values())),'groups_size_at_least_2':sum(len(v)>1 for v in groups.values()),'cross_split_normalized_text_overlap':split_overlap,'splits':splitstats},
      'metric_definitions':{'unicode_tokens':'Maximal Unicode-alphanumeric spans after normalization; underscores excluded','ROUGE':'Macro average F1 per CSV source/reference pair; ngram multiset overlap, ROUGE-L longest common token subsequence; empty overlap F1=0; no stemming. This custom implementation is not claimed numerically identical to every ROUGE package.','ascii_diagnostic':'Google rouge/tokenize.py default core: lowercase, replace [^a-z0-9]+ by space, split whitespace, retain ^[a-z0-9]+$; no stemmer','numbers':'Counter of digit sequences with optional decimal/date/time separators [٫.,:/] after normalization; written number words not parsed'}
    }
    (HERE/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    writecsv('row_diagnostics.csv',details)
    writecsv('number_change_flags.csv',numflag,fields=list(details[0]))
    writecsv('duplicate_groups.csv',duplines)
    writecsv('source_reference_conflicts.csv',conflicts)
    assignments.sort(key=lambda r:r['source_csv_line'])
    writecsv('proposed_group_split.csv',assignments)
    record_fields=['stable_id','group_id','source_csv_line','source_record_sha256','ai','human','style','transformation']
    records=[{k:r[k]for k in record_fields}for r in assignments]
    writecsv('records.csv',records,fields=record_fields)
    manifest={'schema_version':1,'status':'New proposed split; not a reconstruction of any historical evaluation','dataset_path':os.path.relpath(FULL,HERE),'dataset_sha256':sha(FULL),'dataset_rows':len(rows),'records_csv':'records.csv','records_sha256':sha(HERE/'records.csv'),'normalization':out['normalization'],'stable_id_definition':'PH- plus first20hex SHA256(dataset_file_sha256 + |csv-line| + physical_record_number). source_csv_line counts CSV record positions including header; records contain no multiline values.','source_record_sha256_definition':'SHA256 canonical UTF-8 JSON of original CSV row, ensure_ascii=False,sort_keys=True,separators=(comma,colon)','splits':{},'per_split_csv':{}}
    for external,internal in [('train','train'),('dev','development'),('test','test')]:
        selected=[{k:r[k]for k in record_fields}for r in assignments if r['split']==internal]
        writecsv(external+'.csv',selected,fields=record_fields)
        manifest['splits'][external]=[r['stable_id']for r in selected]
        manifest['per_split_csv'][external]={'file':external+'.csv','sha256':sha(HERE/(external+'.csv')),'rows':len(selected),'groups':len({r['group_id']for r in selected})}
    (HERE/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    writecsv('unverified_full_minus_train_multiset.csv',inferred,fields=names)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
