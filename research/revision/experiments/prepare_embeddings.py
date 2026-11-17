"""Pin and encode source text only; generation and reference text are excluded."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,urllib.request,importlib.metadata

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());records=a.manifest.parent/manifest['records_csv']
    assert sha(records)==manifest['records_sha256']
    model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    with urllib.request.urlopen('https://huggingface.co/api/models/'+model,timeout=30) as response:info=json.load(response)
    revision=info['sha'];assert len(revision)==40
    (a.out/'embedding_selection.json').write_text(json.dumps({'model':model,'revision':revision,'records_sha256':sha(records),'policy':'source column only; no generation or reference encoding'},indent=2)+'\n')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';os.environ['HF_HUB_DISABLE_XET']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch,numpy as np
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    with records.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    rows.sort(key=lambda r:r['stable_id']);ids=[r['stable_id']for r in rows];texts=[r['ai']for r in rows]
    encoder=SentenceTransformer(model,revision=revision,cache_folder=str(a.cache),device='cpu')
    vectors=encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
    assert vectors.shape==(len(rows),384) and np.isfinite(vectors).all()
    path=a.out/'vectors.npz';np.savez_compressed(path,ids=np.array(ids),vectors=vectors)
    metadata={'method':'sentence-transformers','model':model,'revision':revision,'source_field':'ai','reference_text_encoded':False,'dimension':384,'normalized':True,'records_sha256':sha(records),'vectors_sha256':sha(path),'text_sha256':{r['stable_id']:hashlib.sha256(r['ai'].encode()).hexdigest()for r in rows},'versions':{name:importlib.metadata.version(name)for name in ['torch','sentence-transformers','transformers','numpy','huggingface-hub']},'max_seq_length':encoder.max_seq_length}
    (a.out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','model':model,'revision':revision,'rows':len(rows),'dimension':384}))

if __name__=='__main__':main()
