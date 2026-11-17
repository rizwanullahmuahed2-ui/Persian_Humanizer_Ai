"""Read-only corpus, SQLite metadata and Git lineage audit; no model calls."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import io
import json
import sqlite3
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Local Git checkout to inspect without changing it')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--original-pdf', type=Path, help='Optional separately supplied PDF for byte comparison')
    args = parser.parse_args()
    ROOT, OUT = args.repo.resolve(), args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    full = rows(ROOT/'data/persian_humanizer_dataset.csv')
    train = rows(ROOT/'data/train_set.csv')
    key = lambda row: (row.get('ai', '').strip(), row.get('human', '').strip())
    fc, tc = collections.Counter(map(key, full)), collections.Counter(map(key, train))
    result = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'full_rows': len(full),
        'full_unique_pairs': len(fc),
        'full_unique_ai': len(set(row['ai'] for row in full)),
        'train_rows': len(train),
        'train_unique_pairs': len(tc),
        'train_unique_ai': len(set(row['ai'] for row in train)),
        'train_rows_not_match_full_pairs': sum((tc-fc).values()),
        'full_rows_absent_from_train_pairs': sum((fc-tc).values()),
        'pair_set_intersection': len(set(fc) & set(tc)),
        'full_styles': dict(collections.Counter(row.get('style') for row in full)),
        'full_transitions': dict(collections.Counter(row.get('transformation') for row in full)),
        'full_blank_ai': sum(not row.get('ai', '').strip() for row in full),
        'full_blank_human': sum(not row.get('human', '').strip() for row in full),
        'full_sha256': sha(ROOT/'data/persian_humanizer_dataset.csv'),
        'train_sha256': sha(ROOT/'data/train_set.csv'),
        'sqlite_indices': [], 'git_history': [],
    }
    for db in sorted(ROOT.glob('data/*/chroma.sqlite3')):
        connection = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)
        metadata = collections.defaultdict(dict)
        for rid, key_name, value in connection.execute('select id,key,string_value from embedding_metadata'):
            metadata[rid][key_name] = value
        pairs = collections.Counter((record.get('ai', ''), record.get('human', '')) for record in metadata.values())
        result['sqlite_indices'].append({
            'path': str(db.relative_to(ROOT)), 'sha256': sha(db),
            'collections': connection.execute('select id,name,dimension,config_json_str from collections').fetchall(),
            'collection_metadata': connection.execute('select * from collection_metadata').fetchall(),
            'embedding_count': connection.execute('select count(*) from embeddings').fetchone()[0],
            'metadata_rows': len(metadata), 'unique_metadata_pairs': len(pairs),
            'metadata_pairs_absent_from_full': sum((pairs-fc).values()),
            'metadata_rows_present_in_full': sum((pairs & fc).values()),
            'document_equals_ai': sum(record.get('chroma:document') == record.get('ai') for record in metadata.values()),
            'document_contains_both_ai_and_human': sum(record.get('ai', '') in record.get('chroma:document', '') and record.get('human', '') in record.get('chroma:document', '') for record in metadata.values()),
        })
        connection.close()
    for revision in subprocess.check_output(['git', 'rev-list', '--all'], cwd=ROOT, text=True).splitlines():
        try:
            content = subprocess.check_output(['git', 'show', revision+':data/persian_humanizer_dataset.csv'], cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        entries = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
        result['git_history'].append({'commit': revision, 'corpus_rows': len(entries), 'corpus_sha256': hashlib.sha256(content).hexdigest()})
    result['paper_pdf_sha256'] = sha(ROOT/'research/Persian_Text_Humanization_Paper.pdf')
    if args.original_pdf:
        result['uploaded_pdf_sha256'] = sha(args.original_pdf)
        result['supplied_pdf_matches_repository'] = result['uploaded_pdf_sha256'] == result['paper_pdf_sha256']
    (OUT/'repository_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key_name: value for key_name, value in result.items() if key_name not in ['full_styles', 'full_transitions', 'git_history']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
