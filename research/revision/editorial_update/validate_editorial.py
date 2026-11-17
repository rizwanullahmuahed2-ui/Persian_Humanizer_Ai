"""Check editorial invariants without API calls or changing research artifacts."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,difflib

ROOT=Path(__file__).resolve().parents[1]

def main():
    before=(ROOT/'editorial_update/pre_copyedit.md').read_text()
    after=(ROOT/'manuscript/Revised_Manuscript_EN.md').read_text()
    a,ab=before.split('\n## References\n',1);b,bb=after.split('\n## References\n',1)
    citations=lambda s:re.findall(r'\[\d+(?:\s*,\s*\d+)*\]',s)
    checks={
      'numeric_literals_unchanged':Counter(re.findall(r'\d+(?:\.\d+)?',a))==Counter(re.findall(r'\d+(?:\.\d+)?',b)),
      'tables_unchanged':[s for s in a.splitlines()if s.startswith('|')]==[s for s in b.splitlines()if s.startswith('|')],
      'headings_unchanged':[s for s in a.splitlines()if s.startswith('#')]==[s for s in b.splitlines()if s.startswith('#')],
      'captions_unchanged':re.findall(r'!\[[^\n]*',a)==re.findall(r'!\[[^\n]*',b),
      'citation_sequence_unchanged':citations(a)==citations(b),
      'bibliography_unchanged_during_copyedit':ab==bb,
      'inline_literals_unchanged':re.findall(r'`[^`\n]+`',a)==re.findall(r'`[^`\n]+`',b),
      'display_equations_unchanged':re.findall(r'\$\$[\s\S]*?\$\$',a)==re.findall(r'\$\$[\s\S]*?\$\$',b),
    }
    assert all(checks.values()),checks
    refs=[int(x)for x in re.findall(r'^\[(\d+)\]',bb,re.M)]
    assert refs==list(range(1,26))
    first=[]
    for group in citations(b):
        for n in map(int,group.strip('[]').split(',')):
            if n not in first:first.append(n)
    assert first==refs
    figures=re.findall(r'!\[Figure (\d+)\.[^\n]*\]\(([^\n]+)\)',b)
    assert [int(n)for n,_ in figures]==list(range(1,8))
    assert all((ROOT/'manuscript'/file).is_file()for _,file in figures)
    metadata=json.loads((ROOT/'editorial_update/figure_notes.json').read_text())
    for file,expected in metadata['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected,file
    seal=ROOT/'editorial_update/EXPERIMENT_FILES_BEFORE_EDITORIAL_UPDATE.json'
    frozen=json.loads(seal.read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in frozen.items())
    checks.update(reference_count=len(refs),all_references_cited=True,figure_count=len(figures),research_files_unchanged=len(frozen),new_api_calls=0,edited_prose_lines=sum(line.startswith('+ ')for line in difflib.ndiff(a.splitlines(),b.splitlines())))
    (ROOT/'editorial_update/COPYEDIT_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
