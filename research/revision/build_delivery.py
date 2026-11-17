"""Assemble reviewable artifacts, excluding all credentials and runtime models."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,re,shutil,subprocess,zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery'
PACKAGE=OUT/'Persian_Humanization_Package'
REVISION_DIRS={'manuscript','report','corpus_analysis','code_audit','experiments','original_figures','editorial_update'}
REVISION_FILES={'build_documents.py','build_delivery.py','code_audit.md','methods_audit.md','original_extracted.txt','verified_references.md'}

def excluded(directory,names):
    return {name for name in names if name in {'.git','.venv','__pycache__','node_modules','runtime','models','.pytest_cache','.ruff_cache'} or name.startswith('.env') or name.endswith(('.pyc','.gguf','.safetensors')) or name in {'inferred_old_holdout_complement.csv','stdout.json'}}

def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    if PACKAGE.exists():shutil.rmtree(PACKAGE)
    PACKAGE.mkdir()
    for child in ROOT.iterdir():
        if child.is_dir() and child.name not in REVISION_DIRS:continue
        if child.is_file() and child.name not in REVISION_FILES:continue
        dest=PACKAGE/'persian_humanization_revision'/child.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if child.is_dir():shutil.copytree(child,dest,ignore=excluded)
        else:shutil.copy2(child,dest)
    snapshot=Path('/workspace/Persian_Humanizer_Ai')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=snapshot).decode().split('\0')
    for name in filter(None,tracked):
        path=snapshot/name
        if not path.is_file():continue
        if any(part.startswith('.env') or part in {'.git','.venv','__pycache__'} for part in Path(name).parts):continue
        if path.suffix in {'.bin','.pickle','.pyc'}:continue
        dest=PACKAGE/'Persian_Humanizer_Ai'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    (PACKAGE/'Persian_Humanizer_Ai/HISTORICAL_SNAPSHOT.md').write_text('This directory contains audit inputs from commit feb0f282430b13277c43dd32ca93ceec13c837df. The old PDF and its printed byline are preserved as historical evidence. Current authorship is corrected: Rizwanullah Muahed is the sole author; Abdul Awal Quraishi is his instructor and the report recipient. Use the revised article and current citation. Binary Chroma caches are omitted; the SQLite metadata files are retained for the documented audit.\n')
    shutil.copy2(OUT/'README_Dari.md',PACKAGE/'README_Dari.md')
    shutil.copy2(OUT/'verify_package.py',PACKAGE/'verify_package.py')
    shutil.copy2(OUT/'PDF_QUALITY_CHECKS.json',PACKAGE/'PDF_QUALITY_CHECKS.json')
    checks={}
    for name in ['article','report']:
        source=ROOT/('manuscript/Revised_Manuscript_EN.pdf'if name=='article'else'report/Revision_Report_Dari.pdf')
        checks[name+'_pdf_sha256']=sha(source)
    a=json.loads((ROOT/'experiments/combined_analysis/analysis.json').read_text());b=json.loads((ROOT/'experiments/numeric_analysis/analysis.json').read_text())
    certificate={'created_at_utc':datetime.now(timezone.utc).isoformat(),'main_query_groups':100,'main_successful_provider_responses':a['validated_responses'],'numeric_query_groups':10,'numeric_successful_provider_responses':b['validated_responses'],'scientific_http_attempts':a['scientific_http_attempts']+b['http_attempts_total'],'preserved_unavailable_model_http404':400,'new_human_ratings':0,'interpretation':'Completed automatic diagnostics and explicitly model-assisted case analysis; no human naturalness accuracy claimed','uncached_success_cost_usd_not_invoice':str(__import__('decimal').Decimal(a['uncached_cost_usd_not_invoice'])+__import__('decimal').Decimal(b['uncached_cost_usd_not_invoice'])),**checks}
    body,bibliography=(ROOT/'manuscript/Revised_Manuscript_EN.md').read_text().split('\n## References\n',1)
    certificate.update(reference_count=len(re.findall(r'^\[\d+\]',bibliography,re.M)),article_figure_count=len(re.findall(r'!\[Figure \d+\.',body)),authors=['Rizwanullah Muahed'],instructor='Abdul Awal Quraishi',editorial_checks='persian_humanization_revision/editorial_update/COPYEDIT_CHECKS.json')
    (PACKAGE/'COMPLETION_CERTIFICATE.json').write_text(json.dumps(certificate,indent=2)+'\n')
    files={}
    for p in sorted(PACKAGE.rglob('*')):
        if not p.is_file():continue
        assert not p.name.startswith('.env')
        if p.suffix in {'.py','.md','.txt','.json','.csv','.yml','.toml','.html'}:
            data=p.read_bytes()
            assert not re.search(rb'sk-proj-[A-Za-z0-9_-]{30,}',data),'Possible credential in '+str(p)
        files[str(p.relative_to(PACKAGE))]=sha(p)
    (PACKAGE/'PACKAGE_MANIFEST.json').write_text(json.dumps({'scope':'Saved evidence and author-review documents; no credentials or runtime weights','files_sha256':files},indent=2)+'\n')
    aliases=[('manuscript/Revised_Manuscript_EN.pdf','Persian_Humanization_Revised_Article.pdf'),('manuscript/Revised_Manuscript_EN.docx','Persian_Humanization_Revised_Article.docx'),('report/Revision_Report_Dari.pdf','Persian_Humanization_Revision_Report_Dari.pdf'),('report/Revision_Report_Dari.docx','Persian_Humanization_Revision_Report_Dari.docx')]
    for source,target in aliases:shutil.copy2(ROOT/source,OUT/target)
    print(json.dumps({'staged_files':len(files),'certificate':certificate},indent=2))

if __name__=='__main__':main()
