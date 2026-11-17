"""Verify the published research offline; no credential or provider client is used."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile

REPO=Path(__file__).resolve().parents[1]

def alias(source,destination):
    try:destination.symlink_to(source,target_is_directory=True)
    except (OSError,NotImplementedError):shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.env*'))

def main():
    manifest=json.loads((REPO/'research/PUBLICATION_MANIFEST.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        path=(REPO/name).resolve();assert path.is_relative_to(REPO) and path.is_file(),name
        with path.open('rb')as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        assert actual==expected,'Published file changed: '+name
    metadata=json.loads((REPO/'research/current/metadata.json').read_text())
    assert metadata['authors']==['Rizwanullah Muahed'] and metadata['instructor_is_coauthor'] is False
    project=REPO/'research/revision'
    with tempfile.TemporaryDirectory(prefix='persian-research-check-')as directory:
        root=Path(directory)
        alias(project,root/'persian_humanization_revision')
        alias(REPO/'research/evidence/Persian_Humanizer_Ai',root/'Persian_Humanizer_Ai')
        first=root/'main_analysis';second=root/'numeric_analysis'
        subprocess.run([sys.executable,'-B',str(project/'experiments/analyze_api_results.py'),'--directory',str(project/'experiments/api_study_v1'),'--secondary-directory',str(project/'experiments/api_study_secondary'),'--artifact-root',str(root),'--out',str(first)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,'-B',str(project/'experiments/analyze_numeric_results.py'),'--directory',str(project/'experiments/api_numeric_diagnostic_run'),'--artifact-root',str(root),'--main-diagnostics',str(first/'response_diagnostics.csv'),'--out',str(second)],check=True,stdout=subprocess.DEVNULL)
        a=json.loads((first/'analysis.json').read_text());b=json.loads((second/'analysis.json').read_text())
        assert a['status']==b['status']=='complete'
        assert a['validated_responses']==800 and b['validated_responses']==80
        assert a['query_groups']==100 and b['independent_source_groups']==10
        assert a['expected_unavailable_model_http404']==400 and a['unexpected_error_count']==b['recorded_errors']==0
    print(json.dumps({'verification':'passed','published_files_verified':len(manifest['files_sha256']),'main_responses':800,'numeric_responses':80,'expected_access_failures_preserved':400,'authors':metadata['authors'],'instructor_is_coauthor':False,'new_api_calls':0,'human_ratings_created':0},indent=2))

if __name__=='__main__':main()
