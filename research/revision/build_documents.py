"""Build the revision documents from local evidence; no model or network calls."""
from pathlib import Path
import json
import subprocess
import re
import base64
import sys

ROOT = Path(__file__).resolve().parent


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    data = json.loads((ROOT/'corpus_analysis/analysis.json').read_text())
    out = ROOT/'manuscript/figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':300, 'font.family':'DejaVu Sans'})
    fig, ax = plt.subplots(1, 2, figsize=(7.2,3.7), gridspec_kw={'width_ratios':[1.2,1]})
    styles=sorted(data['style_counts'].items(),key=lambda x:-x[1])
    shown=styles[:7]+[('Other 15 labels',sum(v for _,v in styles[7:]))]
    ax[0].barh([x[0]for x in shown][::-1],[x[1]for x in shown][::-1],color='#535353')
    ax[0].set_xlabel('Stored pairs');ax[0].set_title('(a) Stored metadata labels',fontsize=10)
    ax[0].set_xlim(0,1950)
    for i,(_,v) in enumerate(shown[::-1]):ax[0].text(v+18,i,str(v),va='center',fontsize=9)
    scores=data['source_reference_rouge_f1_macro'];xs=[0,1,2]
    ax[1].bar([x-.17 for x in xs],[scores[k]for k in ['unicode_r1_f1','unicode_r2_f1','unicode_rl_f1']],width=.34,color='#333333',label='Unicode-aware')
    ax[1].bar([x+.17 for x in xs],[scores[k]for k in ['ascii_r1_f1','ascii_r2_f1','ascii_rl_f1']],width=.34,color='#b5b5b5',edgecolor='black',label='ASCII-only diagnostic')
    ax[1].set_xticks(xs,['ROUGE-1','ROUGE-2','ROUGE-L'],rotation=20);ax[1].set_ylim(0,.6)
    ax[1].set_ylabel('Macro source–rewrite F1');ax[1].legend(frameon=False,fontsize=8.5)
    ax[1].set_title('(b) Tokenization sensitivity',fontsize=10)
    fig.tight_layout(pad=1.8)
    fig.savefig(out/'corpus_audit.png',bbox_inches='tight');fig.savefig(out/'corpus_audit.pdf',bbox_inches='tight');plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.2,3.8));ax.set_xlim(0,10.4);ax.set_ylim(0,4.4);ax.axis('off')
    def box(x,y,w,h,text,done=True):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor='#eeeeee' if done else 'white',edgecolor='black',linestyle='-' if done else '--',linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='black',linewidth=1))
    ax.text(.1,4.12,'CORPUS AND VERSION AUDIT',fontsize=10,weight='bold')
    box(.1,2.85,2.7,.82,'Pinned source/rewrite pairs\n3511 rows')
    box(3.35,2.85,2.7,.82,'Shared-text grouping\n3499 connected groups')
    box(6.65,2.85,3.35,.82,'Train 2438 / Dev 371\nTest 702; disjoint texts')
    arrow((2.86,3.26),(3.28,3.26));arrow((6.12,3.26),(6.58,3.26))
    ax.text(.1,2.32,'GENERATION AND INDEPENDENT ANALYSIS',fontsize=10,weight='bold')
    box(.1,.9,2.7,.96,'Two fixed API models\nFour matched conditions\nSame inputs + instructions')
    box(3.35,.9,2.7,.96,'Frozen prompts + raw data\n880 successful responses\n800 main + 80 numeric')
    box(6.65,.9,3.35,.96,'Independent checks + CIs\nOverlap / length / digit flags\nModel-assisted case inspection')
    arrow((2.86,1.38),(3.28,1.38));arrow((6.12,1.38),(6.58,1.38))
    ax.plot([10.06,10.22,10.22,1.45],[3.26,3.26,2.08,2.08],color='black',linewidth=1)
    arrow((1.45,2.08),(1.45,1.94))
    ax.text(.1,.27,'Completed stages; source files and raw responses are linked by frozen hashes.',fontsize=8.5)
    fig.tight_layout();fig.savefig(out/'evaluation_flow.png',bbox_inches='tight');fig.savefig(out/'evaluation_flow.pdf',bbox_inches='tight');plt.close(fig)
    result_path=ROOT/'experiments/combined_analysis/analysis.json'
    if result_path.exists():
        r=json.loads(result_path.read_text());rows=[v for v in r['paired_contrasts']if v['metric']=='reference_rougeL_f1']
        fig,ax=plt.subplots(figsize=(7.2,3.4))
        labels=[]
        for i,v in enumerate(rows):
            mean=v['mean_paired_difference'];lo=v['percentile_95_ci_low'];hi=v['percentile_95_ci_high']
            ax.errorbar(mean,len(rows)-1-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='black',capsize=4,markersize=5)
            model='GPT-4.1 mini'if '4.1-'in v['model']else'GPT-4o mini'
            control='LLM-0'if'LLM0'in v['contrast']else'Random-4'
            labels.append(model+' | Sem-4 − '+control)
        ax.set_yticks(range(len(rows)),labels[::-1],fontsize=9)
        ax.axvline(0,color='#777',linestyle='--',linewidth=1)
        ax.set_xlabel('Difference in output–reference ROUGE-L F1')
        ax.set_title('Reference agreement: paired 95% intervals',fontsize=10)
        fig.tight_layout();fig.savefig(out/'paired_reference_contrasts.png',bbox_inches='tight');fig.savefig(out/'paired_reference_contrasts.pdf',bbox_inches='tight');plt.close(fig)


def build(md:Path, rtl:bool):
    from docx import Document
    from docx.shared import Cm,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    docx=md.with_suffix('.docx')
    subprocess.run(['pandoc',str(md),'--resource-path='+str(md.parent),'-o',str(docx)],check=True)
    d=Document(docx)
    for section in d.sections:
        section.page_width=Cm(21);section.page_height=Cm(29.7)
        section.top_margin=Cm(1.8);section.bottom_margin=Cm(1.8)
        section.left_margin=Cm(2);section.right_margin=Cm(2)
    for style in d.styles:
        if style.type==1:
            style.font.name='Noto Naskh Arabic' if rtl else 'DejaVu Serif'
            style.font.size=Pt(11.5 if rtl else 10.5);style.font.color.rgb=RGBColor(0,0,0)
            style.paragraph_format.line_spacing=1.24 if rtl else 1.12
            style.paragraph_format.space_after=Pt(6)
            if style.name.startswith('Heading'):
                style.font.size=Pt(16 if style.name=='Heading 1' else 12.5)
                style.paragraph_format.keep_with_next=True
    paras=list(d.paragraphs)+[p for t in d.tables for r in t.rows for c in r.cells for p in c.paragraphs]
    for p in paras:
        if rtl:
            p.alignment=2
            bidi=OxmlElement('w:bidi');bidi.set(qn('w:val'),'1');p._p.get_or_add_pPr().append(bidi)
        for run in p.runs:
            run.font.color.rgb=RGBColor(0,0,0)
            if rtl:
                run.font.name='Noto Naskh Arabic'
                fonts=run._element.get_or_add_rPr().rFonts
                if fonts is not None:fonts.set(qn('w:cs'),'Noto Naskh Arabic')
                ar=OxmlElement('w:rtl');ar.set(qn('w:val'),'1');run._element.get_or_add_rPr().append(ar)
    for t in d.tables:
        for i,r in enumerate(t.rows):
            if i==0:r._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for c in r.cells:
                for p in c.paragraphs:
                    for run in p.runs:run.font.size=Pt(10)
    for shape in d.inline_shapes:
        if shape.width>Cm(16.8):
            ratio=Cm(16.8)/shape.width;shape.width=Cm(16.8);shape.height=int(shape.height*ratio)
    d.core_properties.title=md.read_text().splitlines()[0].lstrip('# ')
    d.core_properties.author='Rizwanullah Muahed'
    d.save(docx)
    html=subprocess.check_output(['pandoc',str(md),'--standalone','--embed-resources','--mathml','--resource-path='+str(md.parent),'-t','html5','--metadata','title='+d.core_properties.title],text=True)
    html=re.sub(r'<header id="title-block-header">[\s\S]*?</header>','',html)
    html=html.replace('<html','<html dir="rtl" lang="fa"' if rtl else '<html lang="en"',1)
    css='''@page{size:A4;margin:18mm 20mm}*{box-sizing:border-box}body{color:#000;font-family:"DejaVu Serif",serif;font-size:10.6pt;line-height:1.4}h1,h2,h3,h4{color:#000!important;-webkit-text-fill-color:#000!important;break-after:avoid}h1{font-size:21pt;line-height:1.24;margin:0 0 18px}h2{font-size:14pt;margin:22px 0 10px}h3{font-size:11.7pt;margin:16px 0 8px}p{margin:0 0 9px;text-align:justify;orphans:3;widows:3}img{display:block;max-width:100%;height:auto;margin:12px auto}figure{margin:16px 0;break-inside:avoid}figcaption{font-size:9.5pt;text-align:left;line-height:1.3}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:9pt;line-height:1.3}thead{display:table-header-group}th,td{border:1px solid #999;padding:5px 6px;vertical-align:top;text-align:left;overflow-wrap:anywhere}th{background:#f0f0f0;color:#000}tr{break-inside:avoid}code{font-family:"DejaVu Sans Mono",monospace;font-size:.82em;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#000;text-decoration:underline;overflow-wrap:anywhere}li{margin-bottom:6px}math{font-size:.95em}'''
    if rtl:css+='body{font-family:"Noto Naskh Arabic",serif;font-size:12pt;line-height:1.6}h1{font-size:21pt}h2{font-size:15pt}h3{font-size:13pt}th,td{text-align:right;font-size:10.5pt}code{direction:ltr;unicode-bidi:isolate}figcaption{text-align:right}'
    css+='body{margin:0;padding:0;max-width:none;width:auto}h1.title{display:none}'
    html=html.replace('</head>','<style>'+css+'</style></head>')
    md.with_suffix('.html').write_text(html)
    print(json.dumps({'docx':str(docx),'html':str(md.with_suffix('.html')),'headings':'black'}))


def render(html:Path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page();page.route('**/*',lambda r:r.abort())
        page.set_content(html.read_text(),wait_until='load')
        page.evaluate('document.fonts.ready')
        colors=page.locator('h1,h2,h3,h4').evaluate_all('(nodes)=>nodes.map(n=>getComputedStyle(n).color)')
        assert colors and set(colors)=={'rgb(0, 0, 0)'},colors
        page.pdf(path=str(html.with_suffix('.pdf')),format='A4',prefer_css_page_size=True,print_background=True,display_header_footer=True,header_template='<div></div>',footer_template='<div style="font-family:Arial;font-size:8px;width:100%;text-align:center;color:#555"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    from pypdf import PdfReader,PdfWriter
    import io
    pdf=html.with_suffix('.pdf')
    reader=PdfReader(io.BytesIO(pdf.read_bytes()));writer=PdfWriter();writer.clone_document_from_reader(reader)
    writer.add_metadata({'/Author':'Rizwanullah Muahed','/Title':page_title(html)})
    with pdf.open('wb') as target:writer.write(target)
    print(html.with_suffix('.pdf'))


def page_title(html):
    from html import unescape
    match=re.search(r'<title>([\s\S]*?)</title>',Path(html).read_text())
    return unescape(match.group(1).strip()) if match else Path(html).stem


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path);args=p.parse_args()
    if args.render:render(args.render)
    else:
        figures()
        extra=ROOT/'editorial_update/add_figures.py'
        if extra.exists():subprocess.run([sys.executable,str(extra)],check=True)
        build(ROOT/'manuscript/Revised_Manuscript_EN.md',False)
        build(ROOT/'report/Revision_Report_Dari.md',True)
