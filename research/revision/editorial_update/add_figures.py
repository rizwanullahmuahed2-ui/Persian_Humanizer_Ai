"""Add publication figures from frozen study artifacts; no model/API calls.

The figures distinguish text overlap, length variation and numeral diagnostics
from human naturalness or factual-accuracy outcomes. Existing figures are kept.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/persian_revision_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript' / 'figures'
MODELS = ('gpt-4.1-mini-2025-04-14', 'gpt-4o-mini-2024-07-18')
MODEL_LABELS = ('GPT-4.1 mini', 'GPT-4o mini')
CONDITIONS = ('LLM0', 'semantic1', 'semantic4', 'random4')
CONDITION_LABELS = ('LLM-0', 'Semantic-1', 'Semantic-4', 'Random-4')


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def save(fig, name):
    png, pdf = OUT / (name + '.png'), OUT / (name + '.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=.12)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return {'png': str(png.relative_to(ROOT)), 'pdf': str(pdf.relative_to(ROOT)),
            'png_sha256': file_hash(png), 'pdf_sha256': file_hash(pdf)}


def architecture():
    fig, ax = plt.subplots(figsize=(7.25, 5.7))
    ax.set(xlim=(0, 10), ylim=(0, 8))
    ax.axis('off')

    def box(x, y, width, height, content):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0.045', facecolor='#f0f0f0',
                     edgecolor='#111111', linewidth=1))
        ax.text(x + width/2, y + height/2, content, ha='center', va='center',
                fontsize=10.3, linespacing=1.2)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=10, color='#111111', linewidth=1.0))

    def elbow(points):
        ax.plot([p[0] for p in points], [p[1] for p in points], color='#111111', lw=1)

    box(.15, 6.55, 2.65, 1.05, 'Frozen training pairs\n2438 source–rewrite pairs\nStable IDs join both fields')
    box(3.15, 6.55, 2.65, 1.05, 'Encode source field only\nMultilingual MiniLM\nImmutable revision')
    box(6.15, 6.55, 3.55, 1.05, 'Normalized training vectors\nCosine ranking; train only\nNo test input/reference indexed')
    arrow((2.86, 7.075), (3.08, 7.075))
    arrow((5.86, 7.075), (6.08, 7.075))

    box(.15, 4.65, 2.65, 1.05, 'Held-out source input\n100 main query groups\n10 numeral query groups')
    box(3.15, 4.65, 2.65, 1.05, 'Encode input source\nSame embedding model\nNo reference text encoded')
    box(6.15, 4.65, 3.55, 1.05, 'Select paired demonstrations\nNearest 1 / nearest 4\nOr seeded random 4 from train')
    arrow((2.86, 5.175), (3.08, 5.175))
    arrow((5.86, 5.175), (6.08, 5.175))
    arrow((7.925, 6.49), (7.925, 5.76))
    ax.text(8.05, 6.1, 'rank / join IDs', fontsize=9, va='center')

    box(.15, 2.65, 5.65, 1.15, 'One shared preservation instruction + original input\n0 / 1 / 4 / 4 complete demonstrations\nThe zero-example condition has no fallback')
    box(6.15, 2.65, 3.55, 1.15, 'Fixed generator snapshot\nGPT-4.1 mini or GPT-4o mini\nOne fresh response per cell')
    arrow((1.475, 4.59), (1.475, 3.86))
    ax.text(1.6, 4.22, 'original input', fontsize=9, va='center')
    elbow([(7.925, 4.59), (7.925, 4.15), (4.65, 4.15)])
    arrow((4.65, 4.15), (4.65, 3.86))
    arrow((5.86, 3.225), (6.08, 3.225))

    box(6.15, .55, 3.55, 1.15, 'Raw provider response + usage\nModel, stop reason, response ID\n880 successful outputs; hashes')
    box(.15, .55, 5.65, 1.15, 'Independent artifact and output analysis\nSource retention / reference agreement\nLength / digit flags\nAutomatic diagnostics, not human ratings')
    arrow((7.925, 2.59), (7.925, 1.76))
    arrow((6.09, 1.125), (5.86, 1.125))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    return save(fig, 'new_runtime_architecture')


def outcomes(summary):
    values = {(s['model'], s['condition']): s for s in summary['summaries']}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True)
    x = np.arange(4)
    for panel, (metric, title) in enumerate([
            ('source_rougeL_f1', '(a) Agreement with the input'),
            ('reference_rougeL_f1', '(b) Agreement with associated rewrite')]):
        ax = axes[panel]
        for model_number, model in enumerate(MODELS):
            height = [values[model, condition][metric]['mean'] for condition in CONDITIONS]
            offset = -.18 if model_number == 0 else .18
            bars = ax.bar(x + offset, height, width=.34,
                          color='#424242' if model_number == 0 else '#d7d7d7',
                          edgecolor='black', linewidth=.6,
                          label=MODEL_LABELS[model_number])
            for bar, value in zip(bars, height):
                ax.text(bar.get_x()+bar.get_width()/2, value+.018, f'{value:.3f}',
                        ha='center', va='bottom', fontsize=9, rotation=90)
        ax.set_xticks(x, CONDITION_LABELS, rotation=23, ha='right')
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Macro ROUGE-L F1')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.09, right=.99, top=.83, bottom=.25, wspace=.18)
    return save(fig, 'new_generation_outcomes')


def lengths(records):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    statistics = []
    for model_number, model in enumerate(MODELS):
        ax = axes[model_number]
        groups = []
        for condition in CONDITIONS:
            values = [float(row['output_source_token_ratio']) for row in records
                      if row['model'] == model and row['condition'] == condition]
            assert len(values) == 100
            assert all(0 <= x < 2.35 for x in values), 'Length observations would be clipped'
            groups.append(values)
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            statistics.append({'model': model, 'condition': condition,
                               'n':len(values), 'q1':float(q1), 'median':float(median),
                               'q3':float(q3), 'min':min(values), 'max':max(values)})
        bp = ax.boxplot(groups, positions=np.arange(4), widths=.56, patch_artist=True,
                 whis=1.5, showfliers=True,
                 boxprops={'linewidth':.8, 'edgecolor':'black'},
                 medianprops={'linewidth':1.5, 'color':'black'},
                 whiskerprops={'linewidth':.8, 'color':'black'},
                 capprops={'linewidth':.8, 'color':'black'},
                 flierprops={'marker':'.', 'markersize':3, 'markerfacecolor':'#555555',
                             'markeredgecolor':'#555555', 'alpha':.75})
        for box in bp['boxes']:box.set_facecolor('#d4d4d4')
        ax.axhline(1, color='#666666', ls='--', linewidth=.9)
        ax.set_xticks(range(4), CONDITION_LABELS, rotation=23, ha='right')
        ax.set_ylim(0, 2.35)
        ax.set_title(MODEL_LABELS[model_number], fontsize=10)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Output / input Unicode-token count')
    fig.subplots_adjust(left=.09, right=.99, top=.9, bottom=.25, wspace=.18)
    return save(fig, 'new_length_distributions'), statistics


def numeral_coverage(main_summary, numeric_summary):
    assert all(s['source_digit_eligible'] == 0 for s in main_summary['summaries'])
    cells = {(s['model'], s['condition']):s for s in numeric_summary['summaries']}
    counts = np.array([[cells[m,c]['digit_multiset_flags'] for c in CONDITIONS]
                       for m in MODELS])
    assert counts.tolist() == [[0,0,1,0],[1,1,1,1]]
    assert all(s['source_digit_eligible'] == 10 for s in numeric_summary['summaries'])
    fig = plt.figure(figsize=(7.2, 3.15))
    left = fig.add_axes([.01, .24, .34, .68]);left.axis('off')
    left.text(.02, .98, '(a) Source-digit coverage', fontsize=10,
              weight='bold', va='top')
    for y, header, denominator, detail in [
            (.59, 'Primary random sample', '0 / 100', 'digit-bearing inputs'),
            (.16, 'Targeted posthoc supplement', '10 / 10', 'digit-bearing inputs')]:
        left.add_patch(FancyBboxPatch((.02,y),.96,.31,boxstyle='round,pad=.02',
                       facecolor='#eeeeee',edgecolor='#777777',linewidth=.7))
        left.text(.06,y+.235,header,fontsize=9.3,va='center')
        left.text(.06,y+.122,denominator,fontsize=14,weight='bold',va='center')
        left.text(.58,y+.12,detail.replace(' ', '\n',1),fontsize=9,va='center')
    right=fig.add_axes([.50,.28,.48,.51])
    right.imshow(counts,cmap='Greys',vmin=0,vmax=2,aspect='auto',interpolation='none')
    right.set_xticks(range(4),CONDITION_LABELS,rotation=23,ha='right',fontsize=9)
    right.set_yticks(range(2),MODEL_LABELS,fontsize=9)
    right.set_title('(b) Digit-expression mismatch flags',fontsize=10,pad=13)
    for row in range(2):
        for col in range(4):
            right.text(col,row,f'{counts[row,col]} / 10',ha='center',va='center',
                       fontsize=10.5,color='black',weight='bold')
    right.set_xticks(np.arange(-.5,4,1),minor=True)
    right.set_yticks(np.arange(-.5,2,1),minor=True)
    right.grid(which='minor',color='#999999',linewidth=.5)
    right.tick_params(which='minor',bottom=False,left=False)
    fig.text(.03,.045,'All five flags concern one 8–17 → 8–5 afternoon conversion; they are not five factual errors.',fontsize=9)
    return save(fig, 'new_numerical_diagnostics'), counts.tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
                         'axes.labelsize':9.5,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.titlecolor':'black','text.color':'black',
                         'axes.labelcolor':'black'})
    primary=ROOT/'experiments/combined_analysis/analysis.json'
    numeric=ROOT/'experiments/numeric_analysis/analysis.json'
    diagnostics=ROOT/'experiments/combined_analysis/response_diagnostics.csv'
    summary=json.loads(primary.read_text());supplement=json.loads(numeric.read_text())
    records=read_csv(diagnostics)
    assert summary['validated_responses']==len(records)==800
    assert summary['query_groups']==100 and supplement['validated_responses']==80
    notes={'scope':'Additional presentation figures generated from completed, frozen artifacts; no new inference or human ratings',
           'plot_script_sha256':file_hash(__file__),
           'source_sha256':{str(p.relative_to(ROOT)):file_hash(p)for p in [primary,numeric,diagnostics,ROOT/'corpus_analysis/split_manifest.json',ROOT/'experiments/api_study_v1/plan.json',ROOT/'experiments/api_study_secondary/plan.json',ROOT/'experiments/api_numeric_diagnostic_run/plan.json']},
           'figures':[]}
    figure=architecture();figure.update({
       'purpose':'Clarifies the actual controlled API experiment architecture, with source-only train embedding and a zero-example path without the production fallback.',
       'caption':'Architecture of the controlled API experiment. Only training source fields are indexed; associated rewrites are joined by stable pair identifiers after example selection. The original held-out input reaches the shared preservation prompt in every condition. This research harness is distinct from the earlier local application.',
       'suggested_insertion':'Section 5.1, after explaining source-only training vectors and prompt construction.',
       'evidence':['experiments/api_study_v1/plan.json','experiments/api_study_secondary/plan.json','experiments/api_numeric_diagnostic_run/plan.json'],
       'note':'Diagram of executed method; it does not add ChromaDB, reranking or a local Llama run to the API study.'})
    notes['figures'].append(figure)
    figure=outcomes(summary);figure.update({
       'purpose':'Makes the eight actual mean outcome cells comparable across source retention and reference agreement.',
       'caption':'Mean output–input and output–associated-rewrite ROUGE-L F1 for the same 100 query groups under each condition and model. The full zero-to-one scale is shown. Associated rewrites are unvalidated references; these overlap scores are not naturalness or factual-accuracy measures.',
       'suggested_insertion':'Section 6.1, after Table 6.',
       'data':[{k:s[k] for k in ['model','condition','n_responses','source_rougeL_f1','reference_rougeL_f1']}for s in summary['summaries']]})
    notes['figures'].append(figure)
    figure,stats=lengths(records);figure.update({
       'purpose':'Shows per-query shortening/expansion that mean length ratios alone conceal.',
       'caption':'Output/input Unicode-token ratios for 100 matched query groups per condition. Boxes show the median and interquartile range; whiskers extend to observations within 1.5 interquartile ranges, with remaining observations plotted individually. The dashed line denotes equal token counts, not preserved meaning.',
       'suggested_insertion':'Section 6.1, after the paragraph comparing mean length ratios.',
       'boxplot_statistics':stats})
    notes['figures'].append(figure)
    figure,counts=numeral_coverage(summary,supplement);figure.update({
       'purpose':'Separates missing numeral coverage in the primary sample from the deliberately targeted supplement, and prevents interpreting token flags as factual errors.',
       'caption':'Source-digit coverage and the separate posthoc numeral diagnostic. The main sample contains no digit-bearing source; all ten eligible test sources are therefore examined separately. Heatmap cells give digit-expression mismatch flags out of ten sources per condition. All five flags concern one working-hours input rewritten from 8–17 to 8–5 with an afternoon marker; they are not five factual errors or a population error estimate.',
       'suggested_insertion':'Section 6.3, after Table 8 and its time-conversion interpretation.',
       'flag_count_matrix':counts,'row_models':list(MODELS),'column_conditions':list(CONDITIONS)})
    notes['figures'].append(figure)
    path=Path(__file__).with_name('figure_notes.json')
    path.write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'added_figures':len(notes['figures']),'metadata':str(path),'outputs':[f['png']for f in notes['figures']]}))


if __name__=='__main__':main()
