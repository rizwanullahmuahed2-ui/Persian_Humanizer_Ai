"""Serialize the assistant's source-relative qualitative inspection of a frozen sample.

This script contains the inspection notes; it does not infer human ratings or call
any language model. Original quoted text is retained verbatim for traceability.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
selection_path=ROOT/'CASE_INSPECTION_SELECTION.json'
selection=json.loads(selection_path.read_text())
assert hashlib.sha256(selection_path.read_bytes()).hexdigest()==(ROOT/'CASE_INSPECTION_SELECTION.sha256').read_text().strip()
plans={name:json.loads((ROOT/name/'plan.json').read_text()) for name in {j['run'] for j in selection['jobs']}}
lookup={name:{j['job_id']:j for j in p['jobs']}for name,p in plans.items()}
MINI='gpt-4.1-mini-2025-04-14'
FOUR='gpt-4o-mini-2024-07-18'
notes=[]
for selected in selection['jobs']:
    plan_job=lookup[selected['run']][selected['job_id']]
    response_path=ROOT/selected['run']/'responses'/(selected['job_id']+'.json')
    raw_path=response_path.with_name(selected['job_id']+'.raw.json')
    result=json.loads(response_path.read_text());assert result['status']=='complete'
    raw=json.loads(raw_path.read_text());assert result['output']==raw['choices'][0]['message']['content']
    source,output=plan_job['input'],result['output']
    q,model,condition=selected['query_id'],selected['model'],selected['condition']
    observed=[];cautions=[];register='در این خوانش موردی، تغییر روشنِ لحن از واژه‌های خروجی مشخص نشد.'
    evidence=[]
    def ev(s,o,reason):
        assert s in source and o in output,(s,o)
        evidence.append({'source_quote':s,'output_quote':o,'interpretation':reason})
    if q=='PH-766d358e592cdfe5767e':
        observed.append('جمله همچنان به انجام مشترکِ کار مربوط به تغییر اشاره دارد.')
        if 'خواهیم کرد' in output:
            observed.append('فعل به صورت صریح آینده درآمده است.')
            cautions.append({'type':'temporal_nuance','strength':'possible_not_established_error','note':'می‌کنیم می‌تواند حال یا برنامهٔ آینده باشد؛ خواهیم کرد تعبیر آینده را صریح می‌سازد. بدون زمینه نمی‌توان آن را خطای قطعی نامید.'})
            ev('مدیریت می‌کنیم','مدیریت خواهیم کرد','صریح‌شدن زمان آینده؛ نه اثبات تغییر واقعیت.')
        elif 'پیش می‌بریم' in output:
            observed.append('مدیریت‌کردن با پیش‌بردن جایگزین شده است.')
            cautions.append({'type':'predicate_nuance','strength':'possible_not_established_error','note':'پیش‌بردن به پیشرفت یا اجرای تغییر نزدیک‌تر است؛ معادل دقیق مدیریت در همهٔ زمینه‌ها نیست.'})
            ev('مدیریت می‌کنیم','پیش می‌بریم','فعل مرتبط عوض شده؛ برابری کامل به زمینه وابسته است.')
        elif 'کنترل' in output:
            observed.append('مدیریت با کنترول جایگزین شده است.')
            cautions.append({'type':'predicate_nuance','strength':'possible_not_established_error','note':'کنترول و مدیریت دامنهٔ معنایی یکسان ندارند؛ این یک احتیاط معنایی است، نه خطای قطعیِ این جملهٔ بی‌زمینه.'})
            ev('مدیریت','کنترل','تغییر فعل/رابطهٔ اصلی.')
        if ' رو ' in output:
            register='صورت «رو» به لحن عامیانه‌تر میل می‌کند؛ ورودی صورت معیار «را» دارد.'
            cautions.append({'type':'register_shift','strength':'direct_surface_observation','note':register})
            ev('را','رو','نشانهٔ روشن تبدیل صورت معیار به عامیانه.')
    elif q=='PH-1684c05b6347a653321f':
        observed.append('وابستگی اعلام عمومی به تأیید نهایی با عبارت بستگی‌داشتن بازنویسی شده است.')
        ev('منوط به تایید نهایی','به تایید نهایی' if 'به تایید نهایی' in output else 'تایید نهایی','شرط تأیید در خروجی حاضر است؛ ترتیب عبارت تغییر کرده است.')
        if 'داره' in output:
            register='فعل «داره» از صورت رسمی جملهٔ ورودی عامیانه‌تر است.'
            cautions.append({'type':'register_shift','strength':'direct_surface_observation','note':register})
            ev('است','داره','تغییر صورت فعل به عامیانه.')
    elif q=='PH-40bdae8e718bc5468db2':
        observed.append('جملهٔ مربوط به تحلیل کمی و بازنمایی زنان با افزودن یا جابه‌جایی تعبیر «در رسانه‌ها» بازنویسی شده است.')
        cautions.append({'type':'source_ambiguity','strength':'undetermined_from_source_alone','note':'ترکیب «زنان رسانه‌ای» می‌تواند زنان شاغل در رسانه را نیز برساند؛ «زنان در رسانه‌ها» تعبیر بازنمایی زنان در رسانه است. ورودی به‌تنهایی برای قطعی‌گرفتن یکی از این دو کافی نیست.'})
        ev('زنان رسانه‌ای','زنان در رسانه‌ها' if 'زنان در رسانه‌ها' in output else 'از زنان در رسانه‌ها','خروجی یکی از تعبیرهای عبارت مبهم را صریح می‌کند.')
        if 'نمایندگی' in output:
            cautions.append({'type':'technical_term_shift','strength':'possible_not_established_error','note':'نمایندگی در بعضی زمینه‌ها معنای نماینده‌بودن دارد و جایگزین بی‌قیدِ بازنمایی نیست؛ این مورد به بررسی موضوعی نیاز دارد.'})
            ev('بازنمایی','نمایندگی','تغییر اصطلاح مربوط به representation.')
        elif 'تصویر' in output:
            observed.append('اصطلاح بازنمایی با تعبیر تصویر یا تصویرسازی جایگزین شده است؛ نزدیکی معنایی به موضوع و تعریف اصطلاح وابسته است.')
    elif q=='PH-f3b2b088dccaeabbc9a1':
        observed.append('فعل اداری اتخاذشدن با گرفته‌شدن ساده شده است.')
        if 'پس از مشورت' in output:
            observed.append('مشاوره با مشورت بیان شده و ترتیب زمانی ضمنی‌تر، صریح شده است.')
            ev('با مشاوره کارشناسان','پس از مشورت با کارشناسان','جمله همچنان نقش مشورت را ذکر می‌کند؛ پس از ترتیب زمانی را صریح می‌کند.')
        elif 'راهنمایی' in output:
            cautions.append({'type':'expert_role_nuance','strength':'possible_not_established_error','note':'راهنمایی و مشاوره نزدیک‌اند، اما نقش هدایت‌کننده می‌تواند از صرف مشورت قوی‌تر باشد.'})
            ev('مشاوره','راهنمایی','نقش کارشناسان با واژهٔ متفاوت توصیف شده است.')
        elif 'همکاری' in output:
            cautions.append({'type':'expert_role_change','strength':'source_relation_not_exactly_preserved','note':'همکاری اعم از مشورت و ممکن است مشارکت در انجام کار باشد؛ ورودی مشخصاً مشاوره را ذکر می‌کند.'})
            ev('مشاوره','همکاری','رابطهٔ مشورت به همکاری عوض شده است.')
        elif 'کمک' in output:
            cautions.append({'type':'expert_role_change','strength':'source_specificity_reduced','note':'کمک نقش کلی‌تری از مشاوره دارد و نوع مشارکت مشخصِ ورودی را صریح نگه نمی‌دارد.'})
            ev('مشاوره','کمک','نوع مشخص مشارکت به تعبیر عمومی‌تر تبدیل شده است.')
        elif 'نظر' in output:
            observed.append('تعبیر «با نظر» به نقش مشورت نزدیک است، ولی شکل دقیق روند مشاوره را بیان نمی‌کند.')
            ev('مشاوره','نظر','تغییر بیان نقش کارشناسان؛ برابری دقیق به زمینه وابسته است.')
    elif q=='PH-70f7edba57adffbb1749':
        observed.append('جمله همچنان دربارهٔ جلسهٔ هیئت مدیره و ترتیب رسیدگی است.')
        if 'فوری' not in output:
            cautions.append({'type':'explicit_qualifier_omission','strength':'direct_surface_observation_with_semantic_consequence','note':'قید صریح فوری حذف شده است. اولویت به‌تنهایی فوریت را تضمین نمی‌کند.'})
            ev('فوری','در اولویت قرار دارد','ارجحیت حفظ شده، ولی الزام زمانیِ صریح ورودی در خروجی نیست.')
        elif 'اولویت' in output:
            cautions.append({'type':'agenda_priority_relation','strength':'possible_not_established_error','note':'دردستورکاربودن و دراولویت‌بودن دقیقاً یک رابطه نیستند؛ فوریت در این خروجی‌ها همچنان صریح است.'})
            ev('دستور کار فوری','اولویت فوری','موضوعِ دستور کار به اولویت تعبیر شده است.')
        else:
            ev('دستور کار فوری','فوری','فوریت به‌صورت صریح نگه داشته شده است.')
    elif q=='PH-8430ee192d142acff80c':
        if 'مقصد نرسید' in output:
            observed.append('رویداد ارسال‌نشدن به نرسیدن به مقصد تبدیل شده است.')
            cautions.append({'type':'event_relation_change','strength':'clear_non_equivalence','note':'پیام می‌تواند ارسال شود ولی به مقصد نرسد. بنابراین، ارسال‌نشدن و نرسیدن به مقصد معادل کامل نیستند؛ نفی در هر دو بودن این تفاوت را رفع نمی‌کند.'})
            ev('ارسال نشد','به دست مقصد نرسید','تغییر رویداد از ارسال به تحویل/رسیدن.')
        else:
            observed.append('عدم ارسال پیام حفظ شده است؛ تفاوت‌ها در زمان فعل، واژهٔ فرستادن یا حذف اشارهٔ این است.')
            ev('ارسال نشد','فرستاده نشد' if 'فرستاده نشد' in output else 'ارسال نشد','نفی همان رویداد ارسال در خروجی حاضر است.')
            if source.startswith('این')and not output.startswith('این'):
                observed.append('واژهٔ اشارهٔ «این» حذف شده است؛ اهمیت آن به زمینهٔ ارجاع بستگی دارد.')
    elif q=='PH-0ba65753f93d3382d459':
        observed.append('نیاز به بازرسی سالانه به دستور باید هر سال بازرسی شوند تبدیل شده است.')
        ev('بازرسی سالانه','هر سال' if 'هر سال' in output else 'سالی' if 'سالی' in output else 'سالانه','دورهٔ سالانه با تعبیر زمانی روشن بیان شده است.')
        if 'یک بار' in output or 'یه بار' in output:
            observed.append('عبارت یک‌بار در سال، یک تعبیر معمول از سالانه را صریح کرده است؛ برای شمار دقیق دفعات باید زمینه بررسی شود.')
        if 'بشن' in output:
            register='فعل «بشن» عامیانه است؛ ورودی صورت رسمی/معیار دارد.'
            cautions.append({'type':'register_shift','strength':'direct_surface_observation','note':register})
            ev('دارند','بشن','تغییر سطح رسمی‌بودن در حالی که تناوب سالانه حفظ شده است.')
        if 'یه' in output:
            observed.append('صورت عامیانهٔ «یه» نیز به خروجی اضافه شده است.')
            ev('سالانه','سالی یه بار','صورت عامیانه در بازنویسیِ ورودی معیار.')
    elif q=='PH-4de4b8271a4807aaaf51':
        observed.append('شرط پس از امضا و نفی قابلیت برهم‌زدن سند در خروجی باقی است.')
        if 'لغو' in output:
            cautions.append({'type':'legal_term_shift','strength':'possible_requires_domain_adjudication','note':'فسخ و لغو در کاربرد حقوقی همیشه یک مفهوم ندارند. این یک احتیاط اصطلاحی نیازمند بررسی متخصص است، نه حکم قطعی حقوقی.'})
            ev('فسخ','لغو','اصطلاح مشخص حقوقی با واژهٔ دیگر جایگزین شده است.')
        else:ev('فسخ ندارد','فسخ نیست','ساختار فعلی ساده‌تر شده و اصطلاح فسخ باقی است.')
        if any(t in output for t in ['رو نداره','دیگه']):
            register='صورت‌هایی مانند رو/نداره یا دیگه، متن حقوقیِ ورودی را عامیانه‌تر می‌کنند.'
            cautions.append({'type':'register_shift','strength':'direct_surface_observation','note':register})
            ev('قابلیت فسخ ندارد','رو نداره' if 'رو نداره' in output else 'دیگه','تغییر سطح رسمی‌بودن با حفظ بخش‌هایی از شرط و نفی.')
    else:raise ValueError(q)
    notes.append({**selected,'source':source,'output':output,'retrieved_ids':plan_job['retrieved_ids'],'retrieved_examples':plan_job['retrieved_examples'],'assessment_provenance':'assistant_model_assisted_qualitative_inspection','human_rating':None,'observed_edits':observed,'cautions':cautions,'register_observation':register,'source_quoted_evidence':evidence,'source_reference_policy':'Judged against input; associated corpus rewrite not assumed gold','saved_response_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),'raw_response_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest()})
result={'scope':'64 model-assisted case notes on8 fixed query groups; not human validation or a population estimate','selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),'independent_query_groups':8,'model_condition_cells_per_query':8,'case_count':len(notes),'human_ratings_created':0,'independent_raters':0,'blinded_human_evaluation':False,'limitations':['Subjective model-assisted inspection; no agreement statistic','Register observations are relative to visible source, not a calibrated dialect or naturalness score','Ambiguous source and technical/legal terminology require domain adjudication','Selection occurred after experiment initiation but before selected response contents were read'],'records':notes}
assert len(notes)==64
(OUT/'notes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'case_notes':len(notes),'independent_queries':8,'human_ratings_created':0}))
