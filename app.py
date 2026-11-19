"""Evidence: a visible, browser-based RAG and evaluation showcase."""
from dataclasses import asdict
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import time

import streamlit as st

from rag import CONTEXT_DIR, EMBEDDING_MODEL, TOP_K, MIN_SIMILARITY, split_markdown
from service import ChatService, RequestBudget, corpus_fingerprint

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Evidence | RAG & Evaluations', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px; padding:2.8rem 2rem 4rem;}
h1 {font-size:clamp(2rem,4vw,3.2rem) !important; letter-spacing:-.045em; font-weight:650 !important;}
h2 {letter-spacing:-.025em;}
[data-testid="stSidebar"] {border-right:1px solid #293448;}
[data-testid="stMetric"] {background:#1A2331; padding:1rem; border:1px solid #2C394D; border-radius:12px;}
[data-testid="stChatMessage"] {border:1px solid #2C394D; border-radius:14px; margin-bottom:1rem;}
.eyebrow {color:#71E3BF; letter-spacing:.16em; font-size:.85rem; font-weight:700;}
.intro {font-size:1.15rem; color:#B9C7D9; max-width:780px; line-height:1.7;}
</style>''', unsafe_allow_html=True)


def setting(name, default=''):
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


@st.cache_resource
def services(api_key, model):
    from openai import OpenAI
    return ChatService(OpenAI(api_key=api_key, timeout=30, max_retries=1), model=model)


@st.cache_resource
def shared_budget():
    return RequestBudget()


def catalog():
    chunks = []
    for path in sorted(CONTEXT_DIR.rglob('*')):
        if (path.is_file() and path.suffix.lower() == '.md' and not path.is_symlink()
                and path.resolve().is_relative_to(CONTEXT_DIR.resolve())):
            chunks.extend(split_markdown(path.read_text(encoding='utf-8-sig'),
                                         path.relative_to(CONTEXT_DIR).as_posix()))
    return chunks


def show_sources(sources):
    if not sources:
        st.caption('No source passed the retrieval cutoff. General-knowledge mode; no source citations.')
    for source in sources:
        with st.expander(f"[{source['label']}] {source['heading'].split(' > ')[-1]} · {source['score']:.3f}"):
            st.caption(f"{source['file']} · part {source['part']} · cosine similarity, not confidence")
            st.markdown(source['text'])


with st.sidebar:
    st.markdown('## ◈ Evidence')
    st.caption('RAG CHATBOT · EVALUATION LAB')
    page = st.radio('Explore', ['Chat', 'Knowledge base', 'Evaluations'], label_visibility='collapsed')
    st.divider()
    st.markdown('**Inspectable by design**')
    st.caption('Ask a question. See the evidence. Measure the answer.')
    st.caption('Python · OpenAI · SQLite · DeepEval')
    st.divider()
    st.caption('Sample data: a fictional Atlas product handbook. No personal or customer data.')
    st.caption('API calls run on the server. Chat history is session-only and is not saved to disk.')

if page == 'Chat':
    st.markdown('<div class="eyebrow">DOCUMENTS INTO ANSWERS</div>', unsafe_allow_html=True)
    st.title('Answers with evidence.')
    st.markdown('<p class="intro">A document assistant you can inspect. Ask about the sample handbook, '
                'then open the passages behind each answer.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric('Sections', len(catalog()))
    c2.metric('Top-K', TOP_K)
    c3.metric('Cutoff', str(MIN_SIMILARITY))
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('question_count', 0)
    st.session_state.setdefault('last_request', 0.0)
    api_key, access_code = setting('OPENAI_API_KEY'), setting('DEMO_ACCESS_CODE')
    unlocked = st.session_state.get('unlocked', False) and bool(access_code)
    if not api_key:
        st.info('Live chat is not configured yet. Explore the knowledge base and saved evaluation results.')
    elif not access_code:
        st.info('Live chat is locked until the host configures DEMO_ACCESS_CODE. No paid calls are available.')
    elif not unlocked:
        with st.form('unlock', clear_on_submit=True):
            entered = st.text_input('Demo access code', type='password')
            submitted = st.form_submit_button('Unlock live chat')
        if submitted:
            if hmac.compare_digest(entered.encode(), access_code.encode()):
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error('The access code did not match.')
    with st.expander('Try a question'):
        st.markdown('- Can an editor change billing?\n- I deleted a project yesterday. Can I recover it?\n'
                    '- Does cancelling immediately remove paid features?\n- What is the capital of France?')
    if st.session_state.messages and st.button('Clear conversation'):
        st.session_state.messages = []
        st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'answer' in message:
                a = message['answer']
                st.caption(f"{a['mode'].capitalize()} · {a['seconds']}s · {a['model']}")
                show_sources(a['sources'])
                with st.expander('Retrieval query'):
                    st.code(a['retrieval_query'], language=None)
    question = st.chat_input('Ask about the handbook…', disabled=not (api_key and unlocked), max_chars=2000)
    if question:
        if st.session_state.question_count >= 30:
            st.warning('This demo session has reached its 30-question limit.')
        elif time.monotonic() - st.session_state.last_request < 3:
            st.warning('Please wait a few seconds before sending another question.')
        elif not shared_budget().reserve():
            st.warning('The server demo budget is exhausted for this hour. Please try later.')
        else:
            st.session_state.question_count += 1
            st.session_state.last_request = time.monotonic()
            try:
                with st.spinner('Finding sources and composing an answer…'):
                    answer = services(api_key, setting('CHAT_MODEL', 'gpt-4o')).ask(
                        question, st.session_state.messages)
                st.session_state.messages.extend([
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer.text, 'answer': asdict(answer)}])
                st.rerun()
            except Exception:
                st.error('The request could not be completed. Check the server API key, quota, and connectivity. '
                         'This is a service error, not a no-source result. No answer has been saved.')

elif page == 'Knowledge base':
    st.markdown('<div class="eyebrow">THE SOURCE OF TRUTH</div>', unsafe_allow_html=True)
    st.title('Explore the knowledge.')
    st.write('These purpose-written fictional documents are the only material included in this demo.')
    chunks = catalog()
    st.caption(f'{len(chunks)} heading-aware chunks · {EMBEDDING_MODEL} · local SQLite embedding cache')
    search = st.text_input('Filter sections', placeholder='Try: recovery, billing, notifications')
    matching = [c for c in chunks if search.lower() in c.text.lower()]
    if not matching:
        st.info('No matching sections.')
    for chunk in matching:
        with st.expander(chunk.heading.split(' > ')[-1]):
            st.caption(f'{chunk.source} · part {chunk.part} · {len(chunk.text)} characters')
            st.markdown(chunk.text)
    st.divider()
    st.markdown('**How retrieval works**')
    st.write('Markdown headings define chunks. OpenAI embeds each chunk. SQLite caches the numeric vectors by '
             'model and content hash. A question is embedded and ranked by cosine similarity. Only matching '
             'passages reach the answer model. Unchanged chunks reuse their stored embeddings.')

else:
    st.markdown('<div class="eyebrow">MEASURE, DON’T GUESS</div>', unsafe_allow_html=True)
    st.title('Evaluation lab.')
    st.write('Retrieval tests measure the sources. DeepEval measures the generated answers. '
             'Scores below come from saved runs, not from a live evaluation on page load.')
    report_path = ROOT / 'reports' / 'latest.json'
    if not report_path.exists():
        st.info('No evaluation has been run yet. No demonstration scores are fabricated.')
        st.code('python evaluate.py --deepeval', language='bash')
    else:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['corpus_sha256'] != corpus_fingerprint():
            st.warning('The knowledge base has changed since this report. These are historical results.')
        if report['dataset_sha256'] != sha256((ROOT / 'eval_cases.json').read_bytes()).hexdigest():
            st.warning('Evaluation questions or reference labels have changed since this report.')
        st.caption(f"Run: {report['timestamp']} · Answer model: {report['answer_model']} · "
                   f"Judge: {report['judge_model'] or 'not run'} · {report['evaluated_cases']}/{report['dataset_cases']} cases")
        stats = report['summary']
        if stats['errors'] or stats['metric_errors']:
            st.warning(f"Run errors: {stats['errors']} answer requests; {stats['metric_errors']} judge checks. "
                       'Metric means below exclude unscored judge errors; inspect the individual cases.')
        cols = st.columns(3)
        for col, key, label in zip(cols, ['top1', 'top4', 'fallback'], ['Top-1 retrieval', 'Top-4 retrieval', 'Correct fallback']):
            item = stats[key]
            col.metric(label, f"{item['hits']}/{item['total']}" if item['total'] else 'Not evaluated')
        st.caption('Retrieval hit = any explicitly expected section selected. Unrelated cases are counted only in fallback. '
                   'The unchanged retrieval cutoff is 0.35. Errors count as failures, not passes.')
        if report['deepeval_enabled']:
            st.subheader('DeepEval · answer quality')
            for name, item in stats['deepeval'].items():
                st.write(f"**{name}** — mean {item['mean']:.3f} across {item['count']} scored cases")
            st.caption('Judge scores are model assessments, not proof of correctness. Faithfulness and contextual '
                       'relevancy are not applied to no-source answers. Reference answers and judge reasons appear below.')
        else:
            st.info('This report contains retrieval/answer checks only. DeepEval has not run.')
        for case in report['cases']:
            with st.expander(f"{case['id']} · {case['question']}"):
                st.caption(f"Expected: {', '.join(case['expected_headings']) or 'no source / fallback'}")
                if case.get('error'):
                    st.error(case['error'])
                else:
                    st.markdown(case['answer']['text'])
                    st.caption('Reference answer: ' + case['expected_output'])
                    show_sources(case['answer']['sources'])
                for metric in case.get('metrics', []):
                    st.write(f"**{metric['name']}** · {metric.get('score', 'not scored')}")
                    st.caption(metric.get('reason') or metric.get('error') or '')
        st.download_button('Download evaluation report', report_path.read_bytes(),
                           file_name='evidence-evaluation.json', mime='application/json')
