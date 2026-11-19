# Evidence · RAG Chatbot & Evaluations

A browser-based document assistant with inspectable sources and real DeepEval
reports. This is a RAG application using pretrained models, not a newly trained
foundation model. No overlay, capture exclusion, click-through, global hotkeys,
microphone access, or interview persona is included.

## What you can demo

- **Chat:** ask typed questions; inspect source passages and similarity scores.
- **Knowledge base:** browse and filter heading-aware Markdown chunks.
- **Evaluations:** view saved retrieval hit rates, DeepEval judge scores, reasons,
  reference answers, model names, run date, and dataset coverage.
- **Fallback:** general-knowledge answers are labeled when no source clears the cutoff.

The included Atlas workspace handbook is **fictional**, written for this demo.
It is not a real product or a set of commitments. No previous personal context,
session logs, interview content, or credentials are included.

## Local setup (Python 3.11 or 3.12)

```powershell
git clone https://github.com/ShubhenduPandey/rag-chatbot-evals.git
cd rag-chatbot-evals
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key"
$env:DEMO_ACCESS_CODE = "choose-a-long-random-passphrase"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `run.bat` after setup and configuring the API key. If no
`DEMO_ACCESS_CODE` environment variable is set, the launcher prints a temporary
random code in its console for you to copy into the browser. It saves no key/code
file and binds the local server only to `127.0.0.1`.
It opens a normal browser-based app; the console remains visible. On macOS/Linux,
use `python3 -m venv .venv` and `.venv/bin/python` instead. `.env.example` is a
reference only; the app does not automatically load `.env` files.

Alternatively create an **ignored** `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key"
DEMO_ACCESS_CODE = "choose-a-long-random-passphrase"
CHAT_MODEL = "gpt-4o"
```

Neither the API key nor the access code is sent as application configuration to
the browser. Visitors enter the access code to unlock paid requests. Browsing
documents and saved reports requires no API key. Conversation state is per browser
session, not shared across users, and is not saved to disk by this app.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. Start with private visibility.
2. Sign in at https://share.streamlit.io and choose **Create app**.
3. Select this repository, branch `main`, entrypoint `app.py`, and Python 3.12.
4. Add the secrets above in Advanced settings. Never commit secrets to GitHub.
5. Deploy, review the app's access settings, and open its returned `streamlit.app` URL.
6. On any other laptop, just open the URL. No clone or Python installation is needed.

Community Cloud requires GitHub authorization. Source-repository visibility and
app visitor access are separate settings: explicitly review both before sharing.
The free hosted filesystem is not durable; the embedding cache can be rebuilt
from the bundled knowledge base. Rebuilding uses paid embedding calls. Keep a
local browser demo and saved report available if hosting/network access fails.

Live chat has a 2,000-character input limit, 800-token answer cap, 30 requests per
browser session, a 3-second cooldown, and 100 requests per server process/hour.
These are demo guards, **not** durable billing limits: restarts and additional
workers reset/multiply counters. Keep hosting access restricted and configure API
usage controls before broad public sharing. Do not expose unrestricted eval runs
on a public button; evaluation is an owner-run command.

## Knowledge and SQLite

Add `.md` files under `knowledge_base/`; subfolders are supported. Files in this
folder are intentionally versioned and visible to demo visitors. **Only put
approved, non-sensitive demo material here.** Keep personal material in the ignored
`private_context/` folder; it is not indexed or served.

`rag.py` retains the original heading-aware chunker and baseline semantic search:
`text-embedding-3-small`, cosine similarity, top 4, threshold 0.35. No reranker or
hybrid search is enabled. SQLite stores JSON vectors in
`.rag_cache/index.sqlite3`, table `embeddings(model, hash, vector)`, keyed by
model plus content hash. Vectors are normalized and reused across local restarts.
Changed chunks are embedded again; deleted chunks are pruned. Query vectors are
temporary. SQLite stores vectors; Python computes cosine scores.

The chunker is shared in origin with the desktop prototype; this repository is
independent and does not import or modify that application. Chat uses GPT-4o by
default. Only selected context and bounded recent conversation are sent for
answers. Chunk/query embeddings and answer requests go to OpenAI; local storage
does not mean offline inference. Failed API requests are displayed as errors,
not silently counted as successful general-knowledge fallbacks.

## Run evaluations

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
# Uses shell OPENAI_API_KEY; unlike the UI this CLI does not read Streamlit secrets.
.\.venv\Scripts\python.exe evaluate.py --deepeval
# Optional smaller paid run:
.\.venv\Scripts\python.exe evaluate.py --deepeval --limit 3
```

`eval_cases.json` contains fixed questions, exact expected section titles, and
reference answers. Top-1/Top-4 are hit rates over relevant questions; unrelated
questions are scored separately for empty retrieval. Every case generates a real
answer. `--deepeval` additionally runs `AnswerRelevancyMetric`, `FaithfulnessMetric`,
and `ContextualRelevancyMetric` using GPT-4o-mini by default (`EVAL_MODEL` override).
Faithfulness/contextual relevancy are skipped when retrieval is empty; this is not
a failure or a fabricated perfect score. Judge scores are not objective guarantees.
The included reference answers are for human inspection; these three metrics do
not measure expected-answer correctness. Contextual precision/recall and a separate
correctness grader would be future additions, not claimed current capabilities.

DeepEval is invoked through local `metric.measure`, without a Confident AI cloud
report upload. DeepEval telemetry is disabled before import. Judge calls still send
the relevant question, answer, and passages to the configured model provider and
incur additional API usage. No judge request is triggered by viewing a report.

Reports save incrementally to `reports/latest.json`. This folder is intentionally
versioned **only for reviewed synthetic demo results** so hosted viewers can see
genuine saved runs without a key. Review before committing; for any private corpus,
change report handling and do not publish it. Reports record corpus/dataset hashes,
models, versions, scores, explanations, and coverage; a changed corpus is flagged
in the UI. Scores from an old or partial run must not be presented as current/full.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_app -v
```

Offline tests check chunking, cache reuse/pruning, cosine retrieval, source/history
prompts, general-knowledge mode, error distinction, limits, evaluation arithmetic,
dataset labels, and Streamlit page/auth flows. No paid API calls are made by tests.

## Demo outline

1. Show the fictional documents in Knowledge base.
2. Ask “I deleted a project yesterday. Can I recover it?” and expand the sources.
3. Ask “What is the capital of France?” and explain the fallback label.
4. Open Evaluations; distinguish deterministic retrieval hits from model-judge scores.
5. Explain the local SQLite cache and the hosted-cache durability limitation.

## References

- https://developers.openai.com/api/docs/guides/embeddings
- https://deepeval.com/docs/getting-started-rag
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/develop/concepts/connections/connecting-to-data
