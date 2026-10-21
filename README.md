# Persian Humanizer

A web app for rewriting formal Persian text into something easier to read. You paste a short passage, choose a tone, and watch the rewrite appear as it is generated.

The project explores Persian and Dari rewriting through a combination of example retrieval and language-model generation. It keeps the original text beside the output so you can compare them and decide whether the wording still says what you meant.

The default model is **Llama 3.1 8B through Ollama**. The backend also supports **GPT-4.1 mini** through the OpenAI API.

![Right-to-left rewriting interface](docs/interface.png)

## What you can do

- Rewrite passages of up to 2,000 characters.
- Choose casual, friendly, or formal wording in the browser.
- Follow retrieval and generation progress while the response streams.
- Select a configured local or OpenAI model, stop a request, copy or download a completed rewrite.
- Call the backend directly for a complete JSON response or a streamed response.

The browser interface is built with HTML, CSS, and JavaScript. It uses a right-to-left layout and includes responsive styles; there is no React app or frontend build step.

## How a rewrite is made

The backend first searches a dataset of paired texts. Each row has an input passage and a reference rewrite. Chroma stores embeddings of the input passages, while the reference rewrites stay in document metadata.

Retrieval looks for up to 20 candidates, preferring examples with the requested style. A cross-encoder then selects up to four examples. If the reranker cannot load, the system uses the initial retrieval order.

The prompt uses at most two of those pairs, and only when both sides are 80 characters or shorter. If none qualify, it uses a built-in fallback example. The selected model receives those examples, the requested style, and the user's text. After generation, the backend removes common labels, extra explanations, and some repeated alternatives.

This is retrieval-guided prompting. The code does not fine-tune a model on the dataset.

### A pair from the dataset

**Input**

> متأسفانه امکان شرکت در جلسه مذکور برای اینجانب میسر نخواهد بود.

**Reference rewrite**

> ببخشید نمی‌تونم بیام جلسه.

This is a stored example, not a recorded live response. Generated wording depends on the model, style, and retrieved examples.

## Main components

| Part | Implementation |
| --- | --- |
| Browser interface | HTML, CSS, vanilla JavaScript |
| API | FastAPI and Uvicorn |
| Rewrite workflow | LangGraph |
| Example retrieval | LangChain integrations and Chroma |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Local generation | Ollama with `llama3.1:8b` |
| Optional hosted generation | OpenAI with `gpt-4.1-mini` |
| Evaluation | Python rules and lexical comparisons |

## Run locally

You need **Python 3.13 or newer**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Ollama](https://ollama.com/). These steps use the project's `pyproject.toml` and `uv.lock`.

### 1. Get the project and install dependencies

```bash
git clone https://github.com/rizwanullahmuahed2-ui/Persian_Humanizer_Ai.git
cd Persian_Humanizer_Ai
uv sync --locked
```

### 2. Prepare the local model

```bash
ollama pull llama3.1:8b
```

Make sure Ollama is running. If it is not already running through the desktop app or a background service, start it in a separate terminal:

```bash
ollama serve
```

The first use of retrieval may also download the embedding and reranking models. Those models run on the CPU in the current implementation. Startup and generation speed depend on your machine, so allow time for downloads and model loading.

### 3. Start the backend

From the project root:

```bash
uv run --locked python backend/api.py
```

The API runs at [http://localhost:8000](http://localhost:8000). Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Open the interface

Visit **http://127.0.0.1:8000/ui/**. FastAPI serves the frontend alongside the API, so a second server is unnecessary. The API root and `/docs` remain available. A separate static preview on port 5500 also works; `frontrend/app.js` sends those requests to `127.0.0.1:8000`.

The historical folder name `frontrend` is retained so existing local scripts keep working. Models load lazily on the first rewrite rather than when opening the interface. A running API is not proof that Ollama, model downloads or hosted inference are ready.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Report service metadata, configured model options, and index-folder status |
| POST | `/humanize` | Return the rewrite and evaluation as JSON |
| POST | `/humanize/stream` | Stream progress, text, and evaluation using server-sent events |
| POST | `/rebuild-index` | Local maintenance only; disabled unless `HUMANIZER_ENABLE_REBUILD=1` |

Try this request in the interactive API docs:

```json
{
  "text": "لطفاً نسبت به ارسال مستندات مربوطه در اسرع وقت اقدام فرمایید.",
  "style": "casual",
  "model": "llama3.1:8b"
}
```

The JSON response includes `original`, `humanized`, `style`, `model`, `evaluation`, `examples_used`, and `error`. The example count describes retrieved candidates, not necessarily the number of pairs included in the prompt.

The streaming endpoint emits named payloads such as `__RETRIEVING__`, `__GENERATING__`, `__TOKEN__`, `__TEXT__`, `__EVAL__`, `__DONE__`, and `__ERROR__`. The final `__TEXT__` payload contains the cleaned rewrite.

### Using GPT instead

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=
```

Restart the backend, then send a request with `"model": "gpt-4.1-mini"`. The key stays on the backend; `.env` is excluded by the repository's `.gitignore`. This option makes requests to OpenAI and incurs API usage charges. The request includes the input text and the examples placed in the prompt.

The browser model selector enables GPT when the backend reports a configured key. This checks configuration only, not billing, model access or provider connectivity. The browser never receives the key.

## Dataset and evaluation

[data/persian_humanizer_dataset.csv](data/persian_humanizer_dataset.csv) currently contains **3,511 paired examples**, with these columns:

- `ai`: the original passage used for retrieval.
- `human`: the reference rewrite.
- `style`: the requested tone.
- `transformation`: the type of rewrite.
- `notes`: additional information about the pair.

The browser offers three styles, while the dataset contains more. The column names are dataset labels; they do not establish how each text was authored.

To create a separate evaluation split and generate outputs for review, run these commands from the project root:

```bash
uv run --locked python -m eval.build_test_set
uv run --locked python -m eval.run_evaluation
```

The split builder targets 200 test rows using seed 42 and groups by transformation. The evaluation runner uses `data/train_set.csv` for retrieval and processes the first 30 test rows by default. It writes:

- `eval/human_eval_sheet.csv`: original text, generated text, reference rewrite, and blank columns for human ratings.
- `eval/auto_metrics.csv`: automatic scores for comparison.

The runner uses Ollama by default. After changing the dataset or split, rebuild the corresponding index instead of reusing an index built from earlier data.

Two additional scripts compare embedding models and indexed text strategies:

```bash
uv run --locked python -m eval.ab_test_retrieval
uv run --locked python -m eval.ab_test_content_strategy
```

These scripts measure style and transformation matches among retrieved examples. They do not establish the quality of the final rewrite.

## Current limitations

Meaning preservation is a goal of the prompt, not a guarantee. Read the result before using it, especially when the source includes numbers, names, or technical details. Dari wording also varies: the prompt is broadly Persian, and the dataset includes colloquial Iranian Persian examples.

The automatic scores use word overlap, formal or colloquial markers, and length comparisons. They are useful for inspecting behavior, but a high score does not prove that a rewrite is accurate or natural.

The browser deliberately omits quality-rating bars: streaming and non-streaming paths use different lexical evaluators, neither calibrated as a reliable quality score. Raw evaluation remains available to API clients. Token and final-text payloads are decoded as JSON so Persian text, spaces and line breaks survive streaming. Partial or interrupted output is labelled incomplete and cannot be exported as a completed rewrite.

This remains a local development app without authentication or production rate limiting. CORS defaults to the local port-5500 origins; the interface normally uses the same origin. Index rebuilding requires an explicit maintenance flag. HTTP errors omit raw provider details and generation excerpts are not printed. Do not expose it publicly without deployment hardening.

A successful `/health` response is not an inference test. Its index status checks whether files exist, and the local model is listed as configured without checking Ollama availability.

## Repository layout

```text
backend/
  api.py                  FastAPI endpoints
  humanizer_graph.py      Retrieval, generation, cleanup, and streaming
  model_provider.py       Ollama model selection and the OpenAI client
  rag_engine.py           Dataset loading, embeddings, retrieval, and reranking
  Evaluator.py            Lexical evaluation used by streaming
  requirements.txt        Alternative dependency list
frontrend/
  index.html              Accessible RTL interface
  styles.css              Responsive layout
  app.js                  Model selection and request state
  stream.mjs              Incremental SSE decoding
data/
  persian_humanizer_dataset.csv
  train_set.csv
  chroma_db/              Earlier stored index
  chroma_db_v2/           Default stored index
eval/
  build_test_set.py
  run_evaluation.py
  dari_benchmark.py
  ab_test_retrieval.py
  ab_test_content_strategy.py
pyproject.toml
uv.lock
```

---

Maintained by [Rizwanullah Muahed](https://github.com/rizwanullahmuahed2-ui).


## Checks and research

Run the lightweight checks without neural downloads or paid model calls:

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
node --test tests/stream.test.mjs
```

These tests replace generation with controlled functions. They verify request validation, model selection, safe failures, disabled maintenance and fragmented UTF-8 streaming; they do not establish rewrite quality or a successful live provider call. Full generation additionally needs the application dependencies and either a working Ollama model or a valid funded OpenAI configuration.

[Research manuscript status](research/README.md): university-approved and unpublished, as reported by the owner. The actual manuscript is pending; no PDF or publication details have been invented.
