# Generative AI tutor

A chatbot that answers questions about generative AI — transformers, embeddings, LLMs,
diffusion, vision and audio models — **using only a library of textbooks and papers**,
and showing the source of every answer.

Three interfaces onto the same pipeline: a one-shot question in the terminal, a
conversation with memory, and a web app.

```
your question
     │
     ├─ rewritten into a standalone question using the conversation
     ├─ embedded, then matched against 8,291 chunks of 13 documents
     ▼
[ system prompt + the 8 nearest chunks + the conversation + your question ]
     ▼
 LLM on Groq  ──▶  answer + the documents it came from
```

## Quick start

Python 3.10+, about 2 GB of disk for the dependencies (PyTorch) and 60 MB of PDFs.
A free Groq API key: https://console.groq.com

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python download_data.py                                  # fetch the 13 documents
Set-Content .env "GROQ_API_KEY=your_key_here" -Encoding ascii
python rag.py --rebuild                                  # embed them once, ~8 minutes
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_data.py
echo "GROQ_API_KEY=your_key_here" > .env
python rag.py --rebuild
```

Embedding happens once; after that the database is read from `storage/` in a few seconds.
Without activating the venv, call its interpreter directly, e.g.
`.\.venv\Scripts\python.exe app.py`.

Then pick an interface:

| Command | What you get |
|---|---|
| `python rag.py "your question"` | one question, with the retrieved chunks printed |
| `python rag_chat.py` | a conversation in the terminal, `end` to stop |
| `python app.py` | a web app on http://127.0.0.1:7860 |
| `python app.py --share` | the same, plus a temporary public link |

Useful flags: `--rebuild` (re-embed after changing documents or settings),
`--plain` (turn off question rewriting, to compare), `--port 7870`, and for the web app
`--light` / `--dark` (it follows the browser's colour scheme otherwise).

The app is three columns:

```
┌─────────────┬───────────────────────────┬──────────────────────┐
│ Settings    │ Conversation              │ Sources              │
│ top-k       │ answers stream in         │ title · author       │
│ temperature │ example questions         │ page · match bar     │
└─────────────┴───────────────────────────┴──────────────────────┘
```

## The files

| File | What it is |
|---|---|
| `rag.py` | read, split, embed, store, retrieve, answer. All settings live at the top |
| `rag_chat.py` | adds memory and question rewriting; the terminal chat |
| `app.py` | the Gradio web app: streaming answers, a citations panel, live controls |
| `download_data.py` | the list of source documents and how to fetch them |

`rag_chat.py` and `app.py` both import `rag.py`, so `CHUNK_SIZE`, `TOP_K`, `EMBED_MODEL`
and `TEMPERATURE` are configured in one place.

## The library

13 documents, 1,953 pages, 8,291 chunks.

**Textbooks** — Jurafsky & Martin, *Speech and Language Processing*; Simon Prince,
*Understanding Deep Learning*; Xiao & Zhu, *Foundations of Large Language Models*.

**Papers** — ViT, CLIP, LLaVA, Flamingo, DDPM, Latent Diffusion, Whisper, wav2vec 2.0,
a survey of multimodal foundation models, and Weidinger et al. on the risks of harm
from language models.

The PDFs are **not** in this repository. They are free to download and read, but they
are not mine to republish, so `download_data.py` fetches them from the original sources.

## Settings, and why they are what they are

Each of these was measured on this library, not guessed.

| Setting | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 300 | at the usual default of 800, a test question retrieved four chunks that never mentioned the answer and the bot said "I don't know"; at 300 it answered. Long chunks average too many topics into one vector |
| `TOP_K` | 8 | at 4, "How does CLIP connect images and text?" found only CLIP's results and not its method. More documents need a wider net |
| question rewriting | on | a follow-up containing "it" retrieved chunks at 0.24 and failed; rewritten first, the same question retrieved at 0.56 and was answered |
| `TEMPERATURE` | 0.1 | this is a reference tool, not a creative writer |
| model | `openai/gpt-oss-120b` | `llama-3.3-70b-versatile`, which most tutorials still name, has been retired by Groq and now returns 404 |

## Design choices

- **Streaming answers** and a **citations panel** showing the real book title, author
  and page number of every passage the answer was built from, with its match score.
- **Live controls** for how many passages are retrieved (top-k) and for the model's
  temperature, so the two settings that decide answer quality can be tried on the spot.
- **Per-message chat engine.** One shared bot with one memory would put two visitors
  to the same public link into the same conversation. Here the history comes from the
  browser and a fresh engine is built per message — measured at 0.05 ms, against 2.5 s
  for the answer itself.
- **No private attributes.** The slider value is passed to `as_retriever()` rather than
  written into the retriever's internals.
- **The prompt says what to do when the answer is missing.** Without that line the model
  quietly answers from its own memory and you cannot tell retrieval from invention.
- **Sources under every answer**, so any claim can be checked.
- `api_key=` rather than `token=`, which the Groq client silently ignores.

## Limitations

- **English only**, both the documents and the embedding model.
- **Figures and tables are lost** in PDF extraction, so anything explained in a diagram
  is invisible to the bot.
- **The library is academic and mainstream** — textbooks and papers, largely written by
  the labs that build these models. It explains what the technology does far better than
  it questions it. That is a bias chosen by choosing these documents.
- **Similarity scores are relative, not probabilities.** 0.61 does not mean 61% correct.
- It knows these 13 documents and nothing else.

## Possible next steps

Hosting on Hugging Face Spaces for a permanent link, and a proper evaluation set — a
few dozen questions with known answers, to measure how often the right chunk is
actually retrieved instead of judging it by feel.
